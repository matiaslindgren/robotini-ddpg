"""
Interface between the TensorFlow Agents framework and the Robotini racing environment.
Allows TensorFlow agents to interact with the race car.
"""
import logging
import time
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

from redis import Redis
import numpy as np
from scipy.spatial.distance import minkowski
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from robotini_ddpg.util import sleep_until
from robotini_ddpg.model import features
from robotini_ddpg.simulator import camera, log_parser


fps_limit = 60
num_laps_per_episode = 1


class RobotiniCarEnv(py_environment.PyEnvironment):

    def __init__(self, manager, env_id, redis_socket_path):
        self.manager = manager
        self.env_id = env_id
        self.redis = Redis(unix_socket_path=redis_socket_path)
        self.step_interval_sec = 1.0/fps_limit
        # Continuous action of 2 values: [forward, turn]
        self._action_spec = array_spec.BoundedArraySpec(
                shape=[2],
                dtype=np.float32,
                minimum=[0.1, -0.5],
                maximum=[0.3, 0.5],
                name="forward_and_turn")
        # Neural network inputs
        self._observation_spec = array_spec.BoundedArraySpec(
                shape=features.observation_shape,
                dtype=np.float32,
                minimum=0,
                maximum=1,
                name="observation")
        self._episode_ended = False
        # Global state (within this environment) that persists across episodes
        self.global_state = {
            "episode": 0,
            "lap_count": 0,
            "crash_count": 0,
            "track_segment": 0,
            "prev_frames": [np.zeros(camera.frame_shape)],
            "prev_sim_state": log_parser.to_numpy(log_parser.empty_state()),
        }
        # State that persists across steps but not episodes
        self.episode_state = self.zero_state()

    def zero_state(self):
        g = self.global_state
        return {
            "step_num": 0,
            "episode": g["episode"],
            "track_segment": g["track_segment"],
            "distance": 0,
            "crash_count": 0,
            "init_crash_count": g["crash_count"],
            "return": 0,
            "lap_count": 0,
            "init_lap_count": g["lap_count"],
            "next_step_time": time.perf_counter(),
            "prev_frames": g["prev_frames"],
            "prev_sim_state": g["prev_sim_state"],
        }

    def empty_observation(self):
        return np.zeros(self._observation_spec.shape, np.float32)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_state_snapshot(self, sim_state):
        s = self.episode_state
        g = self.global_state
        x_img, y_img = features.observation_to_xy_images(s["observation"])
        return {
            "env_id": self.env_id,
            "episode": g["episode"],
            "original_frame": camera.frame_to_base64(s["original_frame"]),
            "observation_x": camera.frame_to_base64(x_img),
            "observation_y": camera.frame_to_base64(y_img),
            "step_num": int(s["step_num"]),
            "speed": float(s["speed"]),
            "action": s["action"].tolist(),
            "return": float(s["return"]),
            "distance": float(s["distance"]),
            "position": sim_state["position"].tolist(),
            "rotation": sim_state["rotation"].tolist(),
            "track_angle": sim_state["track_angle"],
            "track_segment": sim_state["track_segment"],
            "lap_time": sim_state["lap_time"],
            "lap_count": s["lap_count"],
            "total_lap_count": g["lap_count"],
            "total_crash_count": g["crash_count"],
            "crash_count": s["crash_count"],
        }

    def write_state_snapshot(self, data):
        state_json = json.dumps(data).encode("utf-8")
        self.redis.hset(self.env_id, "state_snapshot.json", state_json)

    def _reset(self):
        self._episode_ended = False
        self.episode_state = self.zero_state()
        return ts.restart(self.empty_observation())

    def do_action(self, action):
        self.manager.send_action(self.env_id, "forward", float(action[0]))
        self.manager.send_action(self.env_id, "turn", float(action[1]))

    def terminate(self, observation, **ts_kwargs):
        self._episode_ended = True
        for k in ("prev_frames", "prev_sim_state", "track_segment"):
            self.global_state[k] = self.episode_state[k]
        self.episode_state = self.zero_state()
        return ts.termination(observation, **ts_kwargs)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        s = self.episode_state
        g = self.global_state
        s["step_num"] += 1
        s["next_step_time"] += self.step_interval_sec

        if s["step_num"] == 1:
            g["episode"] += 1
            logger.info("'%s' - begin episode %d", self.env_id, g["episode"])

        # Read car state and do action
        frames, sim_state = self.manager.get_car_state(self.env_id)
        self.do_action(action)

        # If camera frames are empty, use previous values
        frames = s["prev_frames"] = (frames or s["prev_frames"])
        sim_state = s["prev_sim_state"] = (sim_state or s["prev_sim_state"])

        # Feature extraction
        frame, observation = features.camera_frames_to_observation(frames)

        # Update state and compute reward
        s["speed"] = np.linalg.norm(sim_state["velocity"])
        s["action"] = action
        s["observation"] = observation
        s["original_frame"] = frame
        s["distance"] += 0 if "position" not in s else minkowski(s["position"], sim_state["position"], 2)
        s["position"] = sim_state["position"]
        reward = features.reward(s, g, sim_state)

        # Write state snapshot to global cache (Redis)
        s["return"] += reward
        s["lap_count"] = sim_state["lap_count"] - s["init_lap_count"]
        s["crash_count"] = sim_state["crash_count"] - s["init_crash_count"]
        g["lap_count"] = sim_state["lap_count"]
        g["crash_count"] = sim_state["crash_count"]
        s["track_segment"] = sim_state["track_segment"]
        self.write_state_snapshot(self.get_state_snapshot(sim_state))

        if s["lap_count"] >= num_laps_per_episode:
            logger.info("'%s' - end episode at step %d: completed %d laps successfully",
                    self.env_id, s["step_num"], s["lap_count"])
            return self.terminate(observation, reward=reward)

        sleep_until(s["next_step_time"])
        return ts.transition(observation, reward=reward)
