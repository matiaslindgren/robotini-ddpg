"""
Interface between the TensorFlow Agents framework and the Robotini racing environment.
Allows TensorFlow agents to interact with the race car.
"""
import logging
import time
import json

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
                minimum=[0.001, -0.5],
                maximum=[0.4, 0.5],
                name="forward_and_turn")
        # Neural network inputs
        self._observation_spec = array_spec.BoundedArraySpec(
                shape=features.observation_shape,
                dtype=np.float32,
                minimum=0,
                maximum=1,
                name="observation")
        self._episode_ended = False
        # Epoch state that persists across episodes for this environment, i.e. car
        self.epoch_state = {
            "episode": 0,
            "lap_count": 0,
            "crash_count": 0,
            "track_segment": 0,
            "simulator_state": log_parser.to_numpy(log_parser.empty_state()),
        }
        # State that persists across steps but not episodes
        self.episode_state = self.zero_state()

    def zero_state(self):
        epoch = self.epoch_state
        return {
            "step_num": 0,
            "episode": epoch["episode"],
            "track_segment": epoch["track_segment"],
            "distance": 0,
            "crash_count": 0,
            "init_crash_count": epoch["crash_count"],
            "return": 0,
            "lap_count": 0,
            "init_lap_count": epoch["lap_count"],
            "next_step_time": time.perf_counter(),
        }

    def empty_observation(self):
        return np.zeros(self._observation_spec.shape, np.float32)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_state_snapshot(self, simulator_state):
        episode = self.episode_state
        epoch = self.epoch_state
        sim = simulator_state
        x_img, y_img = features.observation_to_xy_images(episode["observation"])
        return {
            "env_id": self.env_id,
            "episode": epoch["episode"],
            "original_frame": camera.frame_to_base64(episode["original_frame"]),
            "observation_x": camera.frame_to_base64(x_img),
            "observation_y": camera.frame_to_base64(y_img),
            "step_num": int(episode["step_num"]),
            "speed": float(episode["speed"]),
            "action": episode["action"].tolist(),
            "return": float(episode["return"]),
            "distance": float(episode["distance"]),
            "position": sim["position"].tolist(),
            "rotation": sim["rotation"].tolist(),
            "track_angle": sim["track_angle"],
            "track_segment": sim["track_segment"],
            "lap_time": sim["lap_time"],
            "lap_count": episode["lap_count"],
            "total_lap_count": epoch["lap_count"],
            "total_crash_count": epoch["crash_count"],
            "crash_count": episode["crash_count"],
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
        self.episode_state = self.zero_state()
        return ts.termination(observation, **ts_kwargs)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        episode = self.episode_state
        epoch = self.epoch_state
        episode["step_num"] += 1
        episode["next_step_time"] += self.step_interval_sec

        if episode["step_num"] == 1:
            epoch["episode"] += 1
            logging.info("'%s' - begin episode %d", self.env_id, epoch["episode"])

        # Read car state and do action
        frames, sim_state = self.manager.get_car_state(self.env_id)
        self.do_action(action)

        # Use simulator state from previous step if got empty state
        sim_state = epoch["simulator_state"] = (sim_state or epoch["simulator_state"])

        # Extract model inputs from camera frame buffer
        frame, observation = features.camera_frames_to_observation(frames)

        # Update state and compute reward for this step
        episode["speed"] = np.linalg.norm(sim_state["velocity"])
        episode["action"] = action
        episode["observation"] = observation
        episode["original_frame"] = frame
        if "position" in episode:
            episode["distance"] += minkowski(episode["position"], sim_state["position"], 2)
        episode["position"] = sim_state["position"]
        reward = features.reward(episode, epoch, sim_state)

        # Update state for computing reward at next step
        episode["return"] += reward
        episode["lap_count"] = sim_state["lap_count"] - episode["init_lap_count"]
        episode["crash_count"] = sim_state["crash_count"] - episode["init_crash_count"]
        epoch["lap_count"] = sim_state["lap_count"]
        epoch["crash_count"] = sim_state["crash_count"]
        episode["track_segment"] = sim_state["track_segment"]

        # Write JSON snapshot of current state into Redis
        self.write_state_snapshot(self.get_state_snapshot(sim_state))

        # Terminate episode if car did enough laps
        if episode["lap_count"] >= num_laps_per_episode:
            logging.info("'%s' - end episode at step %d after %d completed laps",
                    self.env_id, episode["step_num"], episode["lap_count"])
            return self.terminate(observation, reward=reward)

        # Still going, throttle FPS and transition to next step
        sleep_until(episode["next_step_time"])
        return ts.transition(observation, reward=reward)
