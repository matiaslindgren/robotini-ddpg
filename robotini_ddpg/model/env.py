"""
Interface between the TensorFlow Agents framework and the Robotini racing environment.
Allows TensorFlow agents to learn about and interact with the race car.
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


class RobotiniCarEnv(py_environment.PyEnvironment):

    def __init__(self, manager, env_id, redis_socket_path, fps_limit=60):
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
                shape=features.input_shape,
                dtype=np.float32,
                minimum=0,
                maximum=1,
                name="observation")
        self._state = self.zero_state()
        self._episode_ended = False

    @staticmethod
    def zero_state():
        return {
            "step_num": 0,
            "distance": 0,
            "return": 0,
            "next_step_time": time.perf_counter(),
            "prev_frames": [np.zeros(camera.frame_shape)],
            "prev_sim_state": log_parser.to_numpy(log_parser.empty_state()),
        }

    def empty_observation(self):
        return np.zeros(self._observation_spec.shape, np.float32)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_state_snapshot(self, sim_state):
        s = self._state
        x_img, y_img = features.observation_to_xy_images(s["observation"])
        return {
            "env_id": self.env_id,
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
            "track_angle": float(sim_state["track_angle"]),
            "track_segment": int(sim_state["track_segment"]),
        }

    def write_state_snapshot(self, data):
        state_json = json.dumps(data).encode("utf-8")
        self.redis.hset(self.env_id, "state_snapshot.json", state_json)

    def _reset(self):
        self._episode_ended = False
        self._state = self.zero_state()
        return ts.restart(self.empty_observation())

    def do_action(self, action):
        self.manager.send_action(self.env_id, "forward", float(action[0]))
        self.manager.send_action(self.env_id, "turn", float(action[1]))

    def terminate(self, observation, **ts_kwargs):
        self._episode_ended = True
        self._state = self.zero_state()
        return ts.termination(observation, **ts_kwargs)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        s = self._state
        s["step_num"] += 1
        s["next_step_time"] += self.step_interval_sec

        if s["step_num"] == 1:
            logger.info("'%s' - starting new episode", self.env_id)

        frames, sim_state = self.manager.get_car_state(self.env_id)
        self.do_action(action)
        if frames:
            s["prev_frames"] = frames
        elif "prev_frames" not in s:
            sleep_until(s["next_step_time"])
            return ts.transition(self.empty_observation(), reward=0)
        else:
            frames = s["prev_frames"]
        if sim_state:
            s["prev_sim_state"] = sim_state
        elif "prev_sim_state" not in s:
            sleep_until(s["next_step_time"])
            return ts.transition(self.empty_observation(), reward=0)
        else:
            sim_state = s["prev_sim_state"]
        frame, observation = features.camera_frames_to_observation(frames)

        s["speed"] = np.linalg.norm(sim_state["velocity"])
        s["action"] = action
        s["observation"] = observation
        s["original_frame"] = frame
        s["distance"] += 0 if "position" not in s else minkowski(s["position"], sim_state["position"], 2)
        s["position"] = sim_state["position"]
        reward = features.reward(s, sim_state)
        s["return"] += reward
        self.write_state_snapshot(self.get_state_snapshot(sim_state))

        if s["step_num"] > 1000:
            logger.info("'%s' - terminating episode", self.env_id)
            return self.terminate(observation, reward=0)

        sleep_until(s["next_step_time"])
        return ts.transition(observation, reward=reward)
