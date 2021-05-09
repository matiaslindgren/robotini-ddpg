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
from tf_agents.environments import tf_py_environment, batched_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from robotini_ddpg import features, util
from robotini_ddpg.simulator import camera, log_parser, manager


class RobotiniCarEnv(py_environment.PyEnvironment):

    def __init__(self, env_id, redis_socket_path, laps_per_episode, max_env_steps_per_episode, forward_range, turn_range, fps_limit=60):
        self.manager = None
        self.env_id = env_id
        self.redis = Redis(unix_socket_path=redis_socket_path)
        self.step_interval_sec = 1.0/fps_limit
        self.laps_per_episode = laps_per_episode
        self.max_env_steps_per_episode = max_env_steps_per_episode
        # Continuous action of 2 values: [forward, turn]
        self._action_spec = array_spec.BoundedArraySpec(
                shape=[2],
                dtype=np.float32,
                minimum=[forward_range[0], turn_range[0]],
                maximum=[forward_range[1], turn_range[1]],
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
            "return_to_track_count": 0,
            "track_segment": 0,
            "frames": [np.zeros(camera.frame_shape)],
            "simulator_state": log_parser.to_numpy(log_parser.empty_state()),
        }
        # State that persists across steps but not episodes
        self.episode_state = self.zero_state()

    def zero_state(self):
        epoch = self.epoch_state
        now = time.perf_counter()
        return {
            "env_step": 0,
            "env_step_clock": now,
            "episode": epoch["episode"],
            "track_segment": epoch["track_segment"],
            "track_segment_time": 0,
            "track_segment_begin_clock": now,
            "distance": 0,
            "crash_count": 0,
            "return": 0,
            "lap_count": 0,
            "next_env_step_time": now,
        }

    def empty_observation(self):
        return np.zeros(self._observation_spec.shape, np.float32)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_car_state(self):
        return self.manager.get_car_state(self.env_id)

    def get_state_snapshot(self, simulator_state):
        episode = self.episode_state
        epoch = self.epoch_state
        sim = simulator_state
        x_img, y_img = features.observation_to_xy_images(episode["observation"])
        return {
            "team_id": self.env_id,
            "car_id": self.manager.get_car(self.env_id).car_id,
            "car_color": self.manager.teams[self.env_id].color,
            "episode": epoch["episode"],
            "original_frame": camera.frame_to_base64(episode["original_frame"]),
            "observation_x": camera.frame_to_base64(x_img),
            "observation_y": camera.frame_to_base64(y_img),
            "env_step": int(episode["env_step"]),
            "speed": float(episode["speed"]),
            "action": episode["action"].tolist(),
            "return": float(episode["return"]),
            "distance": float(episode["distance"]),
            "position": sim["position"].tolist(),
            # "rotation": sim["rotation"].tolist(),
            "track_angle": sim["track_angle"],
            "track_segment": sim["track_segment"],
            "track_segment_time": episode["track_segment_time"],
            "lap_time": sim["lap_time"],
            "lap_count": episode["lap_count"],
            "total_lap_count": epoch["lap_count"],
            "total_crash_count": epoch["crash_count"],
            "crash_count": episode["crash_count"],
            "return_to_track_count": epoch["return_to_track_count"],
            "num_commands_in_queue": self.manager.get_car(self.env_id).cmd_queue.qsize(),
            "num_frames_in_queue": self.manager.get_car(self.env_id).frame_queue.qsize(),
        }

    def write_state_snapshot(self, data):
        state_json = json.dumps(data).encode("utf-8")
        self.redis.hset(self.env_id, "state_snapshot.json", state_json)

    def _reset(self):
        self._episode_ended = False
        self.episode_state = self.zero_state()
        return ts.restart(self.empty_observation())

    def do_action(self, action):
        self.manager.send_command(self.env_id, {"action": "forward", "value": float(action[0])})
        self.manager.send_command(self.env_id, {"action": "turn", "value": float(action[1])})

    def transition(self, observation, **ts_kwargs):
        util.sleep_until(self.episode_state["next_env_step_time"])
        return ts.transition(observation, **ts_kwargs)

    def terminate(self, observation, **ts_kwargs):
        self._episode_ended = True
        self.episode_state = self.zero_state()
        return ts.termination(observation, **ts_kwargs)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        episode = self.episode_state
        epoch = self.epoch_state
        episode["env_step"] += 1
        # For throttling FPS
        episode["next_env_step_time"] += self.step_interval_sec
        # Use the same clock value in all time delta computations for this step
        episode["env_step_clock"] = time.perf_counter()

        if episode["env_step"] == 1:
            epoch["episode"] += 1
            logging.debug("'%s' - begin episode %d", self.env_id, epoch["episode"])

        # Read car state and do action
        frames, sim_state = self.get_car_state()
        self.do_action(action)

        # Skip step if there is no state, can happen if the simulator is laggy
        if not sim_state:
            logging.warning("'%s' - simulator state is empty, skipping step %d", self.env_id, episode["env_step"])
            return self.transition(self.empty_observation(), reward=0)
        if not frames:
            logging.warning("'%s' - camera frame buffer is empty, skipping step %d", self.env_id, episode["env_step"])
            return self.transition(self.empty_observation(), reward=0)

        # Merge camera frame buffer and extract neural net inputs
        frame, observation = features.camera_frames_to_observation(frames)

        # Update state for computing reward at this step
        episode["speed"] = np.linalg.norm(sim_state["velocity"])
        episode["action"] = action
        episode["observation"] = observation
        episode["original_frame"] = frame
        if "position" in episode:
            episode["distance"] += minkowski(episode["position"], sim_state["position"], 2)
        episode["position"] = sim_state["position"]

        # Compute reward at current env step
        reward = features.reward(episode, epoch, sim_state)

        # Update state for computing reward at next step
        returned_to_track = sim_state["return_to_track_count"] > epoch["return_to_track_count"]
        crossed_segment = (
            sim_state["track_segment"] > 0 and sim_state["track_segment"] == episode["track_segment"] + 1
            or
            sim_state["track_segment"] == 0 and episode["track_segment"] > sim_state["track_segment"])

        for count_key in ["lap_count", "crash_count"]:
            count = max(0, sim_state[count_key])
            if count == epoch[count_key] + 1:
                episode[count_key] += 1
            epoch[count_key] = count

        episode["return"] += reward
        epoch["return_to_track_count"] = sim_state["return_to_track_count"]
        episode["track_segment"] = sim_state["track_segment"]

        if crossed_segment:
            episode["track_segment_time"] = episode["env_step_clock"] - episode["track_segment_begin_clock"]
            episode["track_segment_begin_clock"] = episode["env_step_clock"]

        # Write JSON snapshot of current state into Redis
        self.write_state_snapshot(self.get_state_snapshot(sim_state))

        if episode["lap_count"] >= self.laps_per_episode:
            # Enough laps, terminate episode
            logging.debug("'%s' - end episode at step %d after %d laps",
                    self.env_id, episode["env_step"], episode["lap_count"])
            return self.terminate(observation, reward=reward)
        if episode["env_step"] >= self.max_env_steps_per_episode:
            # Too many steps, terminate episode
            logging.debug("'%s' - end episode at step %d, too many steps for one episode",
                    self.env_id, episode["env_step"])
            return self.terminate(observation, reward=reward)
        if returned_to_track:
            # Car position reset after crash, terminate episode
            logging.debug("'%s' - end episode at step %d, car position reset in simulator",
                    self.env_id, episode["env_step"])
            return self.terminate(observation, reward=reward)

        # Still going, throttle FPS and transition to next step
        return self.transition(observation, reward=reward)


def create_batched_robotini_env(team_ids, car_socket_url, env_kwargs, isolation=False):
    car_envs = [RobotiniCarEnv(team_id, **env_kwargs) for team_id in team_ids]
    teams = list(manager.teams_from_envs(car_envs, car_socket_url))

    batch_env = batched_py_environment.BatchedPyEnvironment(car_envs, multithreading=not isolation)
    tf_env = tf_py_environment.TFPyEnvironment(batch_env, isolation=isolation)

    return teams, tf_env
