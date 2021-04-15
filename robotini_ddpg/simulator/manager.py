"""
Resource manager for simulator interactions.
"""
from collections import OrderedDict
import json

import numpy as np
from redis import Redis
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex

from . import connection, camera, log_parser


class Team:
    def __init__(self, env, car_suffix, color, car_socket_url):
        self.env = env
        self.color = color
        self.car_socket_url = car_socket_url
        self.car = None
        self.car_suffix = car_suffix

    def new_car(self):
        assert not self.car, "team {} already has car {}".format(self.env.env_id, self.car.car_id)
        self.car = connection.CarConnection(
                self.car_socket_url,
                self.env.env_id + "_" + self.car_suffix,
                self.color,
                self.env.env_id)
        self.car.start()
        return self.car.car_id

    def destroy_car(self):
        self.car.stop()
        self.car = None


class SimulatorManager:
    def __init__(self, envs, car_suffix, car_socket_url, log_socket_url, redis_socket_path):
        cmap = get_cmap("tab20")
        self.teams = OrderedDict(
                (env.env_id, Team(env, car_suffix, to_hex(cmap(i)), car_socket_url))
                for i, env in enumerate(envs))
        self.log_parser = log_parser.LogParser(log_socket_url, redis_socket_path)
        self.redis = Redis(unix_socket_path=redis_socket_path)

    def __enter__(self):
        for t in self.teams.values():
            t.env.manager = self
        self.log_parser.start()
        for t in self.teams.values():
            t.new_car()
        return self

    def __exit__(self, *exception_data):
        for t in self.teams.values():
            t.destroy_car()
            t.env.manager = None
        self.log_parser.stop()

    def get_simulator_state_or_empty(self, team_id):
        car_id = self.teams[team_id].car.car_id
        state_json = self.redis.hget(car_id, "simulator_state.json")
        if state_json is None:
            return log_parser.to_numpy(log_parser.empty_state())
        state = json.loads(state_json.decode("utf-8"))
        return log_parser.to_numpy(state)

    def get_car_state(self, team_id):
        car = self.teams[team_id].car
        frames = list(car.read_camera_frames())
        simulator_state = self.get_simulator_state_or_empty(team_id)
        return frames, simulator_state

    def send_command(self, team_id, cmd):
        self.teams[team_id].car.send_command(cmd)

    def new_car(self, team_id):
        return self.teams[team_id].new_car()

    def destroy_car(self, team_id):
        car_id = self.teams[team_id].car.car_id
        self.redis.delete(car_id)
        self.redis.delete(team_id)
        self.teams[team_id].destroy_car()
