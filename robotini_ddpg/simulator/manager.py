from collections import OrderedDict
import json

import numpy as np
from redis import Redis
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex

from . import connection, camera, log_parser


def create_connections(car_socket_url, team_ids):
    cmap = get_cmap("tab20")
    connections = OrderedDict()
    for color_num, team_id in enumerate(team_ids):
        color = to_hex(cmap(color_num))
        conn = connection.CarConnection(car_socket_url, team_id, team_id, color)
        connections[team_id] = conn
    return connections


class SimulatorManager:
    def __init__(self, team_ids, car_socket_url, log_socket_url, redis_socket_path):
        self.connections = create_connections(car_socket_url, team_ids)
        self.log_parser = log_parser.LogParser(log_socket_url, redis_socket_path)
        self.redis = Redis(unix_socket_path=redis_socket_path)

    def __enter__(self):
        self.log_parser.start()
        for c in self.connections.values():
            c.start()
        return self

    def __exit__(self, *exception_data):
        while self.connections:
            _, c = self.connections.popitem(last=False)
            c.stop()
        self.log_parser.stop()

    def get_simulator_state_or_empty(self, team_id):
        car_id = self.connections[team_id].car_id
        state_json = self.redis.hget(car_id, "simulator_state.json")
        if state_json is None:
            return {}
        state = json.loads(state_json.decode("utf-8"))
        return log_parser.to_numpy(state)

    def get_car_state(self, team_id):
        conn = self.connections[team_id]
        frames = list(conn.read_camera_frames())
        simulator_state = self.get_simulator_state_or_empty(team_id)
        return frames, simulator_state

    def send_action(self, team_id, k, v):
        self.connections[team_id].action(k, v)
