from collections import defaultdict
from multiprocessing import Process, Queue
import json
import logging
import math
import queue
import signal
import socket
import time

import numpy as np
from redis import Redis

from robotini_ddpg.simulator import connection


def empty_state():
    return {
        "position": 3*[0],
        "velocity": 3*[0],
        "rotation": 4*[0],
        "track_angle": 0,
        "track_segment": 0,
        "crashed": False,
        "lap_time": 0,
        "lap_count": 0,
    }


def to_numpy(state):
    for k in ["position", "velocity", "rotation"]:
        state[k] = np.array(state[k], dtype=np.float32)
    return state


def parse_simulator_logdata(sim_data):
    state = defaultdict(empty_state)
    if sim_data["type"] == "GameStatus":
        for car_data in sim_data["cars"]:
            car = state[car_data.pop("name")]
            p = car_data["position"]
            v = car_data["velocity"]
            r = car_data["rotation"]
            car["position"] = [p["x"], p["y"], p["z"]]
            car["velocity"] = [v["x"], v["y"], v["z"]]
            car["rotation"] = [r["x"], r["y"], r["z"], r["w"]]
            car["track_angle"] = car_data["trackAngle"]
            car["track_segment"] = car_data["trackSegment"]
    elif sim_data["type"] == "CarCrashed":
        car = state[sim_data["car"]["name"]]
        car["crashed"] = True
    elif sim_data["type"] == "CurrentStandings":
        for standing in sim_data["standings"]:
            if standing["type"] == "LapCompleted":
                if not standing["dnf"] and not math.isnan(standing["lastLap"]):
                    car = state[standing["car"]["name"]]
                    car["lap_time"] = standing["lastLap"]
                    car["lap_count"] = standing["lapCount"]
    return state


def log_parse_loop(stop_msg, stop_msg_q, simulator_spectator_url, redis_socket_path):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    redis = Redis(unix_socket_path=redis_socket_path)
    with connection.connect(simulator_spectator_url) as sock:
        json_decoder = json.JSONDecoder()
        partial_msg = ''
        while True:
            try:
                if stop_msg_q.get(block=False) == stop_msg:
                    break
            except queue.Empty:
                pass
            msg = sock.recv(2048)
            msg = partial_msg + msg.decode("utf-8")
            partial_msg = ''
            for line in msg.split('\n'):
                line = line.strip()
                if not line:
                    continue
                pos = 0
                while pos < len(line):
                    try:
                        logdata, pos = json_decoder.raw_decode(line, pos)
                        # parsed_out_q.put(logdata, block=False)
                        car2state = parse_simulator_logdata(logdata)
                        for car_name, state in car2state.items():
                            state_json = json.dumps(state).encode("utf-8")
                            redis.hset(car_name, "simulator_state.json", state_json)
                    except json.JSONDecodeError:
                        partial_msg = line[pos:]
                        break


def state_cache_loop(stop_msg, stop_msg_q, redis_socket_path, parsed_q):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    redis = Redis(unix_socket_path=redis_socket_path)
    while True:
        try:
            if stop_msg_q.get(block=False) == stop_msg:
                break
        except queue.Empty:
            pass
        try:
            logdata = parsed_q.get(block=False)
            car2state = parse_simulator_logdata(logdata)
            for car_name, state in car2state.items():
                state_json = json.dumps(state).encode("utf-8")
                redis.hset(car_name, "simulator_state.json", state_json)
        except queue.Empty:
            time.sleep(connection.TIMEOUT)


class LogParser:
    def __init__(self, simulator_spectator_url, redis_socket_path):
        self.stop_msg_q = Queue()
        self.parsed_out_q = Queue()
        name = "simulator-log-parser"
        self.log_parse_proc = Process(
                name=name,
                target=log_parse_loop,
                args=(name, self.stop_msg_q, simulator_spectator_url, redis_socket_path))

    def start(self):
        self.log_parse_proc.start()

    def stop(self):
        p = self.log_parse_proc
        self.stop_msg_q.put(p.name)
        p.join(timeout=1)
        if p.exitcode is None:
            print(p.name, "did not terminate properly, killing process")
            p.terminate()
