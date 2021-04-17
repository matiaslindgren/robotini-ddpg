"""
Log parser for simulator JSON logs.
Connects to the spectator socket and reads JSON objects in a child process.
Selected events are extracted for each car and sent to Redis using the car id as key.
"""
from collections import defaultdict
from multiprocessing import Process, Queue
import contextlib
import json
import logging
import math
import queue
import signal
import socket

import numpy as np
from redis import Redis

from robotini_ddpg.simulator import connection
from robotini_ddpg import util


def empty_state():
    return {
        "position": 3*[0],
        "velocity": 3*[0],
        "rotation": 4*[0],
        "track_angle": 0,
        "track_segment": 0,
        "crash_count": 0,
        "return_to_track_count": 0,
        "lap_time": 0,
        "lap_count": 0,
        # "is_empty": True,
    }


def to_numpy(state):
    for k in ("position", "velocity", "rotation"):
        if k in state:
            state[k] = np.array(state[k], dtype=np.float32)
    return state


def parse_simulator_log_entry(log_entry):
    state = defaultdict(dict)
    if log_entry["type"] == "GameStatus":
        for car_data in log_entry["cars"]:
            car = state[car_data.pop("name")]
            p = car_data["position"]
            v = car_data["velocity"]
            r = car_data["rotation"]
            car["position"] = [p["x"], p["y"], p["z"]]
            car["velocity"] = [v["x"], v["y"], v["z"]]
            car["rotation"] = [r["x"], r["y"], r["z"], r["w"]]
            car["track_angle"] = car_data["trackAngle"]
            car["track_segment"] = car_data["trackSegment"]
    elif log_entry["type"] == "CarCrashed":
        car = state[log_entry["car"]["name"]]
        car["crashed"] = True
    elif log_entry["type"] == "CarReturnedToTrack":
        car = state[log_entry["car"]["name"]]
        car["returned_to_track"] = True
    elif log_entry["type"] == "CurrentStandings":
        for standing in log_entry["standings"]:
            if standing["type"] == "LapCompleted":
                if not standing["dnf"] and not math.isnan(standing["lastLap"]):
                    car = state[standing["car"]["name"]]
                    car["lap_time"] = standing["lastLap"]
                    car["lap_count"] = standing["lapCount"]
    return state


def update_car_state(old, new):
    crashed = new.pop("crashed", False)
    returned_to_track = new.pop("returned_to_track", False)
    old.update(new)
    old["crash_count"] += int(crashed)
    old["return_to_track_count"] += int(returned_to_track)


def log_parse_loop(stop_msg, stop_msg_queue, simulator_spectator_url, redis_socket_path):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    global_state = Redis(unix_socket_path=redis_socket_path)
    local_state = defaultdict(empty_state)

    json_decoder = json.JSONDecoder()
    recv_buf_size = 2*1024

    with contextlib.closing(connection.connect(simulator_spectator_url)) as sock:
        sock.settimeout(connection.TIMEOUT)
        prev_msg = ''
        while True:
            try:
                if stop_msg_queue.get_nowait() == stop_msg:
                    break
            except queue.Empty:
                pass

            # Try reading new log data
            try:
                msg = sock.recv(recv_buf_size)
            except socket.timeout:
                # No data, try later
                continue

            # Concat previous and new data, then try reading all lines from the result
            full_msg = prev_msg + msg.decode("utf-8")
            prev_msg = ''

            for line in full_msg.split('\n'):
                line = line.strip()
                if not line:
                    continue

                # Parse all JSON objects on this line
                pos = 0
                while pos < len(line):
                    try:
                        log_entry, pos = json_decoder.raw_decode(line, pos)
                    except json.JSONDecodeError:
                        # line[pos:] does not yet contain a full JSON object, try again later
                        prev_msg = line[pos:]
                        break

                    # Got one valid JSON object log entry, search for interesting messages
                    # and update local state (dict) and global state (Redis)
                    new_state = parse_simulator_log_entry(log_entry)
                    for car_id, state in new_state.items():
                        update_car_state(local_state[car_id], state)
                        state_json = json.dumps(local_state[car_id])
                        global_state.hset(car_id, "simulator_state.json", state_json.encode("utf-8"))


class LogParser:
    def __init__(self, simulator_spectator_url, redis_socket_path):
        self.stop_msg_queue = Queue()
        name = "simulator-log-parser"
        self.proc = Process(
                name=name,
                target=log_parse_loop,
                args=(name, self.stop_msg_queue, simulator_spectator_url, redis_socket_path))

    def start(self):
        self.proc.start()

    def join(self):
        if self.proc.is_alive():
            self.stop_msg_queue.put(self.proc.name, block=True)
            util.join_or_terminate(self.proc, 1)
