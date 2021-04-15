"""
Connections between a car and the simulator.
For sending actions and receiving camera frames.
The communication loop runs in a child process.
"""
from multiprocessing import Process, Queue
import contextlib
import json
import queue
import signal
import socket
import time

import numpy as np

from . import camera


TIMEOUT = 1/60


def connect(url):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    host, port = url.split(':')
    sock.connect((host, int(port)))
    return sock


def send_json(sock, data):
    msg_string = json.dumps(data, separators=(',', ':')) + "\n"
    msg = bytes(msg_string, "utf-8")
    err = sock.sendall(msg)
    assert err is None, "failed sock.sendall '{:s}', with msg of length {:d}".format(err, len(msg))


def communication_loop(stop_msg_q, simulator_url, login_cmds, frames, commands):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    def false_until_stop():
        while True:
            try:
                if stop_msg_q.get(block=False):
                    break
            except queue.Empty:
                pass
            yield False
        while True:
            yield True
    stop_msg = false_until_stop()
    with contextlib.closing(connect(simulator_url)) as sock:
        try:
            for login_cmd in login_cmds:
                send_json(sock, login_cmd)
            sock.settimeout(TIMEOUT)
            while not next(stop_msg):
                try:
                    buf = camera.read_buffer(sock)
                except socket.timeout:
                    continue
                except Exception as e:
                    print("unexpected exception during camera read:")
                    print(repr(e))
                    continue
                frames.put(buf, block=False)
                k = False
                while not next(stop_msg):
                    try:
                        cmd = commands.get(block=False)
                        send_json(sock, cmd)
                        k = True
                    except queue.Empty:
                        break
                if not k:
                    send_json(sock, {"action": "forward", "value": 0})
        except Exception as e:
            print("terminating communication_loop after unhandled exception:")
            print(repr(e))


class CarConnection:
    def __init__(self, simulator_url, car_id, car_color, team_id):
        self.car_id = car_id
        login_cmds = [
            {"name": self.car_id, "teamId": team_id, "color": car_color},
            {"move": True},
        ]
        self.stop_msg = Queue()
        self.frame_queue = Queue()
        self.cmd_queue = Queue()
        self.proc = Process(
                name="simulator-car-connection-{}".format(self.car_id),
                target=communication_loop,
                args=(self.stop_msg, simulator_url, login_cmds, self.frame_queue, self.cmd_queue))

    def start(self):
        self.proc.start()

    def stop(self):
        if not self.proc.is_alive():
            return
        self.stop_msg.put(self.proc.name)
        self.proc.join(timeout=0.5)
        if self.proc.exitcode is None:
            print(self.proc.name, "did not terminate properly, killing process")
            self.proc.terminate()

    def read_camera_frames(self):
        timeout = 2
        while True:
            try:
                buf = self.frame_queue.get(block=timeout > 0, timeout=timeout)
                yield camera.buffer_to_frame(buf)
                timeout = 0
            except queue.Empty:
                return

    def send_command(self, cmd):
        self.cmd_queue.put(cmd, block=False)
