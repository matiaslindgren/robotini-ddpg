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


def communication_loop(stop_msg, simulator_url, login_cmds, frames, commands):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    default_cmd = {"action": "forward", "value": 0}
    with contextlib.closing(connect(simulator_url)) as sock:
        for login_cmd in login_cmds:
            send_json(sock, login_cmd)
        sock.settimeout(TIMEOUT)
        while True:
            try:
                if stop_msg.get(block=False):
                    break
            except queue.Empty:
                pass
            try:
                buf = camera.read_buffer(sock)
            except socket.timeout:
                continue
            frames.put(buf, block=False)
            sent_something = False
            while True:
                try:
                    cmd = commands.get(block=False)
                    send_json(sock, cmd)
                    sent_something = True
                except queue.Empty:
                    break
            if not sent_something:
                send_json(sock, default_cmd)


class CarConnection:

    def __init__(self, simulator_url, car_name, team_id, car_color):
        self.car_id = team_id if car_name == team_id else team_id + "_" + car_name
        login_cmds = [
            {"name": car_name, "teamId": team_id, "color": car_color},
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
        self.stop_msg.put(self.proc.name)
        self.proc.join(timeout=0.5)
        if self.proc.exitcode is None:
            print(self.proc.name, "did not terminate properly, killing process")
            self.proc.terminate()

    def read_camera_frames(self):
        """
        Read all available camera frames from the buffer.
        Blocks until next frame is available, thus always returning at least one frame.
        """
        buf = self.frame_queue.get(block=True)
        yield camera.buffer_to_frame(buf)
        while True:
            try:
                buf = self.frame_queue.get(block=False)
                yield camera.buffer_to_frame(buf)
            except queue.Empty:
                return

    def action(self, key, value):
        self.cmd_queue.put({"action": key, "value": value}, block=False)
