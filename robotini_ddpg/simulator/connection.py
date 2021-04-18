"""
Connections between a car and the simulator.
For sending actions and receiving camera frames.
The communication loop runs in a child process.
"""
from multiprocessing import Process, Queue
import contextlib
import json
import logging
import queue
import signal
import socket
import time

import numpy as np

from . import camera
from robotini_ddpg import util


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


def communication_loop(stop_msg, stop_msg_queue, simulator_url, login_cmds, frames, commands):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    with contextlib.closing(connect(simulator_url)) as sock:
        # Send login commands to simulator
        for login_cmd in login_cmds:
            send_json(sock, login_cmd)

        sock.settimeout(TIMEOUT)
        while True:

            # Check for stop-the-loop message
            try:
                if stop_msg_queue.get_nowait() == stop_msg:
                    break
            except queue.Empty:
                pass

            # Read camera and send commands
            num_sent_commands = 0
            try:
                buf = camera.read_buffer(sock)
                # Got camera frames, put on queue for env to read
                frames.put(buf, block=True)
                # Send all queued commands
                while True:
                    try:
                        cmd = commands.get_nowait()
                        send_json(sock, cmd)
                        num_sent_commands += 1
                    except queue.Empty:
                        break
            except socket.timeout:
                # No frames, try later
                pass
            except:
                logging.exception("unexpected exception during camera read")
            finally:
                # Always send something back
                if num_sent_commands == 0:
                    send_json(sock, {"action": "forward", "value": 0})


class CarConnection:
    def __init__(self, simulator_url, car_id, car_color, team_id):
        self.car_id = car_id
        login_cmds = [
            {"name": self.car_id, "teamId": team_id, "color": car_color},
            {"move": True},
        ]
        self.stop_msg_queue = Queue()
        self.frame_queue = Queue()
        self.cmd_queue = Queue()
        name = "simulator-car-connection-{}".format(self.car_id)
        self.proc = Process(
                name=name,
                target=communication_loop,
                args=(name, self.stop_msg_queue, simulator_url, login_cmds, self.frame_queue, self.cmd_queue))

    def start(self):
        self.proc.start()

    def stop(self):
        if self.proc.is_alive():
            self.stop_msg_queue.put(self.proc.name, block=True)

    def join(self):
        if self.proc.is_alive():
            util.drain_queue(self.frame_queue)
            util.drain_queue(self.cmd_queue)
            util.join_or_terminate(self.proc, 1)

    def read_camera_frames(self, wait_on_first=0.1):
        while True:
            try:
                buf = self.frame_queue.get(block=wait_on_first > 0, timeout=wait_on_first)
                frame = camera.buffer_to_frame(buf)
                if frame is not None:
                    yield frame
                wait_on_first = 0
            except queue.Empty:
                return

    def send_command(self, cmd):
        self.cmd_queue.put_nowait(cmd)
