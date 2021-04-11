import base64

import cv2
import numpy as np


frame_shape = (80, 128, 3)

class rgb_idx:
    B, G, R = 0, 1, 2

def read_buffer(socket):
    length_as_bytes = socket.recv(2)
    length = length_as_bytes[0] * 256 + length_as_bytes[1]
    buf = socket.recv(length)
    while len(buf) < length:
        buf = buf + socket.recv(length - len(buf))
    return buf

def buffer_to_frame(buf):
    nparr = np.frombuffer(buf, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

def frame_to_base64(frame, size=None):
    ret, jpg = cv2.imencode('.jpg', frame)
    if size:
        jpg.resize(size)
    return 'data:image/png;base64,' + base64.b64encode(jpg.tobytes()).decode("utf-8")
