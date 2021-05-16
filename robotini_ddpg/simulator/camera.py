"""
Simulator camera frame reader from the Robotini Python template.
OpenCV has been replaced with TensorFlow.
"""
import base64

import numpy as np
import tensorflow as tf


frame_shape = (80, 128, 3)

class rgb_idx:
    R, G, B = 0, 1, 2

def read_buffer(socket):
    length_as_bytes = socket.recv(2)
    length = length_as_bytes[0] * 256 + length_as_bytes[1]
    buf = socket.recv(length)
    while len(buf) < length:
        buf = buf + socket.recv(length - len(buf))
    return buf

def buffer_to_frame(buf):
    img = tf.io.decode_image(buf, channels=3, dtype=tf.uint8)
    return tf.reshape(img, frame_shape).numpy()

def frame_to_base64(frame):
    jpg = tf.io.encode_jpeg(frame, format='rgb').numpy()
    data_str = base64.b64encode(jpg).decode("utf-8")
    return 'data:image/png;base64,' + data_str
