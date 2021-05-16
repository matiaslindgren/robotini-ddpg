"""
Simulator camera frame reader from the Robotini Python template.
OpenCV has been replaced with TensorFlow.
"""
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
    nparr = np.frombuffer(buf, np.uint8)
    frame = tf.io.decode_image(nparr,
            # Return a single frame instead of a batch
            expand_animations=False)
    return frame.numpy()

def frame_to_base64(frame):
    png = tf.io.encode_png(frame)
    base64 = tf.io.encode_base64(base64)
    return 'data:image/png;base64,' + base64.numpy().decode("utf-8")
