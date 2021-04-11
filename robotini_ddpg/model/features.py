import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom
from sklearn.preprocessing import minmax_scale

from robotini_ddpg.simulator.camera import rgb_idx


n_x = n_y = 3
input_shape = (n_x+n_y, 3)


def camera_frames_to_observation(frames):
    frame = np.stack(frames).max(axis=0)
    frame = frame[20:,:,:].astype(np.float32)
    x = frame.mean(axis=0)
    y = frame.mean(axis=1)
    x = minmax_scale(x, (0, 1), axis=-1)
    y = minmax_scale(y, (0, 1), axis=-1)
    x = zoom(x, (n_x/x.shape[0], 1))
    y = zoom(y, (n_y/y.shape[0], 1))
    o = np.concatenate((x, y))
    o = np.clip(o, 0, 1)
    assert not np.isnan(o).any(), "nan inputs"
    return frame, o


def reward(env_state, sim_state):
    return env_state["speed"]


def observation_to_xy_images(o, scale=10):
    x = o[:n_x]
    y = o[n_x:]
    x_img = np.tile(np.expand_dims(255*x, 0), (scale, 1, 1))
    y_img = np.transpose(np.tile(np.expand_dims(255*y, 0), (scale, 1, 1)), (1, 0, 2))
    return x_img, y_img


def color_component_mass(color, c):
    total = tf.reduce_sum(color, axis=[1, 2])
    mass = total - tf.reduce_sum(color[:,:,c], axis=1)
    return tf.clip_by_value(mass, 0, 100)


def turn_from_color_mass(observation):
    y = observation[:,n_x:]
    yr = color_component_mass(y, rgb_idx.R)
    yg = color_component_mass(y, rgb_idx.G)
    yb = color_component_mass(y, rgb_idx.B)
    turn = (yr - yg)/tf.clip_by_value(yb, 0.01, 10)
    return turn
