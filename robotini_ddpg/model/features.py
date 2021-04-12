"""
Feature engineering, pre-, and post-processing, the reward function.
"""
import tensorflow as tf
import numpy as np
from scipy.ndimage import zoom
from sklearn.preprocessing import minmax_scale

from robotini_ddpg.simulator.camera import rgb_idx


n_x = n_y = 3
observation_shape = (n_x+n_y, 3)

speed_penalty_threshold = 2.0
complete_lap_bonus = 20.0
complete_track_segment_bonus = 2.0
crash_penalty = 5.0
max_track_angle_penalty_threshold = 50.0

class RewardWeight:
    speed = 0.1
    crash = 1.0
    pass_segment = 1.0
    wrong_direction = 0.5


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


def reward(episode_state, global_state, simulator_state):
    s, g, sim = episode_state, global_state, simulator_state
    total = 0.0

    # More speed, more reward
    total += RewardWeight.speed * (s["speed"] - speed_penalty_threshold)

    # Crash, penalty
    if sim["crash_count"] > g["crash_count"]:
        total -= RewardWeight.crash * crash_penalty

    # Complete lap, big bonus
    if sim["lap_count"] > g["lap_count"]:
        # Slower lap => less bonus, clip at 1 second lap time (minimum/best possible)
        lap_time_weight = 1.0 / np.sqrt(np.clip(sim["lap_time"], 1, None))
        total += RewardWeight.pass_segment * lap_time_weight * complete_lap_bonus

    # Complete track segment (not finish line), smol bonus
    if sim["track_segment"] > s["track_segment"] >= 0:
        total += RewardWeight.pass_segment * complete_track_segment_bonus

    # Too large deviation from correct direction, penalty
    delta_track_angle = sim["track_angle"] - max_track_angle_penalty_threshold
    total -= RewardWeight.wrong_direction * np.clip(delta_track_angle, 0, None)

    return total


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
