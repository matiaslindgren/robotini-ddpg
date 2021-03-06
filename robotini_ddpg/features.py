"""
Feature engineering, pre- and post-processing, and the reward function.
"""
import time

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import minmax_scale

from robotini_ddpg.simulator.camera import rgb_idx


n_x = n_y = 4
observation_shape = (n_x+n_y, 3)

complete_track_segment_bonus = 1.0
max_track_segment_time = 1.0
crash_penalty = 5.0
max_track_angle = 100.0

class RewardWeight:
    wrong_direction = 10.0
    track_segment_time = 1.0
    crash = 0.5
    track_segment_passed = 0.2


def reward(episode_state, epoch_state, simulator_state):
    episode = episode_state
    epoch = epoch_state
    sim = simulator_state
    total = 0.0

    # Complete track segment => small reward
    if sim["track_segment"] > 0 and sim["track_segment"] == episode["track_segment"] + 1:
        total += RewardWeight.track_segment_passed * complete_track_segment_bonus
        segment_time = episode["env_step_clock"] - episode["track_segment_begin_clock"]
        if segment_time < max_track_segment_time:
            # Fast completion => extra reward
            total += RewardWeight.track_segment_time * complete_track_segment_bonus / np.clip(segment_time, 0.01, None)

    # Crash => penalty. More speed during crash => more penalty
    if sim["crash_count"] > epoch["crash_count"]:
        total -= RewardWeight.crash * crash_penalty * (1 + abs(episode["speed"]))

    # Too large deviation from correct direction, penalty
    # Skip on first segment since the angle might be very wrong during/after spawn
    if episode["num_crossed_track_segments"] > 0 and sim["track_angle"] > max_track_angle:
        delta_track_angle = sim["track_angle"] - max_track_angle
        track_angle_penalty = np.clip(delta_track_angle / max_track_angle, 0, None)
        total -= RewardWeight.wrong_direction * track_angle_penalty

    # Crossed a segment in the wrong direction, penalty
    if sim["track_segment"] == episode["track_segment"] - 1:
        total -= RewardWeight.wrong_direction * complete_track_segment_bonus

    return total


def camera_frames_to_observation(frame_batch):
    frame = np.stack(frame_batch).max(axis=0)
    frame = frame[20:,:,:].astype(np.float32)
    frame = threshold_rgb(frame)
    x = frame.mean(axis=0)
    y = frame.mean(axis=1)
    x = minmax_scale(x, (0, 1), axis=-1)
    y = minmax_scale(y, (0, 1), axis=-1)
    x = x.reshape((n_x, x.shape[0]//n_x) + x.shape[1:])
    y = y.reshape((n_y, y.shape[0]//n_y) + y.shape[1:])
    x = x.mean(axis=1)
    y = y.mean(axis=1)
    o = np.concatenate((x, y)).astype(np.float32)
    assert not np.isnan(o).any(), "nan inputs"
    return frame, o

def observation_to_xy_images(o, scale=10):
    x = o[:n_x]
    y = o[n_x:]
    x_img = np.tile(np.expand_dims(255*x, 0), (scale, 1, 1))
    y_img = np.tile(np.expand_dims(255*y, 0), (scale, 1, 1))
    y_img = np.transpose(y_img, (1, 0, 2))
    return x_img, y_img


def component_sum(v, c):
    s = tf.reduce_sum(v[:,:,c], axis=1)
    return tf.clip_by_value(s, 0, 100)

def compute_turn_from_color_mass(observation):
    y = observation[:,n_x:]
    yr = component_sum(y, rgb_idx.R)
    yg = component_sum(y, rgb_idx.G)
    yb = component_sum(y, rgb_idx.B)
    turn = (yg - yr)/tf.clip_by_value(yb, 0.01, 10)
    return turn


# Simplified non-opencv version of find_bgr_pixels_over_treshold from the Robotini Python template
def threshold_rgb(img, threshold=100.0):
    r, g, b = (
        img[:,:,rgb_idx.R],
        img[:,:,rgb_idx.G],
        img[:,:,rgb_idx.B],
    )
    max_px = np.maximum(np.maximum(r, g), b)

    def process_channel(chan):
        chan[chan < max_px] = 0.0
        return np.where(chan > threshold, 255.0, 0.0)

    r = process_channel(r)
    g = process_channel(g)
    b = process_channel(b)

    return np.stack((r, g, b), axis=-1)
