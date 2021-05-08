"""
Feature engineering, pre- and post-processing, and the reward function.
"""
import time

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import minmax_scale

from robotini_ddpg.simulator.camera import rgb_idx


n_x = n_y = 4
observation_shape = (n_x+n_y, 3)

complete_track_segment_bonus = 2.0
complete_lap_bonus = 100.0
crash_penalty = 5.0
max_track_angle = 90.0

class RewardWeight:
    wrong_direction = 1.0
    track_segment_time = 2.0
    crash = 0.5
    track_segment_passed = 0.1


def reward(episode_state, epoch_state, simulator_state):
    episode = episode_state
    epoch = epoch_state
    sim = simulator_state
    total = 0.0

    # Complete track segment (not finish line), smol bonus
    if sim["track_segment"] > 0 and sim["track_segment"] == episode["track_segment"] + 1:
        # Faster completion, more bonus
        segment_time = episode["env_step_clock"] - episode["track_segment_begin_clock"]
        segment_time_weight = RewardWeight.track_segment_time / (0.1 + segment_time)
        total += RewardWeight.track_segment_passed * complete_track_segment_bonus * segment_time_weight

    # # Complete lap, big bonus
    # if sim["lap_count"] > epoch["lap_count"]:
    #     # Slower lap => less bonus, clip at 1 second lap time (minimum/best possible)
    #     lap_time_weight = 1.0 / np.clip(sim["lap_time"], 1, None)
    #     total += RewardWeight.track_segment_passed * lap_time_weight * complete_lap_bonus

    # Crash, penalty. More speed, more penalty
    if sim["crash_count"] > epoch["crash_count"]:
        total -= RewardWeight.crash * crash_penalty * (1 + abs(episode["speed"]))

    # Too large deviation from correct direction, penalty
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
    x = frame.mean(axis=0)
    y = frame.mean(axis=1)
    x = minmax_scale(x, (0, 1), axis=-1)
    y = minmax_scale(y, (0, 1), axis=-1)
    x = x.reshape((n_x, x.shape[0]//n_x) + x.shape[1:])
    y = y.reshape((n_y, y.shape[0]//n_y) + y.shape[1:])
    x = x.mean(axis=1)
    y = y.mean(axis=1)
    o = np.concatenate((x, y))
    assert not np.isnan(o).any(), "nan inputs"
    return frame, o


def observation_to_xy_images(o, scale=10):
    x = o[:n_x]
    y = o[n_x:]
    x_img = np.tile(np.expand_dims(255*x, 0), (scale, 1, 1))
    y_img = np.transpose(np.tile(np.expand_dims(255*y, 0), (scale, 1, 1)), (1, 0, 2))
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
