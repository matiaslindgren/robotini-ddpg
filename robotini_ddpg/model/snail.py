"""
Hand-written policy based on RGB component values.
"""
import numpy as np
import tensorflow as tf
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step, trajectory

from robotini_ddpg.model import features


class BlueSnailPolicy(tf_policy.TFPolicy):
    def __init__(self, *args, forward_action=0.05, **kwargs):
        self.forward_action = forward_action
        super().__init__(*args, **kwargs)

    def _action(self, time_step, policy_state, seed):
        turn = features.turn_from_color_mass(time_step.observation)
        forward = tf.repeat(tf.constant([self.forward_action]), turn.shape[0])
        action = tf.stack((forward, turn), axis=1)
        return policy_step.PolicyStep(action, policy_state)

    def do_step(self, env, time_step):
        action_step = self.action(time_step)
        next_time_step = env.step(action_step.action)
        traj = trajectory.Trajectory(
                time_step.step_type,
                time_step.observation,
                action_step.action,
                action_step.info,
                next_time_step.step_type,
                next_time_step.reward,
                next_time_step.discount)
        return traj, next_time_step


def run_snail_until_finish_line(batched_tf_env, forward_action=0.05):
    policy = BlueSnailPolicy(
            batched_tf_env.time_step_spec(),
            batched_tf_env.action_spec(),
            forward_action=forward_action,
            clip=False)
    time_step = batched_tf_env.reset()
    still_running = {env.env_id for env in batched_tf_env.pyenv.envs}
    while still_running:
        _, time_step = policy.do_step(batched_tf_env, time_step)
        for env in batched_tf_env.pyenv.envs:
            if env.env_id in still_running:
                _, sim_state = env.get_car_state()
                if sim_state and sim_state["track_segment"] == 0:
                    still_running.remove(env.env_id)
    batched_tf_env.reset()
