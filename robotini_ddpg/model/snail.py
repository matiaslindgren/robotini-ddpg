"""
Hand-written policy based on RGB component values.
"""
import numpy as np
import tensorflow as tf
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step, trajectory

from robotini_ddpg.model import features


class BlueSnailPolicy(tf_policy.TFPolicy):
    def __init__(self, *args, constant_forward=0.04, **kwargs):
        self.constant_forward = constant_forward
        super().__init__(*args, **kwargs)

    def _action(self, time_step, policy_state, seed):
        turn = features.turn_from_color_mass(time_step.observation)
        forward = tf.repeat(tf.constant([self.constant_forward]), turn.shape[0])
        action = tf.stack((forward, turn), axis=1)
        return policy_step.PolicyStep(action, policy_state)


# modified PyDriver.run from
# https://www.tensorflow.org/agents/tutorials/4_drivers_tutorial
def run_snail_in_envs(envs, max_episodes=None, max_steps=None):
    assert max_episodes or max_steps, "at least one of {max_episodes, max_steps} must be defined"
    max_episodes = max_episodes or np.inf
    max_steps = max_steps or np.inf
    policies = [BlueSnailPolicy(env.time_step_spec(), env.action_spec(), clip=False)
                for env in envs]
    time_steps = [env.reset() for env in envs]
    num_episodes = num_steps = 0
    while num_episodes < max_episodes and num_steps < max_steps:
        for i, (env, policy, time_step) in enumerate(zip(envs, policies, time_steps)):
            action_step = policy.action(time_step)
            next_time_step = env.step(action_step.action)
            traj = trajectory.Trajectory(
                    time_step.step_type,
                    time_step.observation,
                    action_step.action,
                    action_step.info,
                    next_time_step.step_type,
                    next_time_step.reward,
                    next_time_step.discount)
            num_episodes += np.sum(traj.is_last())
            num_steps += np.sum(~traj.is_boundary())
            time_steps[i] = next_time_step
