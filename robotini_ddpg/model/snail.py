"""
Hand-written policy based on RGB component values.
"""
import tensorflow as tf
from tf_agents.policies import tf_policy
from tf_agents.trajectories import policy_step

from robotini_ddpg.model import features


class BlueSnailPolicy(tf_policy.TFPolicy):
    def __init__(self, *args, constant_forward=0.05, **kwargs):
        self.constant_forward = constant_forward
        super().__init__(*args, **kwargs)

    def _action(self, time_step, policy_state, seed):
        turn = features.turn_from_color_mass(time_step.observation)
        forward = tf.repeat(tf.constant([self.constant_forward]), turn.shape[0])
        action = tf.stack((forward, turn), axis=1)
        return policy_step.PolicyStep(action, policy_state)
