# Original file (accessed 03/2021):
# https://github.com/tensorflow/agents/blob/v0.7.1/tf_agents/agents/ddpg/examples/v2/train_eval_rnn.py

# License in original file:
# ---------------------------------------
# Copyright 2020 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------
"""
Training script for DDPG.
"""
import argparse
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

import yaml
import tensorflow as tf

from tf_agents.agents.ddpg import actor_rnn_network
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment, batched_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from robotini_ddpg.model import env, snail
from robotini_ddpg.simulator import manager


class Config:
    def __init__(self, path):
        with open(path) as f:
            self.__dict__.update(yaml.safe_load(f.read()))


def train(conf, cache_dir, car_socket_url, log_socket_url, redis_socket_path):
    cache_dir = os.path.expanduser(cache_dir)
    train_dir = os.path.join(cache_dir, 'train')
    eval_dir = os.path.join(cache_dir, 'eval')
    policy_dir = os.path.join(cache_dir, 'policies')

    train_summary_writer = tf.compat.v2.summary.create_file_writer(
            train_dir, flush_millis=conf.summaries_flush_secs * 1000)
    train_summary_writer.set_as_default()

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
            eval_dir, flush_millis=conf.summaries_flush_secs * 1000)
    eval_metrics = [
            tf_metrics.AverageReturnMetric(buffer_size=conf.num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=conf.num_eval_episodes)
    ]

    train_team_ids = ["env{}_train".format(i) for i in range(1, conf.num_parallel_envs+1)]
    eval_id = "env{}_eval".format(len(train_team_ids)+1)

    print("agent monitor webapp startup command:\n\n"
          "{:s} robotini_ddpg/monitor/webapp.py {:s} {:s}\n\n".format(
              os.environ["PYTHON_BIN"], redis_socket_path, ' '.join(train_team_ids + [eval_id])))

    global_step = tf.compat.v1.train.get_or_create_global_step()
    with (tf.compat.v2.summary.record_if(lambda: tf.math.equal(global_step % conf.summary_interval, 0)),
          manager.SimulatorManager(
              train_team_ids + [eval_id],
              car_socket_url,
              log_socket_url,
              redis_socket_path) as simulator_manager):
        train_envs = [env.RobotiniCarEnv(simulator_manager, team_id, redis_socket_path)
                      for team_id in train_team_ids]
        train_env = batched_py_environment.BatchedPyEnvironment(train_envs, multithreading=True)
        train_env = tf_py_environment.TFPyEnvironment(train_env, isolation=False)

        eval_env = env.RobotiniCarEnv(simulator_manager, eval_id, redis_socket_path)
        eval_env = tf_py_environment.TFPyEnvironment(eval_env, isolation=False)

        actor_net = actor_rnn_network.ActorRnnNetwork(
                train_env.time_step_spec().observation,
                train_env.action_spec(),
                input_fc_layer_params=conf.actor_fc_layers,
                lstm_size=conf.actor_lstm_size,
                output_fc_layer_params=conf.actor_output_fc_layers,
        )

        critic_net_input_specs = (
                train_env.time_step_spec().observation,
                train_env.action_spec())

        critic_net = critic_rnn_network.CriticRnnNetwork(
                critic_net_input_specs,
                observation_fc_layer_params=conf.critic_obs_fc_layers,
                joint_fc_layer_params=conf.critic_joint_fc_layers,
                lstm_size=conf.critic_lstm_size,
                output_fc_layer_params=conf.critic_output_fc_layers,
        )

        tf_agent = ddpg_agent.DdpgAgent(
                train_env.time_step_spec(),
                train_env.action_spec(),
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                        learning_rate=conf.actor_learning_rate),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                        learning_rate=conf.critic_learning_rate),
                ou_stddev=conf.ou_stddev,
                ou_damping=conf.ou_damping,
                target_update_tau=conf.target_update_tau,
                target_update_period=conf.target_update_period,
                gamma=conf.gamma,
                reward_scale_factor=conf.reward_scale_factor,
                debug_summaries=conf.debug_summaries,
                summarize_grads_and_vars=conf.summarize_grads_and_vars,
                train_step_counter=global_step)
        tf_agent.initialize()

        logging.info("actor net")
        actor_net.summary(print_fn=logging.info)
        logging.info("critic net")
        critic_net.summary(print_fn=logging.info)

        train_metrics = [
                tf_metrics.NumberOfEpisodes(),
                tf_metrics.EnvironmentSteps(),
                tf_metrics.AverageReturnMetric(batch_size=conf.num_parallel_envs),
                tf_metrics.AverageEpisodeLengthMetric(batch_size=conf.num_parallel_envs),
        ]

        eval_policy = tf_agent.policy
        collect_policy = tf_agent.collect_policy
        initial_collect_policy = tf_agent.collect_policy

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                tf_agent.collect_data_spec,
                batch_size=train_env.batch_size,
                max_length=conf.replay_buffer_capacity)

        cache_fill_driver = dynamic_episode_driver.DynamicEpisodeDriver(
                train_env,
                snail.BlueSnailPolicy(
                    train_env.time_step_spec(),
                    train_env.action_spec(),
                    constant_forward=0.05,
                    clip=False),
                num_episodes=conf.cache_fill_episodes*conf.num_parallel_envs)

        initial_collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
                train_env,
                initial_collect_policy,
                observers=[replay_buffer.add_batch] + train_metrics,
                num_episodes=conf.initial_collect_episodes*conf.num_parallel_envs)

        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
                train_env,
                collect_policy,
                observers=[replay_buffer.add_batch] + train_metrics,
                num_episodes=conf.collect_episodes_per_iteration*conf.num_parallel_envs)

        if conf.use_tf_functions:
            initial_collect_driver.run = common.function(initial_collect_driver.run)
            cache_fill_driver.run = common.function(cache_fill_driver.run)
            collect_driver.run = common.function(collect_driver.run)
            tf_agent.train = common.function(tf_agent.train)

        logging.info("Filling simulator state caches for %d episodes in %d parallel environments",
                conf.cache_fill_episodes, conf.num_parallel_envs)
        cache_fill_driver.run()

        logging.info("Initializing replay buffer by collecting experience for %d episodes in %d parallel environments",
                conf.initial_collect_episodes, conf.num_parallel_envs)
        initial_collect_driver.run()

        results = metric_utils.compute(
                eval_metrics,
                eval_env,
                eval_policy,
                num_episodes=conf.num_eval_episodes,
        )
        metric_utils.log_metrics(eval_metrics)

        time_step = None
        policy_state = collect_policy.get_initial_state(train_env.batch_size)

        timed_at_step = global_step.numpy()
        time_acc = 0

        # Dataset generates trajectories with shape [BxTx...]
        dataset = replay_buffer.as_dataset(
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                sample_batch_size=conf.batch_size,
                num_steps=conf.train_sequence_length + 1).prefetch(tf.data.experimental.AUTOTUNE)
        iterator = iter(dataset)

        def train_step():
            experience, _ = next(iterator)
            return tf_agent.train(experience)

        if conf.use_tf_functions:
            train_step = common.function(train_step)

        eval_policy_saver = policy_saver.PolicySaver(eval_policy, batch_size=eval_env.batch_size)

        for iteration in range(1, conf.num_iterations+1):
            logging.info("Iteration %d - Collecting experience for %d episodes in %d parallel environments",
                    iteration, conf.collect_episodes_per_iteration, conf.num_parallel_envs)
            start_time = time.time()
            time_step, policy_state = collect_driver.run(
                    time_step=time_step,
                    policy_state=policy_state,
            )

            logging.info("Iteration %d - Training", iteration)
            for _ in range(conf.train_steps_per_iteration):
                train_loss = train_step()

            logging.info("Iteration %d - Evaluating", iteration)

            time_acc += time.time() - start_time

            if global_step.numpy() % conf.log_interval == 0:
                logging.info('step = %d, loss = %f', global_step.numpy(),
                                         train_loss.loss)
                steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
                logging.info('%.3f steps/sec', steps_per_sec)
                tf.compat.v2.summary.scalar(
                        name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                timed_at_step = global_step.numpy()
                time_acc = 0

            for train_metric in train_metrics:
                train_metric.tf_summaries(
                        train_step=global_step, step_metrics=train_metrics[:2])

            if global_step.numpy() % conf.eval_interval == 0:
                results = metric_utils.compute(
                        eval_metrics,
                        eval_env,
                        eval_policy,
                        num_episodes=conf.num_eval_episodes,
                        # train_step=global_step,
                        # summary_writer=eval_summary_writer,
                        # summary_prefix='Metrics',
                )
                metric_utils.log_metrics(eval_metrics)

            if global_step.numpy() % conf.save_interval == 0:
                save_path = os.path.join(policy_dir, "policy_step{:d}".format(global_step.numpy()))
                logging.info("saving policy to '%s'", save_path)
                eval_policy_saver.save(save_path)


def run(config_path, **kwargs):
    train(Config(config_path), **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--car-socket-url", type=str, required=True)
    parser.add_argument("--log-socket-url", type=str, required=True)
    parser.add_argument("--redis-socket-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    args, sys.argv[1:] = parser.parse_known_args()
    run(**vars(args))
