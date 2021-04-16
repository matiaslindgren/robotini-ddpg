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

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

import yaml
import tensorflow as tf

from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from robotini_ddpg.model import env, snail, ddpg
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

    team_ids = ["env{}".format(i+1) for i in range(conf.num_train_envs+conf.num_eval_envs)]
    train_team_ids = team_ids[:conf.num_train_envs]
    eval_team_ids = team_ids[conf.num_train_envs:]

    print("agent monitor webapp startup command:\n\n"
          "{:s} robotini_ddpg/monitor/webapp.py {:s} {:s}\n\n".format(
              os.environ["PYTHON_BIN"], redis_socket_path, ' '.join(team_ids)))

    global_step = tf.compat.v1.train.get_or_create_global_step()

    train_teams, train_env = env.create_batched_tf_env(
            train_team_ids, redis_socket_path, car_socket_url)
    eval_teams, eval_env = env.create_batched_tf_env(
            eval_team_ids, redis_socket_path, car_socket_url)

    tf_agent = ddpg.create_agent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            conf.actor,
            conf.critic,
            conf.target_update_tau)

    logger.info("Actor network:")
    tf_agent._actor_network.summary(print_fn=logging.info)
    logger.info("Critic network:")
    tf_agent._critic_network.summary(print_fn=logging.info)

    train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(batch_size=conf.num_train_envs),
            tf_metrics.AverageEpisodeLengthMetric(batch_size=conf.num_train_envs),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy
    initial_collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=conf.replay_buffer_capacity)

    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            initial_collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_steps=conf.initial_collect_steps)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_steps=conf.collect_steps_per_iteration)

    if conf.use_tf_functions:
        initial_collect_driver.run = common.function(initial_collect_driver.run)
        collect_driver.run = common.function(collect_driver.run)
        tf_agent.train = common.function(tf_agent.train)

    logger.info("Testing train environment with trivial policy")
    car_suffix = "train_env_test"
    with manager.SimulatorManager(train_teams, car_suffix, log_socket_url, redis_socket_path) as sim_manager:
        snail.run_snail_until_finish_line(train_env)

    logger.info("Testing eval environment with trivial policy")
    car_suffix = "eval_env_test"
    with manager.SimulatorManager(eval_teams, car_suffix, log_socket_url, redis_socket_path) as sim_manager:
        snail.run_snail_until_finish_line(eval_env)

    logger.info("Initializing replay buffer by collecting experience for %d steps in %d parallel environments",
            conf.initial_collect_steps, conf.num_train_envs)
    car_suffix = "init_experience"
    with manager.SimulatorManager(train_teams, car_suffix, log_socket_url, redis_socket_path) as sim_manager:
        snail.run_snail_until_finish_line(train_env)
        initial_collect_driver.run()

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
        experience_batch, _ = next(iterator)
        return tf_agent.train(experience_batch)

    if conf.use_tf_functions:
        train_step = common.function(train_step)

    # eval_policy_saver = policy_saver.PolicySaver(eval_policy, batch_size=eval_env.batch_size)
    eval_metrics = [
            tf_metrics.AverageReturnMetric(
                batch_size=conf.num_eval_envs,
                buffer_size=conf.num_eval_episodes),
            tf_metrics.AverageEpisodeLengthMetric(
                batch_size=conf.num_eval_envs,
                buffer_size=conf.num_eval_episodes),
    ]

    for iteration in range(1, conf.num_iterations+1):
        car_suffix = "iteration{:d}".format(iteration)

        start_time = time.time()
        with manager.SimulatorManager(train_teams, car_suffix, log_socket_url, redis_socket_path) as sim_manager:
            snail.run_snail_until_finish_line(train_env)
            logger.info("Iteration %d - Collecting experience for %d steps in %d parallel environments",
                    iteration, conf.collect_steps_per_iteration, conf.num_train_envs)
            time_step, policy_state = collect_driver.run(
                    time_step=time_step,
                    policy_state=policy_state,
            )

        logger.info("Iteration %d - Training", iteration)
        for _ in range(conf.train_steps_per_iteration):
            train_loss = train_step()

        logger.info("Iteration %d - Evaluating", iteration)

        time_acc += time.time() - start_time

        if global_step.numpy() % conf.log_interval == 0:
            logger.info('step = %d, loss = %f', global_step.numpy(),
                                     train_loss.loss)
            steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
            logger.info('%.3f steps/sec', steps_per_sec)
            tf.compat.v2.summary.scalar(
                    name='global_steps_per_sec', data=steps_per_sec, step=global_step)
            timed_at_step = global_step.numpy()
            time_acc = 0

        for train_metric in train_metrics:
            train_metric.tf_summaries(
                    train_step=global_step, step_metrics=train_metrics[:2])

        if global_step.numpy() % conf.eval_interval == 0:
            with manager.SimulatorManager(eval_teams, car_suffix + "_evaluation", log_socket_url, redis_socket_path) as sim_manager:
                snail.run_snail_until_finish_line(eval_env)
                _ = metric_utils.compute(
                        eval_metrics,
                        eval_env,
                        eval_policy,
                        num_episodes=conf.num_eval_episodes,
                        # train_step=global_step,
                        # summary_writer=eval_summary_writer,
                        # summary_prefix='Metrics',
                )
            metric_utils.log_metrics(eval_metrics)

        # if global_step.numpy() % conf.save_interval == 0:
        #     save_path = os.path.join(policy_dir, "policy_step{:d}".format(global_step.numpy()))
        #     logger.info("saving policy to '%s'", save_path)
        #     eval_policy_saver.save(save_path)


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
