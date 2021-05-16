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
Main training script for creating a DDPG agent for the Robotini Racing Simulator.
"""
import argparse
import logging
import os
import sys
import time

import yaml
import numpy as np
import tensorflow as tf

from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import policy_saver, ou_noise_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from robotini_ddpg.simulator.manager import SimulatorManager
from robotini_ddpg.simulator.environment import create_batched_robotini_env
from robotini_ddpg import util, agent


logger = util.reset_logger()
util.remove_logger("tensorflow")

class Config:
    def __init__(self, path):
        with open(path) as f:
            self.__dict__.update(yaml.safe_load(f.read()))


def train(conf, cache_dir, car_socket_url, log_socket_url, redis_socket_path):
    logger.info("Using config:\n%s", yaml.dump(conf.__dict__))

    cache_dirs = {
        k: os.path.join(cache_dir, k)
        for k in ["collection", "training", "evaluation", "saved_policies", "checkpoints"]}
    summary_writers = {
        k: tf.summary.create_file_writer(cache_dirs[k])
        for k in ["collection", "training", "evaluation"]}

    team_ids = ["env{}".format(i+1) for i in range(conf.num_explore_envs+conf.num_eval_envs)]
    explore_team_ids = team_ids[:conf.num_explore_envs]
    eval_team_ids = team_ids[conf.num_explore_envs:]

    env_kwargs = dict(conf.env_kwargs, redis_socket_path=redis_socket_path)

    explore_teams, explore_env = create_batched_robotini_env(
            explore_team_ids, car_socket_url, env_kwargs)
    eval_teams, eval_env = create_batched_robotini_env(
            eval_team_ids, car_socket_url, env_kwargs)

    tf_agent = agent.create(
        conf.agent_type,
        train_env.time_step_spec(),
        train_env.action_spec(),
        conf.actor,
        conf.critic,
        dict(conf.agent_kwargs, target_update_period=max(1, int(conf.train_batches_per_epoch/conf.target_updates_per_epoch))))

    tf_agent._actor_network.summary(print_fn=logging.info)
    if conf.agent_type == "td3":
        tf_agent._critic_network_1.summary(print_fn=logging.info)
        tf_agent._critic_network_2.summary(print_fn=logging.info)
    else:
        tf_agent._critic_network.summary(print_fn=logging.info)

    def get_training_step():
        return int(tf_agent.train_step_counter.numpy())

    logger.info(("\nStart the agent monitor webapp with:\n\n"
                 "%s robotini_ddpg/monitor/webapp.py %s %s\n"),
                os.environ["PYTHON_BIN"], redis_socket_path, ' '.join(team_ids))

    metric_classes = (
        tf_metrics.MinReturnMetric,
        tf_metrics.MaxReturnMetric,
        tf_metrics.AverageReturnMetric,
        tf_metrics.AverageEpisodeLengthMetric,
    )
    collection_metrics = [M(batch_size=conf.num_explore_envs) for M in metric_classes]
    evaluation_metrics = [M(batch_size=conf.num_eval_envs, buffer_size=conf.num_eval_episodes) for M in metric_classes]

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=explore_env.batch_size,
            max_length=conf.replay_buffer_max_length,
            dataset_drop_remainder=True)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
            explore_env,
            tf_agent.collect_policy,
            observers=[replay_buffer.add_batch] + collection_metrics,
            num_steps=conf.batch_size*conf.train_sequence_length*conf.collect_batches_per_epoch)

    evaluation_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            eval_env,
            tf_agent.policy,
            observers=evaluation_metrics,
            num_episodes=conf.num_eval_episodes)

    if conf.use_tf_functions:
        collect_driver.run = common.function(collect_driver.run)
        tf_agent.train = common.function(tf_agent.train)

    # Initialize replay buffer by doing some exploration in the simulator
    car_suffix = "init"
    with SimulatorManager(explore_teams, log_socket_url, redis_socket_path, car_suffix+"_explore"):
        logger.info("Collecting initial experience")
        collect_driver.run()
    logging.info('\n' + '\n'.join(util.format_metric(m) for m in collection_metrics))

    # Dataset generates trajectories with shape [BxTx...]
    dataset = replay_buffer.as_dataset(
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            sample_batch_size=conf.batch_size,
            num_steps=conf.train_sequence_length + 1).prefetch(tf.data.experimental.AUTOTUNE)
    iterator = iter(dataset)

    def train_step():
        experience_batch, _ = next(iterator)
        loss = tf_agent.train(experience_batch)
        return loss

    if conf.use_tf_functions:
        train_step = common.function(train_step)

    eval_policy_saver = policy_saver.PolicySaver(
            tf_agent.policy,
            batch_size=eval_env.batch_size,
            train_step=tf_agent.train_step_counter)

    train_checkpointer = common.Checkpointer(
            ckpt_dir=cache_dirs["checkpoints"],
            max_to_keep=conf.max_checkpoints_to_keep,
            agent=tf_agent,
            policy=tf_agent.policy,
            replay_buffer=replay_buffer,
            global_step=tf_agent.train_step_counter)
    train_checkpointer.initialize_or_restore()

    # Start main training loop
    time_step = None
    policy_state = tf_agent.collect_policy.get_initial_state(explore_env.batch_size)

    for epoch in range(1, conf.max_num_epochs+1):
        logging.info("Epoch %d - training step: %d, frames in replay buffer: %d",
                epoch,
                get_training_step(),
                replay_buffer._num_frames().numpy())
        car_suffix = "step{:d}".format(get_training_step())

        with SimulatorManager(explore_teams, log_socket_url, redis_socket_path, car_suffix+"_explore"):
            logger.info("Collecting experience")
            time_step, policy_state = collect_driver.run(
                    time_step=time_step,
                    policy_state=policy_state)
        logging.info('\n' + '\n'.join(util.format_metric(m) for m in collection_metrics))
        if util.find_first(collection_metrics, "AverageEpisodeLength").result() == 0:
            logging.info("Not writing episode summaries for collection metrics since no car managed to do a full episode")
        else:
            with summary_writers["collection"].as_default():
                for m in collection_metrics:
                    m.tf_summaries(train_step=tf_agent.train_step_counter)

        num_train_steps = conf.train_batches_per_epoch
        logger.info("Training for %d steps (batches)", num_train_steps)
        time_per_step = np.zeros(num_train_steps, np.float32)
        train_losses = np.zeros(num_train_steps, np.float32)
        with summary_writers["training"].as_default(step=tf_agent.train_step_counter):
            for i in range(num_train_steps):
                t0 = time.perf_counter()
                train_loss = train_step()
                t1 = time.perf_counter()
                time_per_step[i] = t1 - t0
                train_losses[i] = train_loss.loss
            metrics_str = [
                "\t\t AverageTrainingLoss = {:.3f}".format(train_losses.mean()),
                "\t\t AverageTrainingStepSec = {:.3f}".format(time_per_step.mean()),
            ]
            logging.info('\n' + '\n'.join(metrics_str))

        if epoch == 1 or get_training_step() % conf.eval_interval == 0:
            logger.info("Evaluating actor policy")
            with SimulatorManager(eval_teams, log_socket_url, redis_socket_path, car_suffix+"_eval"):
                evaluation_driver.run(
                        time_step=None,
                        policy_state=tf_agent.policy.get_initial_state(eval_env.batch_size))
            logging.info('\n' + '\n'.join(util.format_metric(m) for m in evaluation_metrics))
            with summary_writers["evaluation"].as_default():
                for m in evaluation_metrics:
                    m.tf_summaries(train_step=tf_agent.train_step_counter)

            eval_avg_return = util.find_first(evaluation_metrics, "AverageReturn")
            eval_avg_return = eval_avg_return.result().numpy()
            policy_save_dir = os.path.join(
                    cache_dirs["saved_policies"],
                    "{:d}_Step{:d}_AvgEvalReturn{:d}".format(
                        round(time.time()),
                        get_training_step(),
                        round(eval_avg_return)))
            _, best_value_so_far = util.get_best_saved_policy(cache_dirs["saved_policies"])
            if eval_avg_return > best_value_so_far:
                logger.info("Saving new best policy to '%s'", policy_save_dir)
                eval_policy_saver.save(policy_save_dir)

        if get_training_step() % conf.checkpoint_interval == 0:
            train_checkpointer.save(tf_agent.train_step_counter)

        for m in collection_metrics + evaluation_metrics:
            m.reset()


def run(config_path, cache_dir, **kwargs):
    train(Config(config_path), os.path.abspath(cache_dir), **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--car-socket-url", type=str, required=True)
    parser.add_argument("--log-socket-url", type=str, required=True)
    parser.add_argument("--redis-socket-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=True)
    args, sys.argv[1:] = parser.parse_known_args()
    run(**vars(args))
