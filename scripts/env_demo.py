import argparse
import logging
import os
import sys

import yaml
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment, batched_py_environment
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

from robotini_ddpg.simulator.manager import SimulatorManager
from robotini_ddpg.simulator.environment import create_batched_robotini_env
from robotini_ddpg import snail_policy, util

util.reset_logger(level=logging.DEBUG)


def run(config_path, car_socket_url, log_socket_url, redis_socket_path, num_cars, num_steps, isolate_envs):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    team_ids = ["test_snail{}".format(i) for i in range(1, num_cars+1)]
    env_kwargs = dict(config["env_kwargs"], redis_socket_path=redis_socket_path)
    teams, tf_env = create_batched_robotini_env(team_ids, car_socket_url, env_kwargs)

    snail = snail_policy.BlueSnailPolicy(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            forward_action=0.04,
            clip=False)

    avg_return = tf_metrics.AverageReturnMetric(batch_size=num_cars)
    observers = [avg_return]
    driver = dynamic_step_driver.DynamicStepDriver(
            tf_env, snail, observers, num_steps=num_steps)
    driver.run = common.function(driver.run)

    with SimulatorManager(teams, log_socket_url, redis_socket_path):
        initial_time_step = tf_env.reset()
        driver.run(initial_time_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--car-socket-url", type=str, required=True)
    parser.add_argument("--log-socket-url", type=str, required=True)
    parser.add_argument("--redis-socket-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--num-cars", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--isolate-envs", action="store_true", default=False)
    args, sys.argv[1:] = parser.parse_known_args()
    run(**vars(args))
