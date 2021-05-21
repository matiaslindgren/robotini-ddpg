import argparse
import sys

import yaml
import tensorflow as tf
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.utils import common

from robotini_ddpg.simulator.manager import SimulatorManager
from robotini_ddpg.simulator.environment import create_batched_robotini_env


def run(policy_dir, config_path, car_socket_url, log_socket_url, redis_socket_path, num_episodes):
    with open(config_path) as f:
        env_kwargs = yaml.safe_load(f)["env_kwargs"]
    policy = tf.saved_model.load(policy_dir)
    env_kwargs = dict(env_kwargs, redis_socket_path=redis_socket_path)
    teams, run_env = create_batched_robotini_env(["smol-brain"], car_socket_url, env_kwargs)
    driver = dynamic_episode_driver.DynamicEpisodeDriver(
            run_env,
            policy,
            num_episodes=num_episodes)
    driver.run = common.function(driver.run)
    time_step = run_env.reset()
    policy_state = policy.get_initial_state()
    with SimulatorManager(teams, log_socket_url, redis_socket_path) as manager:
        driver.run(time_step=time_step, policy_state=policy_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dir", type=str, required=True)
    parser.add_argument("--car-socket-url", type=str, required=True)
    parser.add_argument("--log-socket-url", type=str, required=True)
    parser.add_argument("--redis-socket-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=3)
    args, sys.argv[1:] = parser.parse_known_args()
    run(**vars(args))
