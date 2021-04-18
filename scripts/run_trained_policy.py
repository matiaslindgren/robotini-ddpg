import argparse
import sys

import tensorflow as tf
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.utils import common

from robotini_ddpg.model import env, features
from robotini_ddpg.simulator import manager


def run(policy_dir, car_socket_url, log_socket_url, redis_socket_path, num_episodes):
    policy = tf.saved_model.load(policy_dir)
    env_kwargs = {
        "redis_socket_path": redis_socket_path,
        "max_steps_per_episode": 1000,
        "laps_per_episode": 2,
    }
    teams, run_env = env.create_batched_tf_env(["DDPG-1", "DDPG-2"], car_socket_url, env_kwargs)
    driver = dynamic_episode_driver.DynamicEpisodeDriver(
            run_env,
            policy,
            num_episodes=num_episodes)
    driver.run = common.function(driver.run)
    time_step = run_env.reset()
    policy_state = policy.get_initial_state()
    with manager.SimulatorManager(teams, log_socket_url, redis_socket_path) as mgr:
        driver.run(time_step=time_step, policy_state=policy_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dir", type=str, required=True)
    parser.add_argument("--car-socket-url", type=str, required=True)
    parser.add_argument("--log-socket-url", type=str, required=True)
    parser.add_argument("--redis-socket-path", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=3)
    args, sys.argv[1:] = parser.parse_known_args()
    run(**vars(args))
