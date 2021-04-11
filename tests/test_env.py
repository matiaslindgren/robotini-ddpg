import argparse
import os
import sys

from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment, batched_py_environment
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

from robotini_ddpg.model import env, snail
from robotini_ddpg.simulator import manager


def run(car_socket_url, log_socket_url, redis_socket_path, num_cars, num_episodes, isolate_envs):
    team_ids = ["test_snail{}".format(i) for i in range(1, num_cars+1)]

    with manager.SimulatorManager(team_ids, car_socket_url, log_socket_url, redis_socket_path) as simulator_manager:
        envs = [env.RobotiniCarEnv(simulator_manager, team_id, redis_socket_path) for team_id in team_ids]
        batch_env = batched_py_environment.BatchedPyEnvironment(envs, multithreading=not isolate_envs)
        tf_env = tf_py_environment.TFPyEnvironment(batch_env, isolation=isolate_envs)

        snail_policy = snail.BlueSnailPolicy(tf_env.time_step_spec(), tf_env.action_spec(), clip=False)

        avg_return = tf_metrics.AverageReturnMetric(batch_size=num_cars)
        observers = [avg_return]
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
                tf_env, snail_policy, observers, num_episodes=num_episodes)
        driver.run = common.function(driver.run)

        initial_time_step = tf_env.reset()
        driver.run(initial_time_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cars", type=int, default=3)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--isolate-envs", action="store_true", default=False)
    parser.add_argument("car_socket_url", type=str)
    parser.add_argument("log_socket_url", type=str)
    parser.add_argument("redis_socket_path", type=str)
    args, sys.argv[1:] = parser.parse_known_args()
    run(**vars(args))
