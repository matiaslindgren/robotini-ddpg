import tensorflow as tf
from tf_agents.agents.ddpg import actor_rnn_network
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.td3 import td3_agent


class BadConfig(Exception):
    pass


def create(agent_type, time_step_spec, action_spec, actor_conf, critic_conf, network_update_period, agent_kwargs):
    agent_kwargs = dict(agent_kwargs, target_update_period=network_update_period)
    if agent_type == "ddpg":
        Agent = ddpg_agent.DdpgAgent
    elif agent_type == "td3":
        Agent = td3_agent.Td3Agent
        agent_kwargs = dict(agent_kwargs, actor_update_period=network_update_period)
    else:
        raise BadConfig("unknown agent type '{}'".format(agent_type))
    actor = actor_rnn_network.ActorRnnNetwork(
        time_step_spec.observation,
        action_spec,
        **actor_conf["kwargs"]
    )
    critic = critic_rnn_network.CriticRnnNetwork(
        (time_step_spec.observation, action_spec),
        **critic_conf["kwargs"],
    )
    agent = Agent(
        time_step_spec,
        action_spec,
        actor_network=actor,
        critic_network=critic,
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=actor_conf["learning_rate"]),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=critic_conf["learning_rate"]),
        **agent_kwargs)
    agent.initialize()
    return agent
