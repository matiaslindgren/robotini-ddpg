import tensorflow as tf
from tf_agents.agents.ddpg import actor_rnn_network
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.td3 import td3_agent


class BadConfig(Exception):
    pass


def create_agent(agent_type, time_step_spec, action_spec, actor_conf, critic_conf, agent_kwargs):
    if conf.agent_type == "ddpg":
        Agent = ddpg_agent.DdpgAgent
    elif conf.agent_type == "td3":
        Agent = td3_agent.Td3Agent
    else:
        raise BadConfig("unknown agent type '{}'".format(conf.agent_type))
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
