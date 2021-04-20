import tensorflow as tf
from tf_agents.agents.ddpg import actor_rnn_network
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.ddpg import ddpg_agent


def create_ddpg_agent(time_step_spec, action_spec, actor_conf, critic_conf, ddpg_kwargs):
    actor = actor_rnn_network.ActorRnnNetwork(
        time_step_spec.observation,
        action_spec,
        **actor_conf["kwargs"]
    )
    critic = critic_rnn_network.CriticRnnNetwork(
        (time_step_spec.observation, action_spec),
        **critic_conf["kwargs"],
    )
    agent = ddpg_agent.DdpgAgent(
        time_step_spec,
        action_spec,
        actor_network=actor,
        critic_network=critic,
        actor_optimizer=tf.keras.optimizers.Adam(learning_rate=actor_conf["learning_rate"]),
        critic_optimizer=tf.keras.optimizers.Adam(learning_rate=critic_conf["learning_rate"]),
        **ddpg_kwargs)
    agent.initialize()
    return agent
