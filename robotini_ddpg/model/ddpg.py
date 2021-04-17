import tensorflow as tf
from tf_agents.agents.ddpg import actor_rnn_network
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.ddpg import ddpg_agent


def create_agent(time_step_spec, action_spec, actor_conf, critic_conf, target_update_period):
    tf_agent = ddpg_agent.DdpgAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_rnn_network.ActorRnnNetwork(
                time_step_spec.observation,
                action_spec,
                **actor_conf["kwargs"]),
            critic_network=critic_rnn_network.CriticRnnNetwork(
                (time_step_spec.observation, action_spec),
                **critic_conf["kwargs"]),
            actor_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=actor_conf["learning_rate"]),
            critic_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=critic_conf["learning_rate"]),
            target_update_period=target_update_period,
            summarize_grads_and_vars=True,
            debug_summaries=False)
    tf_agent.initialize()
    return tf_agent
