# -*- coding: utf-8 -*-
# @Time : 2022/6/2 下午1:45
# @Author :  wangshulei
# @FileName: distri_actor_critic_net.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np


class test_actor:
    def __init__(self,
                 obs_dim=7,
                 act_dim=5,
                 actor_learning_rate=1e-3,
                 agent_index=0,
                 trainable=False,
                 action_span=0.5):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_span = action_span
        self.actor_learning_rate = actor_learning_rate
        self.actor = self.__actor_net(trainable, agent_index, action_span)

    def __actor_net(self, trainable, agent_index, action_span):
        # MADDPG的演员是根据自己智能体的观测值来得到动作的
        input_s = tf.keras.Input(shape=(self.obs_dim,), dtype="float32")
        dense1 = tf.keras.layers.Dense(128, activation='relu')(input_s)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
        actor_output_4 = tf.keras.layers.Dense(self.act_dim - 1, activation='tanh')(dense2)  # 使用tanh输出动作
        actor_output_1 = tf.keras.layers.Dense(self.act_dim - 4, activation='sigmoid')(dense2)  # 使用sigmoid执行攻击分类
        actor_output = tf.keras.layers.Lambda(lambda x: x * np.array(action_span))(actor_output_4)
        actor_model = tf.keras.Model(inputs=input_s,
                                     outputs=[actor_output_1, actor_output],
                                     trainable=trainable,
                                     name=f'agent{agent_index}')
        actor_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.actor_learning_rate))
        return actor_model
