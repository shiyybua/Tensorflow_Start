# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np
import gym
from collections import deque
import random
from gym.wrappers import Monitor

ENV_NAME = 'Pendulum-v0'
RESOURCE_PATH = '../../resource/'
MAX_MEMORY_SIZE = 7000
BATCH_SIZE = 64
GAMMA = 0.99
TRANSLATE_STEP = 500
MAX_EPISODES = 70
MAX_EP_STEPS = 400
RENDER = False

from gym import wrappers

env = gym.make(ENV_NAME)
env = wrappers.Monitor(env, './video', force=True)
# env = Monitor(env, directory='./videos', video_callable=lambda x: True, resume=True)
# env = env.unwrapped


s_dim = env.observation_space.shape[0] # Box(3,)
a_dim = env.action_space.shape[0]   # Box(1,)
a_bound = env.action_space.high # high = 2


class Net:
    def __init__(self, state_dim, action_dim, action_bound):
        self.memory = deque(maxlen=MAX_MEMORY_SIZE)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sess = tf.Session()

        self.state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        self.state_ = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.action_bound = action_bound

        # target的作用主要是针对下一个状态的处理。(做预测)
        self.action = self._action_net(self.state, "action_net_eval")  # 实时更新
        self.action_ = self._action_net(self.state_, "action_net_target", trainable=False)   # 定时更新

        self.critic = self._critic_net(self.action, self.state, "critic_net_eval")    # 实时更新
        self.critic_ = self._critic_net(self.action_, self.state_, "critic_net_target", trainable=False)     # 定时更新

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='action_net_eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='action_net_target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_net_eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_net_target')

        # Action的目的就是要最大化critic的输出值。
        self.action_loss = - tf.reduce_mean(self.critic)
        self.train_action = tf.train.AdamOptimizer(0.01).minimize(self.action_loss, var_list=self.ae_params)

        self.current_reward = GAMMA * self.critic_ + self.reward
        # Critic的目的是最小化和预期之间的误差, labels, predictions 顺序无所谓
        self.critic_loss = tf.losses.mean_squared_error(labels=self.current_reward, predictions=self.critic)
        self.train_critic = tf.train.AdamOptimizer(0.01).minimize(self.critic_loss, var_list=self.ce_params)

        self.sess.run(tf.global_variables_initializer())

        self.translate_counter = 0

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(RESOURCE_PATH + 'DDPG_record',
                                             self.sess.graph)

    def store(self, state, action, reward, state_):
        state = np.array(state).reshape(len(state))
        action = np.array(action).reshape(len(action))
        state_ = np.array(state_).reshape(len(state_))
        self.memory.append([state, action, reward, state_])

    def _action_net(self, state, name_scope, trainable=True):
        with tf.variable_scope(name_scope):
            # 一个全连接层就够了
            layer1 = tf.layers.dense(state, 30, activation=tf.nn.relu, trainable=trainable)
            action_value = tf.layers.dense(layer1, 1, trainable=trainable, activation=tf.nn.tanh)
            # 恢复到实际值范围中，也不是一定要做的操作。
            action_value = tf.multiply(action_value, self.action_bound, name='scaled_a')
            return action_value

    def _critic_net(self, action, state, name_scope, trainable=True):
        with tf.variable_scope(name_scope):
            units = 30
            W_a = tf.get_variable("c_net_a_w", shape=[self.action_dim, units], trainable=trainable)
            W_s = tf.get_variable("c_net_c_w", shape=[self.state_dim, units], trainable=trainable)
            b = tf.get_variable("c_net_b", shape=[1,units], trainable=trainable)
            q = tf.nn.relu(tf.matmul(action, W_a) + tf.matmul(state, W_s) + b)
            return tf.layers.dense(q, 1, trainable=trainable)

    def learn(self):
        batch = random.sample(self.memory, BATCH_SIZE)
        state_batch = [t[0] for t in batch]
        action_batch = [t[1] for t in batch]
        reward_batch = [t[2] for t in batch]
        next_state_batch = [t[3] for t in batch]

        self.sess.run(self.train_action, feed_dict={self.state: state_batch})
        # 即使不feed action_batch是不会报错的，原因是state_batch可以重新算一次action，但是这个新的action值和原来batch里面的值就不一样了。
        # 而且参与运算的variable 设定是只是ce_params，根本就不会更新到action里面的value值。所以直接会导致结果不对。
        self.sess.run(self.train_critic, feed_dict={self.state: state_batch, self.reward: reward_batch,
                                                    self.state_: next_state_batch, self.action: action_batch})

        if self.translate_counter % TRANSLATE_STEP == 0:
            translate = [tf.assign(e1, e2) for e1, e2 in
                         zip(self.at_params, self.ae_params)]
            self.sess.run(translate)

            translate = [tf.assign(e1, e2) for e1, e2 in
                         zip(self.ct_params, self.ce_params)]
            self.sess.run(translate)

            action_loss = self.sess.run(self.action_loss, feed_dict={self.state: state_batch})
            critic_loss = self.sess.run(self.critic_loss, feed_dict={self.state: state_batch, self.reward: reward_batch,
                                                    self.state_: next_state_batch, self.action: action_batch})
            print action_loss, critic_loss
            # tf.summary.scalar('reward', loss)
            # self.train_writer.add_summary(summary, self.translate_counter)

        self.translate_counter += 1


    def choose_action(self, state):
        state = np.array(state).reshape([1, 3])
        action = self.sess.run(self.action, feed_dict={self.state: state})
        return action



if __name__ == '__main__':
    ddpg = Net(s_dim, a_dim, a_bound)
    counter = 0
    var = 3  # control exploration

    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            # clip表示小于np.random.normal(a, var)中小于env.action_space.low的数归于env.action_space.low,大于...high的...归于high
            # 中间部分不变。
            a = np.clip(np.random.normal(a, var), env.action_space.low, env.action_space.high)  # add randomness to action selection for exploration
            # a = np.random.normal(a, env.action_space.high)
            s_, r, done, info = env.step(a)

            # reward 是否除以10 影响不大。只是更倾向于把值拉到0，1附近
            ddpg.store(s, a, r/10.0, s_)

            if counter > MAX_MEMORY_SIZE:
                var *= .9995  # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r


            if j == MAX_EP_STEPS - 1 or done:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var,)
                if ep_reward > -1000: RENDER = True
                break


            counter += 1