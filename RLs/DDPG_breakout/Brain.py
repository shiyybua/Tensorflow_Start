# -*- coding: utf-8 -*

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
import numpy as np
import gym
from collections import deque
import random

ENV_NAME = 'Breakout-v0'
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
env = wrappers.Monitor(env, './video', force=True, video_callable=lambda x: x % 20 == 0)
# env = Monitor(env, directory='./videos', video_callable=lambda x: True, resume=True)
# env = env.unwrapped


# s_dim = env.observation_space.shape[0] # Box(3,)
# a_dim = env.action_space.shape[0]   # Box(1,)
# a_bound = env.action_space.high # high = 2

class Net:
    def __init__(self, image_shape=[84, 84, 1]):
        self.image_shape = image_shape
        self.epsilon = 0.9
        self.memory = deque(maxlen=MAX_MEMORY_SIZE)
        self.sess = tf.Session()

        self.state = tf.placeholder(dtype=tf.float32, shape=[None] + self.image_shape) / 255.0
        self.state_ = tf.placeholder(dtype=tf.float32, shape=[None] + self.image_shape) / 255.0
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None])

        # target的作用主要是针对下一个状态的处理。(做预测)
        self.action = self._action_net(self.state, "action_net_eval")  # 实时更新
        self.action_ = self._action_net(self.state_, "action_net_target", trainable=False)   # 定时更新

        self.critic = self._critic_net(self.state, self.action, "critic_net_eval")    # 实时更新
        self.critic_ = self._critic_net(self.state_, self.action_, "critic_net_target", trainable=False)     # 定时更新

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='action_net_eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='action_net_target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_net_eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_net_target')

        self.shared_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shared')

        # Action的目的就是要最大化critic的输出值。
        self.action_loss = - tf.reduce_mean(self.critic)
        self.train_action = tf.train.AdamOptimizer(0.01).minimize(self.action_loss, var_list=self.shared_params)

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
        state = np.array(state)
        action = np.array(action)
        state_ = np.array(state_)
        self.memory.append([state, action, reward, state_])

    def _action_net(self, state, name_scope, trainable=True):
        with tf.variable_scope(name_scope):
            # Three convolutional layers
            conv1 = tf.contrib.layers.conv2d(
                state, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1", trainable=trainable)
            conv2 = tf.contrib.layers.conv2d(
                conv1, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2", trainable=trainable)
            # Fully connected layer
            fc1 = tf.contrib.layers.fully_connected(
                inputs=tf.contrib.layers.flatten(conv2),
                num_outputs=256,
                scope="fc1", trainable=trainable)
            logits = tf.contrib.layers.fully_connected(fc1, 4, activation_fn=None, trainable=trainable)
            probs = tf.nn.softmax(logits) + 1e-8
            # action_prob = tf.reduce_max(self.probs, axis=1)
            return probs

    def _critic_net(self, state, action, name_scope, trainable=True):
        with tf.variable_scope(name_scope):
            # Three convolutional layers
            conv1 = tf.contrib.layers.conv2d(
                state, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1", trainable=trainable)
            conv2 = tf.contrib.layers.conv2d(
                conv1, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2", trainable=trainable)
            # Fully connected layer
            fc1 = tf.contrib.layers.fully_connected(
                inputs=tf.contrib.layers.flatten(conv2),
                num_outputs=256,
                scope="fc1", trainable=trainable)
            critic_value = tf.contrib.layers.fully_connected(fc1, 4, activation_fn=None, trainable=trainable)

            # W_c = tf.get_variable("critic_variable", shape=[1])
            W_c = tf.get_variable("action_variable", shape=[4, 1], trainable=trainable)
            W_a = tf.get_variable("action_variable", shape=[4, 1], trainable=trainable)
            b = tf.get_variable("c_net_b", shape=[1], trainable=trainable)
            q = tf.nn.sigmoid(tf.matmul(critic_value, W_c) + tf.matmul(action, W_a) + b)

            return q

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

            # action_loss = self.sess.run(self.action_loss, feed_dict={self.state: state_batch})
            # critic_loss = self.sess.run(self.critic_loss, feed_dict={self.state: state_batch, self.reward: reward_batch,
            #                                         self.state_: next_state_batch, self.action: action_batch})
            # print action_loss, critic_loss
            # tf.summary.scalar('reward', loss)
            # self.train_writer.add_summary(summary, self.translate_counter)

        self.translate_counter += 1


    def choose_action(self, state):
        state = np.array(state)
        action_probs = self.sess.run(self.action, feed_dict={self.state: state})
        if np.random.uniform() < self.epsilon:
            return np.argmax(action_probs)
        else:
            return np.random.randint(0, len(action_probs))


if __name__ == '__main__':
    ddpg = Net()
    counter = 0

    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            # a = np.random.normal(a, env.action_space.high)
            s_, r, done, info = env.step(a)

            # reward 是否除以10 影响不大。只是更倾向于把值拉到0，1附近
            ddpg.store(s, a, r, s_)

            if counter > MAX_MEMORY_SIZE:
                ddpg.learn()

            s = s_
            ep_reward += r


            if j == MAX_EP_STEPS - 1 or done:
                print('Episode:', i, ' Reward: %i' % int(ep_reward))
                # if ep_reward > -1000: RENDER = True
                break

            counter += 1