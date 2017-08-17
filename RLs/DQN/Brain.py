# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np
from collections import deque
import random


class DQN:
    def __init__(self, net_name, sess,is_gpu_available=None, image_shape=[84, 84, 1]):
        '''
        :param image_shape: [height, width, channel], (210, 160, 3)
        '''
        self.sess = sess
        self.memory_size = 10000
        self.memory = deque(maxlen=self.memory_size)
        self.image_shape = image_shape
        self.net_name = net_name
        self.observation = tf.placeholder(dtype=tf.float32, shape=[None] + self.image_shape) / 255.0
        self.target_value = tf.placeholder(dtype=tf.float32, shape=[None, 4])
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, 4])
        self.replace_target_iter = 0
        self.steps = 0
        self.batch_size = 64
        self.gamma = 0.9
        self.epsilon = 0.9

        with tf.name_scope(net_name):
            if is_gpu_available is not None:
                for d in is_gpu_available:
                    with tf.device(d):
                        self._build_net()
            else:
                self._build_net()

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


    def _init_weight(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, collections=[self.net_name, tf.GraphKeys.GLOBAL_VARIABLES])

    def _init_biases(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, collections=[self.net_name, tf.GraphKeys.GLOBAL_VARIABLES])

    def conv_layer(self, x, W, b, activation_func=tf.nn.relu):
        return activation_func(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)

    def max_pooling(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    def fc_layer(self, x, W, b, activation_func=tf.nn.relu):
        if activation_func is not None:
            return activation_func(tf.nn.xw_plus_b(x, W, b))
        else:
            return tf.nn.xw_plus_b(x, W, b)


    def _build_net(self):
        with tf.name_scope("conv1"):
            # [None] + self.image_shape
            conv1_W = self._init_weight([10,10,1,32])
            conv1_b = self._init_biases([32])
            conv1_output = self.conv_layer(self.observation, conv1_W, conv1_b)
            conv1_output = self.max_pooling(conv1_output)   # 42 * 42

        with tf.name_scope("conv2"):
            conv2_W = self._init_weight([10,10,32,64])
            conv2_b = self._init_biases([64])
            conv2_output = self.conv_layer(conv1_output, conv2_W, conv2_b)
            conv2_output = self.max_pooling(conv2_output)    # 21 * 21

        with tf.name_scope("conv3"):
            conv3_W = self._init_weight([10,10,64,64])
            conv3_b = self._init_biases([64])
            conv3_output = self.conv_layer(conv2_output, conv3_W, conv3_b)
            conv3_output = self.max_pooling(conv3_output)    # 11 * 11

        with tf.name_scope("fc1"):
            conv3_flat = tf.reshape(conv3_output, [-1, 11*11*64])
            fc1_W = self._init_weight([11*11*64, 620])
            fc1_b = self._init_biases([620])
            fc1_ouput = self.fc_layer(conv3_flat, fc1_W, fc1_b)
            fc1_ouput = tf.nn.dropout(fc1_ouput,keep_prob=0.5)

        with tf.name_scope("fc2"):
            fc2_W = self._init_weight([620, 4])
            fc2_b = self._init_biases([4])

            self.variable_summaries(fc2_W)
            self.variable_summaries(fc2_b)

            # 把算出的值直接当Value值，则不能加softmax
            self.net_ouput = self.fc_layer(fc1_ouput, fc2_W, fc2_b, tf.nn.tanh)

        with tf.name_scope("loss"):
            # Q_action = tf.reduce_sum(tf.multiply(self.net_ouput, self.action), axis=1)
            # Q_action = self.net_ouput
            self.loss = tf.reduce_mean(tf.square(self.target_value - self.net_ouput))
            optimizer = tf.train.AdamOptimizer()
            self.train = optimizer.minimize(self.loss)

    def choose_action(self, observation):
        prob_weights = self.sess.run(self.net_ouput, feed_dict={self.observation: observation[np.newaxis, :]})
        prob_weights = prob_weights.ravel()
        # return np.argmax(prob_weights)
        if np.random.uniform() < self.epsilon:
            return np.argmax(prob_weights)
        else:
            return np.random.randint(0, len(prob_weights))

    def store_transition(self, observation, action, reward, _observation):
        self.memory.append((observation, action, reward, _observation))
        self.steps += 1

    def learn(self, target_net, merged, train_writer):
        batch = random.sample(self.memory, self.batch_size)
        observation_batch = [t[0] for t in batch]
        action_batch = [t[1] for t in batch]
        reward_batch = [t[2] for t in batch]
        next_observation_batch = [t[3] for t in batch]

        q_next = self.sess.run(target_net.net_ouput, feed_dict={target_net.observation: next_observation_batch})
        # y = reward_batch + self.gamma * np.max(q_next, axis=1)
        # y = reward_batch + self.gamma * q_next

        q_next = self.gamma * q_next
        for i in range(self.batch_size):
            element = q_next[i]
            max_arg = np.argmax(element)
            q_next[i][max_arg] += reward_batch[i]

        y = q_next
        # y = y.reshape([self.batch_size,1])

        # convert action into one-hot representation
        action_batch = np.eye(4)[action_batch]
        _, summary = self.sess.run([self.train, merged], feed_dict={self.observation: observation_batch, self.target_value:y,
                                             self.action: action_batch})

        self.replace_target_iter += 1
        if self.replace_target_iter % 500 == 0:
            print 'translated'
            translate = [tf.assign(e1, e2) for e1, e2 in zip(tf.get_collection('target_net'), tf.get_collection('Q_net'))]
            self.sess.run(translate)
            self.replace_target_iter = 0

        if self.replace_target_iter % 20 == 0:
            print 'loss:', self.sess.run(self.loss, feed_dict={self.observation: observation_batch, self.target_value: y,
                                                      self.action: action_batch})

        tf.summary.scalar('reward', sum(reward_batch) * 1.0 / len(reward_batch))
        train_writer.add_summary(summary, self.steps)






