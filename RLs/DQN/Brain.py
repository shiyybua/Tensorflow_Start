import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self, net_name, image_shape=[210, 160, 3]):
        '''
        :param image_shape: [height, width, channel], (210, 160, 3)
        '''
        self.image_shape = image_shape
        self.net_name = net_name
        self.observation = tf.placeholder(dtype=tf.float32, shape=[None] + self.image_shape) / 255.0
        with tf.name_scope(net_name):
            self._build_net()

    def _init_weight(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, collections=[self.net_name, tf.GraphKeys.GLOBAL_VARIABLES])

    def _init_biases(self, shape):
        initial = tf.contants(0.1, shape=shape)
        return tf.Variable(initial, collections=[self.net_name, tf.GraphKeys.GLOBAL_VARIABLES])

    def conv_layer(self, x, W, b, activation_func=tf.nn.relu):
        return activation_func(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)

    def max_pooling(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    def fc_layer(self, x, W, b, activation_func=tf.nn.relu):
        return activation_func(tf.nn.xw_plus_b(x, W, b))


    def _build_net(self):
        with tf.name_scope("conv1"):
            # [None] + self.image_shape
            conv1_W = self._init_weight([10,10,3,32])
            conv1_b = self._init_biases([32])
            conv1_output = self.conv_layer(self.observation, conv1_W, conv1_b)
            conv1_output = self.max_pooling(conv1_output)   # 105 * 80

        with tf.name_scope("conv2"):
            conv2_W = self._init_weight([10,10,32,64])
            conv2_b = self._init_biases([64])
            conv2_output = self.conv_layer(conv1_output, conv2_W, conv2_b)
            conv2_output = self.conv_layer(conv2_output)    # 53 * 40

        with tf.name_scope("conv3"):
            conv3_W = self._init_weight([10,10,64,64])
            conv3_b = self._init_biases([64])
            conv3_output = self.conv_layer(conv2_output, conv3_W, conv3_b)
            conv3_output = self.conv_layer(conv3_output)    # 27 * 20

        with tf.name_scope("fc1"):
            conv3_flat = tf.reshape(conv3_output, [-1, 27*20*64])
            fc1_W = self._init_weight([27*20*64, 1024])
            fc1_b = self._init_biases([1024])
            fc1_ouput = self.fc_layer(conv3_flat, fc1_W, fc1_b)
            fc1_ouput = tf.nn.dropout(fc1_ouput)

        with tf.name_scope("fc2"):
            fc2_W = self._init_weight([1024, 4])
            fc2_b = self._init_biases([4])
            self.net_ouput = self.fc_layer(fc1_ouput, fc2_W, fc2_b)


class Brain():
    def __init__(self):
        self.Q_net = DQN("Q_net")
        self.target_net = DQN("target_net")
        self.memory = np.array()








