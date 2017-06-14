# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np


RESOURCE_PATH = '../resource/'
class Net:
    def __init__(self, name, sess):
        expected_weight = [0.2, 0.5, 0.8, 1.3, 6.1, 7.2, 1, 8]
        expected_bias = 10
        # 一般数值类型用numpy里的
        x_data = np.random.rand(100, 8).astype(np.float32)
        self.y_data = x_data * expected_weight + expected_bias

        self.sess = sess
        c_names = [name, tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.name_scope(name+'var'):
            self.w = tf.Variable(tf.truncated_normal([8], stddev=0.1), collections=c_names)
            self.b = tf.Variable(tf.truncated_normal([1], stddev=0.1), collections=c_names)
            y = x_data * self.w + self.b
            self.variable_summaries(self.w,'w')
            self.variable_summaries(self.b,'b')

        with tf.name_scope(name+"loss"):
            # 设置、最小化损失函数. 这里是y 和 y_data都是向量，数组
            loss = tf.reduce_mean(tf.square(y - self.y_data))
            # 用梯度下降。learning rate 是0.5
            optimizer = tf.train.GradientDescentOptimizer(0.5)
            # 目标是最小化损失函数
            self.train = optimizer.minimize(loss)

    def assign(self):
        translate = [tf.assign(e1,e2) for e1,e2 in zip(tf.get_collection('net1'),tf.get_collection('net2'))]
        # 需要run！
        self.sess.run(translate)

    def variable_summaries(self, var, var_name='summaries'):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(var_name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(RESOURCE_PATH + 'DQN_record',
                                         sess.graph)

    net1 = Net('net1',sess)
    net2 = Net('net2',sess)
    init = tf.global_variables_initializer()
    sess.run(init)
    merged = tf.summary.merge_all()
#----------- 2个net共用一个sess，相互传递参数有效-----------
    print 'net1,w:', sess.run(net1.w)
    print 'net2,w:', sess.run(net2.w)

    net1.assign()

    print 'net1,w:', sess.run(net1.w)
    print 'net2,w:', sess.run(net2.w)

    for step in range(200):
        _, summary = sess.run([net2.train, merged])
        if step % 20 == 0:
            print step, sess.run(net2.w), sess.run(net2.b)
            train_writer.add_summary(summary, step)

    net1.assign()

    for step in range(200, 400):
        _, summary = sess.run([net1.train, merged])
        if step % 20 == 0:
            print step, sess.run(net1.w), sess.run(net1.b)
            train_writer.add_summary(summary, step)