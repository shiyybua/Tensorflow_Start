# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np

true_weight = [0.2, 0.5, 0.8, 0.1, 0.3, 0.1, 0.25, 0.36, 0.88, 0.99]
true_bias = 10

def linear():
    # TODO: x_data值 为什么不能过大
    x_data = np.random.rand(100, 10).astype(np.float32)
    y_data = x_data * true_weight + true_bias

    # OR change to constant
    W_p1 = tf.Variable([0.1, 0.3, 0.5], trainable=False)
    W_p2 = tf.Variable(tf.random_uniform([7], -0.1, 1.0))
    W = tf.concat([W_p1, W_p2], axis=0)
    # W = tf.Variable(tf.random_uniform([10], -0.1, 1.0))
    b = tf.Variable(tf.zeros([1]))

    y = x_data * W + b

    loss_func = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss_func)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for step in range(20000):
        sess.run(train)
        if step % 200 == 0:
            print step, sess.run(W), sess.run(b), sess.run(loss_func)
            # print step, sess.run(loss_func)

linear()

'''
以下是feed variable
data = np.random.uniform(-1, 1, [10,20])

v = tf.placeholder(tf.float32, shape=[10, 20])
X = tf.Variable(v)

print type(data[0][0])
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init, feed_dict={t.v: data})
    print sess.run(t.X)
'''
