# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np

true_weight = [0.3, 0.5, 0.6]
true_bias = 1

# bias = tf.placeholder(tf.float32, [None, 1])
# weight = tf.placeholder(tf.float32)

def linear():
    # TODO: x_data值 为什么不能过大
    x_data = np.random.rand(100, 3).astype(np.float32)
    y_data = x_data * true_weight + true_bias
    # y_data += np.random.rand(100, 3) * a5

    # y = x_data * np.random.rand(100, 3)
    W = tf.Variable(tf.random_uniform([3], -0.1, 1.0))
    b = tf.Variable(tf.zeros([1]))

    y = x_data * W + b

    loss_func = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss_func)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for step in range(2000):
        sess.run(train)
        if step % 200 == 0:
            print step, sess.run(W), sess.run(b)


def sgd():
    batch_size = 10
    length = 100
    batch_num = length / batch_size
    W = tf.Variable(tf.random_uniform([3], -0.1, 1.0))
    b = tf.Variable(tf.zeros([1]))
    x_data = np.random.rand(length, 3).astype(np.float32)
    def generate_data():
        print 'start generating...'
        y_data = x_data * true_weight + true_bias
        y = x_data * W + b

        for i in range(batch_num):
            yield y_data[i:i+batch_size], y

    y_data = tf.placeholder(tf.float32, [None, 3])
    y = tf.placeholder(tf.float32, [None, 3])

    loss_func = tf.reduce_mean(tf.square(y_data - y))
    optimizer = tf.train.GradientDescentOptimizer(0.2)
    train = optimizer.minimize(loss_func)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(200):
        for data in generate_data():
            y1, y2 = data
            print y1, y2
            sess.run(train, feed_dict={y_data:y1, y:y2})
        if epoch % 50 == 0:
            print epoch, sess.run(W), sess.run(b)


sgd()

