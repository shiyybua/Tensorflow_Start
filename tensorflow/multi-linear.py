# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np

true_weight = [0.2, 0.5, 0.8]
true_bias = 10

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

'''
ValueError: No gradients provided for any variable, check your graph for ops that do not support gradients,
between variables ['Tensor("Variable/read:0", shape=(3, 1), dtype=float32)', 'Tensor("Variable_1/read:0", shape=(1,),
dtype=float32)'] and loss Tensor("Mean:0", shape=(), dtype=float32).

这个错误通常是 tf.Variable的错误。在建立loss function时把错误的Variable传进去了。
'''
def sgd():
    batch_size = 10
    length = 100
    batch_num = length / batch_size
    # name 这里不影响什么。
    X = tf.placeholder(tf.float32, [None, 3], name='X1')
    y = tf.placeholder(tf.float32, [None, 1], name='y1')

    W = tf.Variable(tf.random_uniform([3,1], -0.1, 1.0))
    b = tf.Variable(tf.zeros([1]))

    # 以下2个乘积差别很大，程序居然不报错... 入坑
    # y_hat = tf.multiply(X, W) + b
    y_hat = tf.matmul(X, W) + b

    x_data = np.random.rand(length, 3).astype(np.float32)

    def generate_data():
        y_data = x_data.dot(true_weight).reshape((-1, 1)) + true_bias
        for i in range(batch_num):
            yield x_data[i:i+batch_size], y_data[i:i+batch_size]


    loss_func = tf.reduce_mean(tf.square(y_hat - y))
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss_func)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for epoch in range(2001):
        for data in generate_data():
            y1, y2 = data
            sess.run(train, feed_dict={X:y1, y:y2})
        if epoch % 50 == 0:
            print epoch, sess.run(W), sess.run(b)

def sgd2():
    # Define dimensions
    d = 10  # Size of the parameter space
    N = 1000  # Number of data sample

    # create random data
    w = .5 * np.ones(d)
    x_data = np.random.random((N, d)).astype(np.float32)
    y_data = x_data.dot(w).reshape((-1, 1))

    # Define placeholders to feed mini_batches
    X = tf.placeholder(tf.float32, shape=[None, d], name='X')
    y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y')

    # Find values for W that compute y_data = <x, W>
    W = tf.Variable(tf.random_uniform([d, 1], -1.0, 1.0))
    y = tf.matmul(X, W, name='y_pred')

    # Minimize the mean squared errors.
    loss = tf.reduce_mean(tf.square(y_ - y))
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # Before starting, initialize the variables
    init = tf.initialize_all_variables()

    # Launch the graph.
    sess = tf.Session()
    sess.run(init)

    # Fit the line.
    mini_batch_size = 100
    n_batch = N // mini_batch_size + (N % mini_batch_size != 0)
    for step in range(2001):
        i_batch = (step % n_batch) * mini_batch_size
        batch = x_data[i_batch:i_batch + mini_batch_size], y_data[i_batch:i_batch + mini_batch_size]
        sess.run(train, feed_dict={X: batch[0], y_: batch[1]})
        if step % 200 == 0:
            print(step, sess.run(W))


sgd()


