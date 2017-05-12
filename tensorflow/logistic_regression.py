# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np

data_size = 1500

true_w = tf.constant([[1.2,2.5,3.3]])   # 1 * 3
true_b = tf.constant([10.2])

'''
    tips:
    *变量不能随便用，训练的时候会通过改变变量来达到目的。
'''
# x = tf.Variable(tf.random_normal([100,3], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None),dtype=tf.float32)
x = np.random.rand(data_size, 3).astype(np.float32)
x = tf.constant(x, dtype=tf.float32)

true_w = tf.transpose(true_w)   # Turn to 3 * 1

y = tf.nn.xw_plus_b(x, true_w, true_b)
noise = tf.constant(np.random.rand(data_size, 1).astype(np.float32))
y += noise  # add some noise.

w = tf.Variable(tf.random_normal([3,1], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
                     , dtype=tf.float32)
b = tf.Variable(tf.zeros([1], dtype=tf.float32))

y_hat = tf.nn.xw_plus_b(x, w, b)

loss = tf.reduce_mean(tf.square(y_hat - y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print sess.run(w), sess.run(b)
    print '-' * 200

    for i in range(10000):
        sess.run(train)
        if i % 200 == 0:
            print sess.run(w), sess.run(b)