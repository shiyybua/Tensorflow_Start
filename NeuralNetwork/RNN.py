# -*- coding: utf-8 -*
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import static_bidirectional_rnn
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

num_units = 28
time_step = 28
batch_size = 128
mnist = input_data.read_data_sets('../resource/MNIST_data', one_hot=True)

data = tf.placeholder(dtype=tf.float32, shape=[None, time_step * num_units])
labels = tf.placeholder(dtype=tf.int32, shape=[None, 10])

data_x = tf.split(data, time_step, axis=1)
cell_forward = tf.contrib.rnn.BasicLSTMCell(num_units)
cell_backward = tf.contrib.rnn.BasicLSTMCell(num_units)
outputs, output_state_fw, output_state_bw = static_bidirectional_rnn(cell_forward, cell_backward, data_x, dtype=tf.float32)

# projection:
W = tf.get_variable("projection_w", [2 * num_units, 10])
b = tf.get_variable("projection_b", [10])
# outputs = tf.reshape(outputs, [-1, 2 * num_units])
# 直接扔掉time_step的信息，只保留units上的信息。
pred = tf.matmul(outputs[-1], W) + b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))
optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={data: batch_xs, labels: batch_ys})


        if i % 50 == 0:
            print(sess.run(accuracy, feed_dict={data: batch_xs, labels: batch_ys}))

        # a, b, c = sess.run([outputs, output_state_fw, output_state_bw], feed_dict={data: batch_xs, labels: batch_ys})
        # a, b, c = np.array(a), np.array(b), np.array(c)
        # print a.shape, b.shape, c.shape




'''
    outputs, output_state_fw, output_state_bw:
    //这里的2确实和bi-RNN有关。
    outputs: (time_step, batch_size, 2 * num_units)

    //本来本层传递就是2个State
    output_state_fw: (2, batch_size, num_units)
    output_state_bw: (2, batch_size, num_units)

    (28, 10, 28) (2, 10, 28)

'''



