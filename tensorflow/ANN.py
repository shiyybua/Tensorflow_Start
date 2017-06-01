# -*- coding: utf-8 -*
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./resource/MNIST_data', one_hot=True)

# mnist.train.next_batch(1) 是一个tuple， 第一维是数据，第二维是label
# print mnist.train.next_batch(1)[0].shape

epoch = 10
batch_size = 64

layer_id = 0
def hidden_layer(input, layer_size=10):
    with tf.variable_scope("layer%d"%layer_id):
        # tf.random_normal cannot take None as a parameter
        Weights = tf.Variable(tf.random_normal([784, layer_size], dtype=tf.float32), dtype=tf.float32)
        Biases = tf.Variable(tf.zeros([layer_size], dtype=tf.float32),dtype=tf.float32)
        # 乘出来之后每一列就是对应的一组weight值。
        # shape of tf.matmul(input, Weights): [batch_size, layer_size]
        # shape of Biases: [layer_size]
        output = tf.matmul(input, Weights) + Biases
        output = tf.nn.sigmoid(output)

    # [batch_size, layer_size]
    return output

x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y = tf.placeholder(shape=[None, 10], dtype=tf.float32)


prediction = hidden_layer(x)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction),
                                              reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer(0.05)
train = optimizer.minimize(cross_entropy)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(epoch):
        sess.run(train, feed_dict={x:mnist.train.next_batch(batch_size)[0],
                                   y: mnist.train.next_batch(batch_size)[1]})







