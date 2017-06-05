# -*- coding: utf-8 -*
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../resource/MNIST_data', one_hot=True)

# mnist.train.next_batch(1) 是一个tuple， 第一维是数据，第二维是label
# print mnist.train.next_batch(1)[0].shape

epoch = 10000
batch_size = 64
layer_id = 0


def hidden_layer(input, input_size, layer_size):
    global layer_id
    layer_id += 1
    with tf.variable_scope("layer%d"%layer_id):
        # tf.random_normal cannot take None as a parameter
        Weights = tf.Variable(tf.random_normal([input_size, layer_size], dtype=tf.float32), dtype=tf.float32)
        Biases = tf.Variable(tf.zeros([layer_size], dtype=tf.float32),dtype=tf.float32)
        # 乘出来之后每一列就是对应的一组weight值。
        # shape of tf.matmul(input, Weights): [batch_size, layer_size]
        # shape of Biases: [layer_size]
        output = tf.matmul(input, Weights) + Biases
        # 用sigmoid不行，应该是在BP更新参数的时候会非常受影响。
        output = tf.nn.sigmoid(output)

    # [batch_size, layer_size]
    return output


def ouput_layer(input, input_size, layer_size):
    global layer_id
    layer_id += 1
    with tf.variable_scope("layer%d"%layer_id):
        # tf.random_normal cannot take None as a parameter
        Weights = tf.Variable(tf.random_normal([input_size, layer_size], dtype=tf.float32), dtype=tf.float32)
        Biases = tf.Variable(tf.zeros([layer_size], dtype=tf.float32),dtype=tf.float32)
        # 乘出来之后每一列就是对应的一组weight值。
        # shape of tf.matmul(input, Weights): [batch_size, layer_size]
        # shape of Biases: [layer_size]
        output = tf.matmul(input, Weights) + Biases
        # 用sigmoid不行，应该是在BP更新参数的时候会非常受影响。
        output = tf.nn.softmax(output)

    # [batch_size, layer_size]
    return output

x = tf.placeholder(shape=[None, 784], dtype=tf.float32)
y = tf.placeholder(shape=[None, 10], dtype=tf.float32)


h1 = hidden_layer(x, 784, 350)
h2 = hidden_layer(h1, 350, 150)
prediction = ouput_layer(h2, 150, 10)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction),
                                              reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # 以下这个形式是错的，因为每次next batch 会随机产生不同的数据
        # batch_xs = mnist.train.next_batch(batch_size)[0]
        # batch_ys = mnist.train.next_batch(batch_size)[1]
        sess.run(train, feed_dict={x:batch_xs,
                                   y:batch_ys})
        if i % 50 == 0:
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))