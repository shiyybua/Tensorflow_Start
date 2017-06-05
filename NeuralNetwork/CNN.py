# -*- coding: utf-8 -*
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../resource/MNIST_data', one_hot=True)

# mnist.train.next_batch(1) 是一个tuple， 第一维是数据，第二维是label
# print mnist.train.next_batch(1)[0].shape

epoch = 10000
batch_size = 64
layer_id = 0


def init_Weights(shape):
    return tf.Variable(tf.random_normal(shape,dtype=tf.float32))


def init_Biases(shape):
    return tf.Variable(tf.random_normal(shape, dtype=tf.float32))


def conv_layer(x, W, b, activation_func=tf.nn.relu):
    global layer_id
    layer_id += 1
    with tf.variable_scope("layer%d"%layer_id):
        return activation_func(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def fc_layer(x,W,b,activation_func=tf.nn.relu):
    global layer_id
    layer_id += 1
    with tf.variable_scope("layer%d" % layer_id):
        return activation_func(tf.nn.xw_plus_b(x,W,b))
        # return activation_func(tf.matmul(x,W) + b)


W_conv1 = init_Weights([5,5,1,32])
b_conv1 = init_Biases([32])
image = tf.placeholder(shape=[None, 784], dtype=tf.float32) / 255
label = tf.placeholder(shape=[None, 10], dtype=tf.float32)

images = tf.reshape(image, [-1,28,28,1])
conv1 = conv_layer(images, W_conv1, b_conv1)
pooling1 = max_pool_2x2(conv1)  # to 14 * 14

W_conv2 = init_Weights([5,5,32,64])
b_conv2 = init_Biases([64])
conv2 = conv_layer(pooling1, W_conv2, b_conv2)
pooling2 = max_pool_2x2(conv2)  # 7 * 7


W_fc1 = init_Weights([7*7*64, 1024])
b_fc1 = init_Biases([1024])
pooling1_flat = tf.reshape(pooling2,[-1, 7*7*64])
fc1 = fc_layer(pooling1_flat,W_fc1,b_fc1)

W_fc2 = init_Weights([1024, 10])
b_fc2 = init_Biases([10])
prediction = fc_layer(fc1,W_fc2,b_fc2,tf.nn.softmax)


# cross_entropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(prediction),
#                                               reduction_indices=[1]))
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=prediction))

optimizer = tf.train.AdamOptimizer(0.05)
train = optimizer.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        batch_xs, batch_ys = mnist.train.next_batch(10)
        # 以下这个形式是错的，因为每次next batch 会随机产生不同的数据
        # batch_xs = mnist.train.next_batch(batch_size)[0]
        # batch_ys = mnist.train.next_batch(batch_size)[1]
        sess.run(train, feed_dict={image:batch_xs,
                                   label:batch_ys})
        if i % 50 == 0:
            print(sess.run(accuracy, feed_dict={image: mnist.test.images, label: mnist.test.labels}))