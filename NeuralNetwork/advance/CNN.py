# -*- coding: utf-8 -*
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import sys
sys.path.append('/data/caiww/code/')
import tensorflow as tf
import utils
import cifar10_input
FLAGS = utils.FLAGS


# mnist.train.next_batch(1) 是一个tuple， 第一维是数据，第二维是label
# print mnist.train.next_batch(1)[0].shape

epoch = 30000
batch_size = 128
layer_id = 0
data_dir = '../../resource/cifar10/cifar-10-batches-bin'
data_dir = '/data/caiww/resource/cifar10/cifar-10-batches-bin'


def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def init_Weights(shape):
    #  tf.random_normal() is not recommended.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def init_Biases(shape):
    # constant is better.
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_layer(x, W, b, activation_func=tf.nn.relu, is_training=True):
    global layer_id
    layer_id += 1
    with tf.variable_scope("layer%d"%layer_id):
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b
        deep = W.get_shape()[3]
        is_training = tf.convert_to_tensor(is_training)
        bn_conv = batch_norm(conv, deep, is_training)
        return tf.nn.relu(bn_conv)


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def dropout(x, keep_prob=0.5):
    return tf.nn.dropout(x,keep_prob=keep_prob)


def fc_layer(x,W,b,activation_func=tf.nn.relu):
    global layer_id
    layer_id += 1
    with tf.variable_scope("layer%d" % layer_id):
        return activation_func(tf.nn.xw_plus_b(x,W,b))
        # return activation_func(tf.matmul(x,W) + b)


image = tf.placeholder(shape=[None, 24, 24, 3], dtype=tf.float32)
label_holder = tf.placeholder(shape=[None], dtype=tf.int32)
training = tf.placeholder(dtype=tf.bool)

label = tf.one_hot(label_holder,10)

W_conv1 = init_Weights([5,5,3,32])
b_conv1 = init_Biases([32])
conv1 = conv_layer(image, W_conv1, b_conv1,training)
pooling1 = max_pool_2x2(conv1)  # to 12 * 12

W_conv2 = init_Weights([5,5,32,64])
b_conv2 = init_Biases([64])
conv2 = conv_layer(pooling1, W_conv2, b_conv2,training)
pooling2 = max_pool_2x2(conv2)  # 6 * 6


W_fc1 = init_Weights([6*6*64, 1024])
b_fc1 = init_Biases([1024])
pooling1_flat = tf.reshape(pooling2,[-1, 6*6*64])
fc1 = fc_layer(pooling1_flat,W_fc1,b_fc1)
fc1 = dropout(fc1)

W_fc2 = init_Weights([1024, 10])
b_fc2 = init_Biases([10])
prediction = fc_layer(fc1,W_fc2,b_fc2,tf.nn.softmax)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=prediction))

# a high learning rate could not learn anything.
optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(label,1), tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


batch_xs_tensor, batch_ys_tensor = cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    tf.train.start_queue_runners()
    for i in range(epoch):
        batch_xs, batch_ys = sess.run([batch_xs_tensor, batch_ys_tensor])
        sess.run(train, feed_dict={image:batch_xs, label_holder:batch_ys, training: True})
        if i % 50 == 0:
            images = []
            acc = 0
            for _ in range(50):
                batch_xs, batch_ys = sess.run([batch_xs_tensor, batch_ys_tensor])
                acc += sess.run(accuracy, feed_dict={image: batch_xs, label_holder: batch_ys, training:False})
            print acc / 50.0