# -*- coding: utf-8 -*
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
mnist = input_data.read_data_sets('./resource/MNIST_data', one_hot=True)


epoch = 10000
batch_size = 64
layer_id = 0
NUM_GPUS = 1


def init_Weights(shape):
    #  tf.random_normal() is not recommended.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def init_Biases(shape):
    # constant is better.
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_layer(x, W, b, activation_func=tf.nn.relu):
    global layer_id
    layer_id += 1
    with tf.variable_scope("layer%d"%layer_id):
        return activation_func(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)


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


class Net:
    def __init__(self):
        W_conv1 = init_Weights([5,5,1,32])
        b_conv1 = init_Biases([32])
        self.image = tf.placeholder(shape=[None, 784], dtype=tf.float32) / 255
        self.label = tf.placeholder(shape=[None, 10], dtype=tf.float32)

        images = tf.reshape(self.image, [-1,28,28,1])
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
        fc1 = dropout(fc1)

        W_fc2 = init_Weights([1024, 10])
        b_fc2 = init_Biases([10])
        prediction = fc_layer(fc1,W_fc2,b_fc2,tf.nn.softmax)

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=prediction))

        # a high learning rate could not learn anything.
        self.optimizer = tf.train.AdamOptimizer(1e-4)
        gradient = tf.gradients(self.cross_entropy, tf.trainable_variables())

        self.train = self.optimizer.apply_gradients(zip(gradient,tf.trainable_variables()))

        correct_prediction = tf.equal(tf.argmax(self.label,1), tf.argmax(prediction,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(0, grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def run(self):
        for i in range(epoch):
            batch_xs, batch_ys = mnist.train.next_batch(10)
            self.sess.run(self.train, feed_dict={self.image: batch_xs,
                                       self.label: batch_ys})
            if i % 50 == 0:
                print(self.sess.run(self.accuracy, feed_dict={self.image:
                                                                  mnist.test.images, self.label: mnist.test.labels}))


if __name__ == '__main__':
    net = Net()
    net.run()




# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     start = time.time()
#     for i in range(epoch):
#         tower_grads = []
#         for i in xrange(NUM_GPUS):
#             with tf.device('/gpu:%d' % i):
#                 with tf.name_scope('%s_%d' % ("tower", i)) as scope:
#                     batch_xs, batch_ys = mnist.train.next_batch(10)
#                     gradient = tf.gradients(cross_entropy, tf.trainable_variables())
#                     tower_grads.append(gradient)
#         grads = average_gradients(tower_grads)
#         apply_gradient_op = optimizer.apply_gradients(grads)
#
#
#         sess.run(apply_gradient_op, feed_dict={image:batch_xs,
#                                    label:batch_ys})




# if i % 50 == 0:
#     print(sess.run(accuracy, feed_dict={image: mnist.test.images,
#                                         label: mnist.test.labels}), ' --> ', time.time() - start)