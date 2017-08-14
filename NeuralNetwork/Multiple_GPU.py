# -*- coding: utf-8 -*
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
mnist = input_data.read_data_sets('../resource/MNIST_data', one_hot=True)


epoch = 10000
batch_size = 64
layer_id = 0
num_devices = 1
device = 'gpu'


class Net:
    def __init__(self, net_scope):
        with tf.name_scope(net_scope):
            W_conv1 = self.init_Weights([5,5,1,32])
            b_conv1 = self.init_Biases([32])
            self.image = tf.placeholder(shape=[None, 784], dtype=tf.float32) / 255
            self.label = tf.placeholder(shape=[None, 10], dtype=tf.float32)

            images = tf.reshape(self.image, [-1,28,28,1])
            conv1 = self.conv_layer(images, W_conv1, b_conv1)
            pooling1 = self.max_pool_2x2(conv1)  # to 14 * 14

            W_conv2 = self.init_Weights([5,5,32,64])
            b_conv2 = self.init_Biases([64])
            conv2 = self.conv_layer(pooling1, W_conv2, b_conv2)
            pooling2 = self.max_pool_2x2(conv2)  # 7 * 7


            W_fc1 = self.init_Weights([7*7*64, 1024])
            b_fc1 = self.init_Biases([1024])
            pooling1_flat = tf.reshape(pooling2,[-1, 7*7*64])
            fc1 = self.fc_layer(pooling1_flat,W_fc1,b_fc1)
            fc1 = self.dropout(fc1)

            W_fc2 = self.init_Weights([1024, 10])
            b_fc2 = self.init_Biases([10])
            prediction = self.fc_layer(fc1,W_fc2,b_fc2,tf.nn.softmax)

            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=prediction))

            # # a high learning rate could not learn anything.
            # self.optimizer = tf.train.AdamOptimizer(1e-4)
            # gradient = tf.gradients(self.cross_entropy, tf.trainable_variables())
            #
            # self.train = self.optimizer.apply_gradients(zip(gradient,tf.trainable_variables()))

            correct_prediction = tf.equal(tf.argmax(self.label,1), tf.argmax(prediction,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_Weights(self, shape):
        #  tf.random_normal() is not recommended.
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def init_Biases(self, shape):
        # constant is better.
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv_layer(self, x, W, b, activation_func=tf.nn.relu):
        global layer_id
        layer_id += 1
        with tf.variable_scope("layer%d" % layer_id):
            return activation_func(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def dropout(self, x, keep_prob=0.5):
        return tf.nn.dropout(x, keep_prob=keep_prob)

    def fc_layer(self, x, W, b, activation_func=tf.nn.relu):
        global layer_id
        layer_id += 1
        with tf.variable_scope("layer%d" % layer_id):
            return activation_func(tf.nn.xw_plus_b(x, W, b))


def average_gradients(tower_grads):
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
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def build_graph():
    # a high learning rate could not learn anything.
    # gradient = tf.gradients(self.cross_entropy, tf.trainable_variables())
    #
    # self.train = self.optimizer.apply_gradients(zip(gradient, tf.trainable_variables()))

    with tf.variable_scope(tf.get_variable_scope()):
        optimizer = tf.train.GradientDescentOptimizer(0.1)
        # optimizer = tf.train.AdamOptimizer(1e-4)
        tower_grads = []
        nets = []
        for i in xrange(num_devices):
            with tf.device('/%s:%d' % (device,i)):
                with tf.name_scope('%s_%d' % ("CNN_mnist", i)) as scope:
                    net = Net(scope)
                    nets.append(net)
                    loss = net.cross_entropy
                    tf.get_variable_scope().reuse_variables()
                    gradient = optimizer.compute_gradients(loss)
                    tower_grads.append(gradient[-8:])

        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads)
    return apply_gradient_op, nets



if __name__ == '__main__':
    apply_gradient_op, nets = build_graph()

    start = time.time()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            feed_data = {}
            for index, net in enumerate(nets):
                batch_xs, batch_ys = mnist.train.next_batch(10)
                feed_data[net.image] = batch_xs
                feed_data[net.label] = batch_ys
            sess.run(apply_gradient_op, feed_dict=feed_data)

            if i % 50 == 0:
                print(time.time()-start, sess.run(nets[0].accuracy, feed_dict={nets[0].image:mnist.test.images,
                                                            nets[0].label: mnist.test.labels}))


# sess.run(apply_gradient_op, feed_dict={nets[0].image:batch_xs,
#                                                    nets[0].label: batch_ys,
#                                                    nets[1].image:batch_xs1,
#                                                    nets[1].label: batch_ys1})