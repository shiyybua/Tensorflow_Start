# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np

true_weight = [0.2, 0.5, 0.8]
true_bias = 30

def sgd_saver():
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
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss_func)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()

    for epoch in range(2001):
        for data in generate_data():
            y1, y2 = data
            sess.run(train, feed_dict={X:y1, y:y2})
        if epoch % 50 == 0:
            print epoch, sess.run(W), sess.run(b)

    save_path = saver.save(sess, "model/saver/model.ckpt")
    print "Model saved in file: ", save_path
    sess.close()

def loader():
    W = tf.Variable(tf.random_uniform([3, 1], -0.1, 1.0))
    b = tf.Variable(tf.zeros([1]))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'model/saver/model.ckpt')
        print sess.run(W), sess.run(b)

def sgd_saver():
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
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss_func)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()

    max_epoch = 2000
    for epoch in range(max_epoch):
        for data in generate_data():
            y1, y2 = data
            sess.run(train, feed_dict={X:y1, y:y2})
        if epoch % 200 == 0 or epoch == (max_epoch - 1):
            print epoch, sess.run(W), sess.run(b)
            saver.save(sess, 'model/checkpoints/points', global_step=epoch)

    sess.close()

def load_checkpoint():
    W = tf.Variable(tf.random_uniform([3, 1], -0.1, 1.0))
    b = tf.Variable(tf.zeros([1]))

    '''
        或者是：
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        path = ckpt.model_checkpoint_path:
    '''
    path = tf.train.latest_checkpoint('model/checkpoints/', latest_filename=None)
    # path = tf.train.latest_checkpoint('/home/cai/Desktop/tflearn/examples/nlp/model/', latest_filename=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, path)
        print sess.run(W), sess.run(b)

def load_checkpoint2():
    W = tf.Variable(tf.random_uniform([3, 1], -0.1, 1.0))
    b = tf.Variable(tf.zeros([1]))

    '''
        或者是：
        :
    '''
    ckpt = tf.train.get_checkpoint_state('model/checkpoints/')


    # path = tf.train.latest_checkpoint('/home/cai/Desktop/tflearn/examples/nlp/model/', latest_filename=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if ckpt != None:
            print ckpt, '*' * 100
            path = ckpt.model_checkpoint_path
            saver.restore(sess, path)
            print sess.run(W), sess.run(b)
        else:
            sess.run(tf.global_variables_initializer())
            print sess.run(W), sess.run(b)

def load_2_models():
    W = tf.Variable(tf.random_uniform([3, 1], -0.1, 1.0))
    b = tf.Variable(tf.zeros([1]))

    path = tf.train.latest_checkpoint('model/checkpoints/', latest_filename=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        with tf.Session() as sess2:
            saver.restore(sess, path)
            print sess.run(W), sess.run(b)

        # with tf.Session() as sess:
            saver.restore(sess2, 'model/saver/model.ckpt')
            print sess2.run(W), sess2.run(b)

            print sess.run(W), sess.run(b)


# 读取2个model，但是要放在2个session里面

# load_2_models()
load_checkpoint2()
# loader()

# sgd_saver()