# -*- coding: utf-8 -*
import tensorflow as tf

class Net:
    def __init__(self, name, sess):
        self.sess = sess
        c_names = [name, tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.name_scope(name+'var'):
            # 讲顺序的！
            self.w = tf.Variable(tf.truncated_normal([1,10], stddev=0.1), collections=c_names)
            self.b = tf.Variable(tf.truncated_normal([1,10], stddev=0.1), collections=c_names)


    def assign(self):
        translate = [tf.assign(e1,e2) for e1,e2 in zip(tf.get_collection('net1'),tf.get_collection('net2'))]
        self.sess.run(translate)


with tf.Session() as sess:
    net1 = Net('net1',sess)
    net2 = Net('net2',sess)
    init = tf.global_variables_initializer()
    sess.run(init)

    print 'net1,w:', sess.run(net1.w)
    print 'net2,w:', sess.run(net2.w)

    net1.assign()

    print 'net1,w:', sess.run(net1.w)
    print 'net2,w:', sess.run(net2.w)


    # print sess.run(w1)
    # print '*' * 200
    # print sess.run(w1_b1)
    # print '*' * 200
    # sess.run(assign)
    # print sess.run(w1)
    # print '*' * 200
    # print sess.run(w1_b1)
    # print '#' * 200
    # print sess.run(w2)
    # # 要用run才能生效
    # sess.run(translate)
    # print sess.run(w2)