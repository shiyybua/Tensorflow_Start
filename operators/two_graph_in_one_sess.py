# -*- coding: utf-8 -*
import tensorflow as tf

class Net:
    def __init__(self,net_name,sess):
        self.net_name = net_name
        self.sess = sess
        self._build()
        init = tf.global_variables_initializer()
        sess.run(init)

    def _build(self):
        # 这里写name_scope不可以，不能解决variable名字重复问题，它不会在前面加名字前缀。但是可以不固定变量名。
        with tf.name_scope("part1"):
            initial = tf.truncated_normal([1, 10], stddev=0.1)
            self.w = tf.Variable(initial, collections=[self.net_name, tf.GraphKeys.GLOBAL_VARIABLES])
            # tf.Variable(self.net_name+'_w1', [1, 10], initializer=tf.random_normal_initializer(0., 0.3), collections=
            #     [self.net_name, tf.GraphKeys.GLOBAL_VARIABLES])
            self.b = tf.get_variable(self.net_name+'_b1', [1, 10], initializer=tf.constant_initializer(0.1), collections=
                [self.net_name,tf.GraphKeys.GLOBAL_VARIABLES])

    def assign(self):
        translate = [tf.assign(e2, e1) for e1, e2 in zip(tf.get_collection('net1'), tf.get_collection('net2'))]
        self.sess.run(translate)



with tf.Session() as sess:
    net1 = Net("net1", sess)
    net2 = Net("net2", sess)
    print sess.run(net1.w)
    print sess.run(net2.w)
    net1.assign()
    print sess.run(net1.w)



