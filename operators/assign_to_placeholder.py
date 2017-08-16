# -*- coding: utf-8 -*
import tensorflow as tf

'''
    placeholder 和 constant不可以被赋值。
'''
var1 = tf.placeholder(tf.float32, [None])
var2 = tf.get_variable("g", shape=[1])

assign = tf.assign(var2, var1)

with tf.Session() as sess:
    sess.run(assign, feed_dict={var1: [10.0]})
    print sess.run(var2)
