# -*- coding: utf-8 -*
import tensorflow as tf
c_names = ['jsan', tf.GraphKeys.GLOBAL_VARIABLES]
# 讲顺序的！
w1 = tf.get_variable('w1', [1, 10], initializer=tf.random_normal_initializer(0., 0.3), collections=c_names)
b1 = tf.get_variable('b1', [1, 10], initializer=tf.constant_initializer(0.1), collections=c_names)

c_names = ['b', tf.GraphKeys.GLOBAL_VARIABLES]
# 讲顺序的！
w2 = tf.get_variable('w2', [1, 10], initializer=tf.random_normal_initializer(0., 0.3), collections=c_names)
b2 = tf.get_variable('b2', [1, 10], initializer=tf.constant_initializer(0.1), collections=c_names)


w1_b1 = tf.get_collection('jsan')
assign = tf.assign(w1, tf.random_normal([1,10]))

translate = [tf.assign(e2,e1) for e1,e2 in zip(tf.get_collection('jsan'),tf.get_collection('b'))]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print sess.run(w1)
    print '*' * 200
    print sess.run(w1_b1)
    print '*' * 200
    sess.run(assign)
    print sess.run(w1)
    print '*' * 200
    print sess.run(w1_b1)
    print '#' * 200
    print sess.run(w2)
    # 要用run才能生效
    sess.run(translate)
    print sess.run(w2)