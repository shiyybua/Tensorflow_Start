# -*- coding: utf-8 -*

import tensorflow as tf


def prnt(tensor):
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print sess.run(tensor)

a = tf.Variable([1.3,2.5])
# 强制类型转换。
# a = tf.cast(a, tf.int32)
prnt(a)

b = tf.linspace(10.0, 12.0, 2, name="linspace")
prnt(b)

c = tf.equal(a,b)
print tf.Print(a,[a])