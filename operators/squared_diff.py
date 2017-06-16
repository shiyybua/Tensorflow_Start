# -*- coding: utf-8 -*
import tensorflow as tf

x = tf.Variable([1,3,2], dtype=tf.float32)
y = tf.Variable([1,3,1], dtype=tf.float32)
z = tf.squared_difference(
    x,
    y,
    name=None
)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 维度不变
    print sess.run(z)