# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np

# 定义参数所在层的输入维度为fan_in，输出维度为fan_out
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    # return tf.random_uniform((fan_in, fan_out),
    #                          minval=low, maxval=high, dtype=tf.float32)
    return tf.random_uniform_initializer(minval=low, maxval=high, dtype=tf.float32)


w = tf.get_variable("name",shape=[1,10],dtype=tf.float32, initializer=xavier_init(10,60))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print sess.run(w)