# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np

data = np.random.randn(3,4)

print data
# split 沿着维度需要被整除
# tf.split return出来的结果都是一维的，变化的是内部的结构。

# 沿着第二维分割（列）
splited_d1 = tf.split(data, 4, axis=1)
# 沿着第一维分割（行）
splited_d2 = tf.split(data, 3, axis=0)

with tf.Session() as sess:
    print sess.run(splited_d1)
    print sess.run(splited_d2)