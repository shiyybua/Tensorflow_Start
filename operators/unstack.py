# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np

arr = np.random.randint(0,10,[3,4])
arr = tf.convert_to_tensor(arr)
arr_hat = tf.transpose(arr)
# unstack 默认把这个二维数组沿着x轴“打平”
arr_flat = tf.unstack(arr_hat)

with tf.Session() as sess:
    print sess.run(arr)
    print sess.run(arr_hat)
    print sess.run(arr_flat)