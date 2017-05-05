# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np

arr = np.random.randint(1,10,[3,4])
print arr
arr = tf.convert_to_tensor(arr, dtype=tf.float32)

w = tf.Variable([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]], dtype=tf.float32)
b = tf.Variable([10.0,2,3], dtype=tf.float32)

# 这里xw_plus_b调用的是矩阵相乘 matmul。所以传值进来时要主要维度。
arr = tf.nn.xw_plus_b(arr, w, b)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print sess.run(arr)