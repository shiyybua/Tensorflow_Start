# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
'''
    两种softmax，结果一致。
'''
x = range(10)

def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob

print softmax(x)
x = tf.convert_to_tensor(x, dtype=tf.float32)
x_tensor = tf.nn.softmax(x)

topn = tf.nn.top_k(x_tensor,3)

with tf.Session() as sess:
    print sess.run(x_tensor)
    print sess.run(topn)
