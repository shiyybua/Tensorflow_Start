# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np


def prnt(tensor):
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print sess.run(tensor)


a = tf.Variable(tf.random_uniform([3, 3], minval=0, maxval=100, dtype=tf.int32, seed=None, name=None))
prnt(a)

params = tf.constant([10, 20, 30, 40])
ids = tf.constant([0, 1, 2, 4])
with tf.Session() as sess:
    print tf.nn.embedding_lookup(params, ids).eval(session=sess)
