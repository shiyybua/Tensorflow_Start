# -*- coding: utf-8 -*
import tensorflow as tf

# 随即生成shape=[3,4]的数组
labels = tf.Variable(tf.random_normal([3,4], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
                     , dtype=tf.float32)

# -1表示任意行
re_labels = tf.reshape(labels,[-1,1])

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print sess.run(labels)
    print sess.run(re_labels)

