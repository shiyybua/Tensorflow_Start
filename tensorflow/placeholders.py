# -*- coding: utf-8 -*
import tensorflow as tf

def ph1():
    # shape是几维, feed进来的数据也要是几维的。
    x = tf.placeholder(dtype=tf.int32, shape=[None,])
    y = tf.Variable(3)
    z = x + y

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print sess.run(z, feed_dict={x:[1]})

def ph2():
    # 把placeholder放在数组里的时候，feed也一个个的feed就行了。
    x = [tf.placeholder(dtype=tf.int32, shape=[None,]) for _ in range(3)]
    y = tf.Variable([1])
    z = x + y

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print sess.run(z, feed_dict={x[0]:[1], x[1]:[1], x[2]:[1]})

ph2()