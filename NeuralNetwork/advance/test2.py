import tensorflow as tf

training = tf.placeholder(dtype=bool, shape=[1])

with tf.Session() as sess:
    print sess.run(training, feed_dict={training: [True]})