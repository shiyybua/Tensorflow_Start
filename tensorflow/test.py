import tensorflow as tf

x = tf.Variable([[2.3,1],[10,1]], dtype=tf.float32)
b = tf.constant([0,1.2],dtype=tf.float32)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print sess.run(x+b)
