import tensorflow as tf

def prnt(tensor):
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print sess.run(tensor)