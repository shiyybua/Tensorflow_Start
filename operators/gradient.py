# right
import tensorflow as tf

# w1 = tf.get_variable('w1', shape=[3])
# w2 = tf.get_variable('w2', shape=[3])
#
# w3 = tf.get_variable('w3', shape=[3])
# w4 = tf.get_variable('w4', shape=[3])

w1 = tf.Variable([2.0])
w2 = tf.Variable([2.0])
w3 = tf.Variable([2.0])
w4 = tf.Variable([2.0])

z1 = w1 + w2 + w3
z2 = w3 + w4

l = [tf.convert_to_tensor([10.]), tf.convert_to_tensor([20.])]
# Do not calculate any thing.
grads = tf.gradients([z1, z2], [w1, w2, w3, w4], grad_ys=l)

# grads = tf.gradients([z1, z2], [w1, w2, w3, w4], grad_ys=[tf.convert_to_tensor([2.,2.,3.]),
#                                                           tf.convert_to_tensor([3.,2.,4.])])
opt = tf.train.AdamOptimizer(0.05)
train = opt.apply_gradients(zip(grads, [w1,w2,w3,w4]))

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(10):
        # print(sess.run(grads))
        print 'w1', sess.run(w1)
        print 'z1', sess.run(z1)
        print 'grads', sess.run(grads)
        # print 'l', sess.run(l)
        # sess.run(zip(train,(w1,w2,w3,w4)))
        sess.run(train)
        print