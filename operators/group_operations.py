import tensorflow as tf

a = tf.Variable([10])
b = tf.Variable([20])
c = tf.Variable([0])

add = tf.add(a, b)
mul = tf.multiply(a, b)
assign = tf.assign(c, mul)

k = (add, mul, assign)

group = tf.group(*k)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(group)
    print sess.run(c)