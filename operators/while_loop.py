import tensorflow as tf

a = tf.Variable([1,2,3],[2,3,4])
b = tf.constant([3,9])

f = tf.constant(6)

def cond(a,b=0,f=0):
    return tf.reduce_sum(a,axis=0) < 10

def body(a,b,f):
    a = a * 2
    return a, b, f


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    res = sess.run(a)
    print res

    res = sess.run(cond(a))
    print res

    a, b, f = tf.while_loop(cond, body, [a, b, f])
    res = sess.run([a,b,f])
    print res
