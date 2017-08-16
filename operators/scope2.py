# -*- coding: utf-8 -*
import tensorflow as tf

b = tf.get_variable('a_out_variable',shape=[2])
# 即使实在variable_scope外面，但是因为名字前缀一样，还是能调用的到。 这方面tf.get_variable和tf.Variable没差别。
e = tf.Variable([1,2,3], name='scope_in/scope_in2/a_out_variable')
def scope_out():
    with tf.variable_scope("scope_in") as scope:
        with tf.variable_scope("scope_in2") as scope:
            a = tf.get_variable("name", shape=[10])
            # 虽然不附值给任何变量，在外部还是可以取到的。
            tf.get_variable("name_second", shape=[10])
            # 外面取的名字在内部是不会把前缀给加上去的。
            c = b

    # 从最高一级开始找起的
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scope_in')

    return vars,a, c

vars,a, c = scope_out()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print a.name
    print c.name
    var = sess.run(vars)
    for v in var:
        print v

    print '-' * 100

    var = sess.run(tf.get_collection('scope_in'))
    for v in var:
        print v
