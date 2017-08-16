# -*- coding: utf-8 -*
import tensorflow as tf

'''
    tf.Variable: 如果name重复，会自动在原name后天添加，即命名一个新的，而tf.get_variable则不会，所以可能会报错
    tf.name_scope 面对get_variable不起作用，对tf.Variable来说会在前面加name_scope的前缀
    tf.variable_scope 对get_variable和Variable都会加name_scope的前缀
'''
def name_scope():
    with tf.name_scope("a_name_scope"):
        initializer = tf.constant_initializer(value=1)
        var1 = tf.get_variable(name='var1', shape=[1], dtype=tf.float32, initializer=initializer)
        var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
        var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)
        var22 = tf.Variable(name='var2', initial_value=[2.2], dtype=tf.float32)


    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(var1.name)        # var1:0
        print(sess.run(var1))   # [ 1.]
        print(var2.name)        # a_name_scope/var2:0
        print(sess.run(var2))   # [ 2.]
        print(var21.name)       # a_name_scope/var2_1:0
        print(sess.run(var21))  # [ 2.0999999]
        print(var22.name)       # a_name_scope/var2_2:0
        print(sess.run(var22))  # [ 2.20000005]


def variable_scope():
    with tf.variable_scope("a_variable_scope") as scope:
        initializer = tf.constant_initializer(value=3)
        var3 = tf.get_variable(name='var3', shape=[1], dtype=tf.float32, initializer=initializer)
        scope.reuse_variables()
        var3_reuse = tf.get_variable(name='var3', )
        var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
        var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)

    with tf.variable_scope("a_variable_scop") as scope:
        var5 = tf.Variable(name='var4', initial_value=[100], dtype=tf.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(var3.name)  # a_variable_scope/var3:0
        print(sess.run(var3))  # [ 3.]
        print(var3_reuse.name)  # a_variable_scope/var3:0
        print(sess.run(var3_reuse))  # [ 3.]
        print(var4.name)  # a_variable_scope/var4:0
        print(sess.run(var4))  # [ 4.]
        print(var4_reuse.name)  # a_variable_scope/var4_1:0
        print(sess.run(var4_reuse))  # [ 4.]

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='a_variable_scope')
        for v in vars:
            print sess.run(v)


name_scope()