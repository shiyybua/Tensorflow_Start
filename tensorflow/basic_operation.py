# -*- coding: utf-8 -*

import tensorflow as tf


def matrix_mul():
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])

    # 矩阵相乘 1 * 2 and 2 * 1
    product = tf.matmul(matrix1, matrix2)

    sess = tf.Session()

    result = sess.run(product)
    # 返回 1 * 1的矩阵，还是2维。
    print result

    sess.close()


def substract():
    a = tf.Variable([3, 1])
    b = tf.constant([1, 2])

    init = tf.initialize_all_variables()

    # 把a, b换成浮点数时（其中至少包括一个Variable），直接相减结果错误。
    sub = tf.sub(a, b)

    with tf.Session() as sess:
        sess.run(init)
        print sess.run(sub)
        # 可以直接取出2个变量
        print sess.run([sub, b])


def feed():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.mul(input1, input2)

    # 类型设置很重要，不然的话就会溢出！！！
    mulinput = tf.placeholder(tf.int16, shape=(1, 2))
    # c = tf.constant([1.0, 2.0])
    c = tf.placeholder(tf.int16, shape=(1, 2))
    r = tf.add(mulinput, c)

    # init = tf.initialize_all_variables()
    with tf.Session() as sess:
        # sess.run(init)
        # 参数以字典的形式传进去placeholder。
        print sess.run([output], feed_dict={input1: [7.], input2: [3.]})
        print sess.run(output, feed_dict={input1: [7.], input2: [3.]})
        print sess.run(r, feed_dict={mulinput: [[100, 3]], c: [[100, 2]]})


# matrix_mul()
# substract()
feed()