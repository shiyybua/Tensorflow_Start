# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np
import random



expected_weight = 0.1
expected_bias = 0.3
# 一般数值类型用numpy里的
x_data = np.random.rand(100).astype(np.float32)

# x_data = [random.uniform(99, 100) for x in range(100)]
# x_data = np.array(x_data)

y_data = x_data * expected_weight + expected_bias

# 返回一个包含1个元素的一维数组，元素是从0到1 的浮点数
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = x_data * W + b

# 设置、最小化损失函数. 这里是y 和 y_data都是向量，数组
loss = tf.reduce_mean(tf.square(y - y_data))
# 用梯度下降。learning rate 是0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 目标是最小化损失函数
train = optimizer.minimize(loss)

# 初始化所有的变量
init = tf.initialize_all_variables()

sess = tf.Session()

# 每一次run 就是做一次操作。操作内容则是相应的参数。
sess.run(init)

for step in range(200):
    sess.run(train)
    if step % 20 == 0:
        # 参数不同，run所产生的结果是不一样的。比如以下例子，更多的是做一个显示功能而已。而sess.run(train)则是在最小化 损失函数。
        # 理想情况下，最后的结果应该是 W == expected_weight， b = expected_bias
        print step, sess.run(W), sess.run(b)







































