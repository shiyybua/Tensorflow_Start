import tensorflow as tf
import random

def func():
    '''
    This way could replace placeholder in some cases.
    :return:
    '''
    x = random.randint(0, 10)
    x = tf.convert_to_tensor(x)
    return x

with tf.Session() as sess:
    for _ in range(10):
        print sess.run(func())

