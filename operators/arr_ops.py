# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from numpy import random


def multiply():
    # arr = random.random(size=(10, 2, 15, 4)).astype(np.float32)
    arr = random.random(size=(2,4,3)).astype(np.float32)
    arr = tf.convert_to_tensor(arr)

    # pos, neg = tf.split(arr, 2, axis=1) # (10,15,4)
    initial = tf.truncated_normal([4,1], stddev=0.1)
    factors = tf.Variable(initial)

    mul = tf.multiply(arr, factors)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print sess.run(arr)
        print sess.run(factors)
        print sess.run(mul)


def split():
    arr = random.random(size=(2, 2, 3, 4)).astype(np.float32)
    # arr = random.random(size=(2, 15)).astype(np.float32)
    arr = tf.convert_to_tensor(arr)
    pos, neg = map(tf.squeeze, tf.split(arr, 2, axis=1))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print sess.run(arr).shape
        print '-' * 20
        print sess.run(pos).shape
        print '-' * 20
        print sess.run(neg).shape

def avg():
    arr = random.random(size=(2, 3, 4)).astype(np.float32)
    arr = tf.convert_to_tensor(arr)
    avged = tf.reduce_mean(arr, axis=1)

    with tf.Session() as sess:
        print sess.run(arr)
        print sess.run(arr).shape
        print sess.run(avged)
        print sess.run(avged).shape

def Euclidean():
    arr = random.random(size=(2, 4)).astype(np.float32)
    arr = tf.convert_to_tensor(arr)

    arr2 = random.random(size=(2, 4)).astype(np.float32)
    arr2 = tf.convert_to_tensor(arr2)

    result = tf.subtract(arr, arr2)
    result = tf.square(result)
    result = tf.reduce_sum(result, axis=1)
    result = tf.reduce_mean(result, axis=0)


    with tf.Session() as sess:
        print sess.run(arr)
        print sess.run(arr2)
        print sess.run(result)

def divide():
    a = tf.constant([1])
    b = tf.constant([2])
    c = tf.divide(a,2)
    with tf.Session() as sess:
        print sess.run(c)

def sigmoid():
    factors = tf.Variable([[0.5]]*10)
    factors = tf.sigmoid(factors)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print sess.run(factors)

sigmoid()