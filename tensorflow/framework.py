# -*- coding: utf-8 -*

import numpy as np
import tensorflow as tf
import common

array = np.random.rand(1,2)

# 转化成tensor
arr = tf.convert_to_tensor(array)

common.prnt(arr)

# value 必须是tensor
b = tf.add_to_collection("a",arr)
common.prnt(tf.get_collection('a'))

c = tf.constant([1,2,3])
c = tf.pack(c)
common.prnt(c)
