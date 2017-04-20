# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np

# np.identity 生成法向矢量
embedding = np.identity(5, dtype=np.int32)
ids = tf.placeholder(tf.int32, [None])
result = tf.nn.embedding_lookup(embedding, ids)

# RNN中 embedding 是为了把每个词打散，方便放入cell？
with tf.Session() as sess:
    # 可以看出如果feed_dict={ids:[1,2,1]}，则取了embedding的第2行、第3行和第2行，
    # embedding_lookup里面有不同的partition的方法，都大同小异。
    # feed_dict 还可以是多维度的。
    print sess.run(result, feed_dict={ids:[1,2,1]})
