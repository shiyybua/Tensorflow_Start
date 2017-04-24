# -*- coding: utf-8 -*
from tf_seq2seq import TFSeq2seq
import random
import tensorflow as tf

model = TFSeq2seq()
whole_data = model.data_generator()
epoch = 10

def get_batches(whole_data):
    x_whole, y_whole = whole_data
    assert len(x_whole) == len(y_whole)
    x_batch = []
    y_batch = []
    for _ in range(model.batch_size):
        ran_num = random.randint(0, len(x_whole))
        x_batch.append(x_whole[ran_num])
        y_batch.append(y_whole[ran_num])
    return x_batch, y_batch

model.model()

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
for e in range(epoch):
    for _ in range(len(whole_data) / model.batch_size):
        x, y = get_batches(whole_data)
        feed, train_op = model.step(x, y)
        print feed
        sess.run(train_op, feed_dict=feed)

sess.close()




