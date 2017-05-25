# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
import copy
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib.legacy_seq2seq import embedding_attention_seq2seq, sequence_loss

class TFSeq2seq:
    def __init__(self, is_train=True):
        self.batch_size = 8
        self.seq_input_length = 10
        self.output_size_length = 9
        self.vocab_size = 10
        self.hidden_layer_size = 20
        self.num_layers = 8
        self.embedding_size = 128
        self.learning_rate = 0.01

        self.input_size_num = 10
        self.input_size_flat = self.seq_input_length * self.input_size_num
        self.is_train = is_train

    def build_weight(self, shape, name=None, func='truncated_normal'):
        if type(shape) is not list:
            raise TypeError('shape must be a list')
        if func == 'truncated_normal':
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32, name=name))

    def data_generator(self):
        # TODO: yeild
        self.weight = self.build_weight([self.output_size_length])

        # random_integers 生成的int 包括low、high值。
        input_flat = np.random.random_integers(0,self.vocab_size-1,self.input_size_flat)
        # reshape 不支持维度None。
        x = input_flat.reshape([self.input_size_num, self.seq_input_length])
        y = copy.deepcopy(x)
        # 测试阶段默认输出要小于等于输入长度。
        # delete操作是把大于output_size_length的部分删除了。
        return x, np.delete(y, range(self.output_size_length,self.seq_input_length), axis=1)


    def model(self):
        cell = BasicLSTMCell(self.hidden_layer_size)
        if self.is_train:
            cell = DropoutWrapper(cell,
                           input_keep_prob=1.0, output_keep_prob=0.8)

        multi_cell = MultiRNNCell([cell for _ in range(self.num_layers)])
        # 第一维代表的是batch size
        self.x = tf.placeholder(tf.int32, [None, self.seq_input_length])
        self.y = tf.placeholder(tf.int32, [None, self.output_size_length])

        outputs, state = embedding_attention_seq2seq(self.x, self.y, multi_cell, self.vocab_size, self.vocab_size, self.embedding_size)

        if not self.is_train:
            self.output = outputs
        else:
            self.loss_func = sequence_loss(outputs, self.y, self.weight)
            opt = tf.train.AdamOptimizer(self.learning_rate)
            self.opt_op = opt.minimize(self.loss_func)


    def step(self, x, y):
        # TODO: 可能在这里添加token
        feed_dict = {}
        assert x is not None and len(x[0]) == self.seq_input_length
        assert y is not None and len(y[0]) == self.output_size_length


        feed_dict[self.x] = x
        feed_dict[self.y] = y
        ops = self.opt_op

        return feed_dict, ops



#
# seq = TFSeq2seq()
# seq.data_generator()