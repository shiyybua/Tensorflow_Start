# -*- coding: utf-8 -*
import tensorflow as tf
import numpy as np
import random
from tensorflow.contrib.rnn import static_bidirectional_rnn
from tensorflow.contrib.rnn import DropoutWrapper


DATA_PATH = '../retokenized_corpus.txt'
# FEATURE_NUM = 64
BATCH_SIZE = 128
EMBEDDING_SIZE = unit_num = 300         # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
MAX_SEQUENCE_SIZE = time_step = 30      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
DROPOUT_RATE = 0.5
word_embedding = np.random.random([1000, EMBEDDING_SIZE])
sentences = np.random.randint(0, 1000, [5000, MAX_SEQUENCE_SIZE])
tags = np.random.randint(0,9,[5000, MAX_SEQUENCE_SIZE])
TAGS_NUM = 10
sequence_lengths = np.full(BATCH_SIZE, MAX_SEQUENCE_SIZE - 1, dtype=np.int32)


class NER_net:
    def __init__(self, scope_name):
        with tf.variable_scope(scope_name) as scope:
            self._build_net()

    def _build_net(self):
        self.x = tf.placeholder(tf.float32, [None, time_step, unit_num])
        self.y = tf.placeholder(tf.int32, [None, time_step])
        seq_x = tf.reshape(self.x, [-1, time_step * unit_num])
        seq_x = tf.split(seq_x, time_step, axis=1)

        cell_forward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        cell_forward = DropoutWrapper(cell_forward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)
        cell_backward = DropoutWrapper(cell_backward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)

        outputs, output_state_fw, output_state_bw = \
            static_bidirectional_rnn(cell_forward, cell_backward, seq_x, dtype=tf.float32)

        # projection:
        W = tf.get_variable("projection_w", [2 * unit_num, TAGS_NUM])
        b = tf.get_variable("projection_b", [TAGS_NUM])
        x_reshape = tf.reshape(outputs, [-1, 2 * unit_num])
        projection = tf.matmul(x_reshape, W) + b

        output = tf.reshape(projection, [time_step, BATCH_SIZE, TAGS_NUM])
        self.outputs = tf.transpose(output, [1,0,2]) #BATCH_SIZE * time_step * TAGS_NUM

        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.outputs, self.y, sequence_lengths)

        # Add a training op to tune the parameters.
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


def get_batch():
    sample_ids = random.sample(range(len(word_embedding)), BATCH_SIZE)
    x = sentences[sample_ids]   # 64 * time_step
    y = tags[sample_ids]
    batch = []
    for sentence_ids in x:
        data_unit = []
        for id in sentence_ids:
            data_unit.append(word_embedding[id])
        batch.append(data_unit)
    return np.array(batch), np.array(y)


if __name__ == '__main__':
    # the class should be defined outside of tf.Session()
    net = NER_net('clf')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(2000):
          batch_x, batch_y = get_batch()
          tf_unary_scores, tf_transition_params, _, losses = sess.run(
              [net.outputs, net.transition_params, net.train_op, net.loss], feed_dict={net.x:batch_x, net.y:batch_y})

          print losses

          if i % 10 == 0:
            correct_labels = 0
            total_labels = 0
            for tf_unary_scores_, y_, sequence_length_ in zip(tf_unary_scores, batch_y,
                                                              sequence_lengths):

              # # Remove padding from the scores and tag sequence.
              tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
              y_ = y_[:sequence_length_]

              # Compute the highest scoring sequence.
              viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                  tf_unary_scores_, tf_transition_params)

              # Evaluate word-level accuracy.
              correct_labels += np.sum(np.equal(viterbi_sequence, y_))
              total_labels += sequence_length_
            accuracy = 100.0 * correct_labels / float(total_labels)
            print("Accuracy: %.2f%%" % accuracy)

