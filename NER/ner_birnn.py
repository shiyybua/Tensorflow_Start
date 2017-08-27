# -*- coding: utf-8 -*
import sys
sys.path.append('/home/hhly/cai')
import tensorflow as tf
import numpy as np
import random
from tensorflow.contrib.rnn import static_bidirectional_rnn
from tensorflow.contrib.rnn import DropoutWrapper
import utils


DATA_PATH = '../retokenized_corpus.txt'
# FEATURE_NUM = 64
BATCH_SIZE = 128
EMBEDDING_SIZE = unit_num = 300         # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
MAX_SEQUENCE_SIZE = time_step = 100      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
DROPOUT_RATE = 0.5
EPOCH = 20000

embeddings = utils.load_word2vec_embedding()
word_to_id_table, id_to_word_table, tag_to_id_table, id_to_tag_table = utils.build_word_tag_tables()
all_sentences, all_tags = \
    utils.get_sentences(word_to_id_table, tag_to_id_table, max_sequence=MAX_SEQUENCE_SIZE)

TAGS_NUM = len(tag_to_id_table)


class NER_net:
    def __init__(self, scope_name, batch_size):
        self.batch_size = batch_size
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

        # -1 to BATCH_SIZE
        output = tf.reshape(projection, [time_step, self.batch_size, TAGS_NUM])
        self.outputs = tf.transpose(output, [1,0,2]) #BATCH_SIZE * time_step * TAGS_NUM

        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.outputs, self.y, np.array(self.batch_size * [time_step]))

        # Add a training op to tune the parameters.
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


def train(sess, net):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model/checkpoints/')
    if ckpt != None:
        path = ckpt.model_checkpoint_path
        saver.restore(sess, path)
    for i in range(EPOCH):
        batch_x, batch_y, sequence_lengths, batch_x_ids = \
            utils.get_batches(all_sentences, all_tags, id_to_word_table, embeddings, BATCH_SIZE)
        tf_unary_scores, tf_transition_params, _, losses = sess.run(
            [net.outputs, net.transition_params, net.train_op, net.loss],
            feed_dict={net.x: batch_x, net.y: batch_y})

        if i % 20 == 0:
            print "loss:", losses
            correct_labels = 0
            total_labels = 0
            for index, (tf_unary_scores_, y_, sequence_length_) in enumerate(zip(tf_unary_scores, batch_y,
                                                                                 sequence_lengths)):

                # # Remove padding from the scores and tag sequence.
                tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
                y_ = y_[:sequence_length_]

                # Compute the highest scoring sequence.
                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                    tf_unary_scores_, tf_transition_params)

                if index == 0:
                    utils.display_predict(batch_x_ids[index][:sequence_length_], viterbi_sequence, id_to_word_table,
                                          id_to_tag_table)

                # Evaluate word-level accuracy.
                correct_labels += np.sum(np.equal(viterbi_sequence, y_))
                total_labels += sequence_length_
            accuracy = 100.0 * correct_labels / float(total_labels)
            print("Accuracy: %.2f%%" % accuracy)

        if i % (EPOCH / 10) == 0 and i != 0:
            saver.save(sess, 'model/checkpoints/points', global_step=i)


def test(sess, net):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model/checkpoints/')
    if ckpt is None:
        print 'Please train your model first'
        return
    path = ckpt.model_checkpoint_path
    saver.restore(sess, path)

    for data in utils.get_data_from_files(embeddings):
        tf_unary_scores, tf_transition_params, _, losses = sess.run(
            [net.outputs, net.transition_params], feed_dict={net.x: data})




if __name__ == '__main__':
    action = 'train'
    # the class should be defined outside of tf.Session()
    net = NER_net('clf', batch_size=BATCH_SIZE)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        if action == 'train':
            train(sess, net)
        elif action == 'test':
            test(sess, net)



