# -*- coding: utf-8 -*
import sys
sys.path.append('/home/hhly/cai')
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import static_bidirectional_rnn
from tensorflow.contrib.rnn import DropoutWrapper
import utils


DATA_PATH = '../retokenized_corpus.txt'
# FEATURE_NUM = 64
BATCH_SIZE = 128
EMBEDDING_SIZE = unit_num = 300         # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
# MAX_SEQUENCE_SIZE = time_step = 100      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
DROPOUT_RATE = None
EPOCH = 60000

embeddings = utils.load_word2vec_embedding()
word_to_id_table, id_to_word_table, tag_to_id_table, id_to_tag_table = utils.build_word_tag_tables()
all_sentences, all_tags = \
    utils.get_sentences(word_to_id_table, tag_to_id_table)
group = utils.group_by_sentences_padding(all_sentences, all_tags)

TAGS_NUM = len(tag_to_id_table)


class NER_net:
    def __init__(self, scope_name, batch_size):
        self.batch_size = batch_size
        with tf.variable_scope(scope_name) as scope:
            self._build_net()

    def _build_net(self):
        self.time_step = tf.placeholder(tf.int32, 1)
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.time_step, unit_num])
        self.y = tf.placeholder(tf.int32, [self.batch_size, self.time_step])
        seq_x = tf.reshape(self.x, [-1, self.time_step * unit_num])
        seq_x = tf.split(seq_x, self.time_step, axis=1)

        cell_forward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        if DROPOUT_RATE is not None:
            cell_forward = DropoutWrapper(cell_forward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)
            cell_backward = DropoutWrapper(cell_backward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)

        outputs, output_state_fw, output_state_bw = \
            static_bidirectional_rnn(cell_forward, cell_backward, seq_x, dtype=tf.float32)

        rnn_features = tf.transpose(outputs, [1, 0, 2])
        rnn_features = tf.reshape(rnn_features, [-1, 2 * unit_num])

        # CNN
        # You could use more advanced kernel, which is introduced in
        # https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
        filter_deep = 2  # 推荐2 或 1   # 值越大，CNN特征占的比例就越多。 如果是2表示CNN的特征和bi-rnn的特征数量一样。
        cnn_W = tf.get_variable("cnn_w", shape=[EMBEDDING_SIZE, 3, 1, filter_deep])
        cnn_b = tf.get_variable("cnn_b", shape=[filter_deep])
        # it is better to make the units number equal to the RNN unit number
        # cnn_input : (batch_size, time_step, unit_num, 1)
        cnn_input = tf.expand_dims(self.x, axis=3)
        # conv_features : (batch_size, time_step, unit_num, 2)
        conv_features = tf.nn.conv2d(cnn_input, cnn_W, strides=[1, 1, 1, 1], padding='SAME') + cnn_b
        if DROPOUT_RATE is not None:
            conv_features = tf.nn.dropout(conv_features, keep_prob=DROPOUT_RATE)
        conv_features = tf.reshape(conv_features, [-1, unit_num * filter_deep])

        all_feature = tf.concat([rnn_features, conv_features], axis=1)

        # projection:
        W = tf.get_variable("projection_w", [(filter_deep + 2) * unit_num, TAGS_NUM])  # 这里的2是指bi-rnn，所以是个常量
        b = tf.get_variable("projection_b", [TAGS_NUM])
        projection = tf.matmul(all_feature, W) + b

        self.outputs = tf.reshape(projection, [-1, self.time_step, TAGS_NUM])
        # self.outputs = tf.transpose(output, [1,0,2]) #BATCH_SIZE * time_step * TAGS_NUM

        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.outputs, self.y, np.array(self.batch_size * [self.time_step]))

        # Add a training op to tune the parameters.
        self.loss = tf.reduce_mean(-self.log_likelihood)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


def train(sess, net):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model/checkpoints/')
    if ckpt != None:
        path = ckpt.model_checkpoint_path
        print 'loading pre-trained model from %s.....' % path
        saver.restore(sess, path)
    for i in range(EPOCH):
        batch_x, batch_y, sequence_lengths, batch_x_ids, max_length = \
            utils.get_batches(group, id_to_word_table, embeddings, BATCH_SIZE)
        tf_unary_scores, tf_transition_params, _, losses = sess.run(
            [net.outputs, net.transition_params, net.train_op, net.loss],
            feed_dict={net.x: batch_x, net.y: batch_y, net.time_step: max_length})

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
    print 'loading pre-trained model from %s.....' % path
    saver.restore(sess, path)

    for data in utils.get_data_from_files(embeddings):
        x, sequence_length_, words = data
        tf_unary_scores, tf_transition_params = sess.run(
            [net.outputs, net.transition_params], feed_dict={net.x: x})
        tf_unary_scores_ = tf_unary_scores[0][:sequence_length_]

        # Compute the highest scoring sequence.
        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
            tf_unary_scores_, tf_transition_params)

        for w, t in zip(words, viterbi_sequence):
            print w, '(' + id_to_tag_table[t] + ') ',
        print
        print '*' * 100
        # print ' '.join(words)
        # print viterbi_sequence


def test1(sess, net):
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('model/checkpoints/')
    if ckpt is None:
        print 'Please train your model first'
        return
    path = ckpt.model_checkpoint_path
    print 'loading pre-trained model from %s.....' % path
    saver.restore(sess, path)

    batch_x, batch_y, sequence_lengths, batch_x_ids, max_length = \
        utils.get_batches(group, id_to_word_table, embeddings, 10)

    for index, (x, y, sequence_length_) in enumerate(zip(batch_x, batch_y, sequence_lengths)):
        # 不需要提供y。
        tf_unary_scores, tf_transition_params = sess.run(
            [net.outputs, net.transition_params], feed_dict={net.x: [x], net.time_step: max_length})
        tf_unary_scores_ = tf_unary_scores[0][:sequence_length_]

        # Compute the highest scoring sequence.
        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
            tf_unary_scores_, tf_transition_params)

        utils.display_predict(batch_x_ids[index][:sequence_length_], viterbi_sequence, id_to_word_table,
                              id_to_tag_table)
        # print ' '.join(words)
        # print viterbi_sequence


if __name__ == '__main__':
    action = 'test'
    # the class should be defined outside of tf.Session()
    if action == 'test':
        BATCH_SIZE = 1

    net = NER_net('clf', batch_size=BATCH_SIZE)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        if action == 'train':
            train(sess, net)
        elif action == 'test':
            test1(sess, net)



