# -*- coding: utf-8 -*
import numpy as np
import random

W2V_PATH = '../resource/wiki.zh.vec'
CORPUS_PATH = 'data_precess/retokenized_corpus.txt'

embeddings_size = 300
unknown = np.random.normal(size=(embeddings_size))
padding = np.random.normal(size=(embeddings_size))

word_unknown_id = 0
word_padding_id = 1
tag_padding_id = 0


def load_word2vec_embedding(path=W2V_PATH):
    '''
        加载外接的词向量。
        :param path:
        :return:
    '''
    embeddings_index = {}
    f = open(path)
    for line in f:
        values = line.split()
        word = values[0]  # 取词
        coefs = np.asarray(values[1:], dtype='float32')  # 取向量
        embeddings_index[word] = coefs  # 将词和对应的向量存到字典里
    f.close()
    return embeddings_index


def build_word_tag_tables():
    '''
        建立tag,word 对应的ID词表。
        :return:
    '''
    word_id = 2
    tag_id = 1
    word_to_id_table = {'<unkown-word>':0, '<padding-word>':1}
    tag_to_id_table = {'PADDING-TAG':0}
    id_to_word_table = {0: '<unkown-word>', 1: '<padding-word>'}
    id_to_tag_table = {0: 'PADDING-TAG'}
    with open(CORPUS_PATH, 'r') as corpus:
        while True:
            line = corpus.readline()
            if not line: break
            line = line.strip()
            if line == '': continue
            word, tag = line.split()
            if word_to_id_table.get(word) is None:
                word_to_id_table[word] = word_id
                id_to_word_table[word_id] = word
                word_id += 1
            if tag_to_id_table.get(tag) is None:
                tag_to_id_table[tag] = tag_id
                id_to_tag_table[tag_id] = tag
                tag_id += 1

    return word_to_id_table, id_to_word_table, tag_to_id_table, id_to_tag_table


def get_sentences(word_to_id_table, tag_to_id_table, max_sequence=100):
    '''

    :param word_to_id_table: 词转id
    :param tag_to_id_table: tag转id
    :param max_sequence: 一个句子最大的长度，这个值需要通过一个合理的统计得出。
    :return: sentences， tags 里面装的是对应的ID，padding_id
    '''


    sentences = []
    tags = []
    with open(CORPUS_PATH, 'r') as corpus:
        sentence = []
        tag = []
        l = []
        for line in corpus.readlines():
            line = line.strip()
            if line == '':
                l.append(len(sentence))

                # 如果大于最大长度则截断
                if len(sentence) > max_sequence:
                    sentence = sentence[:max_sequence]
                    tag = tag[:max_sequence]
                # 否则填充padding
                else:
                    sentence += [word_padding_id] * (max_sequence - len(sentence))
                    tag += [tag_padding_id] * (max_sequence - len(tag))

                sentences.append(sentence)
                tags.append(tag)
                sentence = []
                tag = []
            else:
                word, t = line.split(" ")
                sentence.append(word_to_id_table[word])
                tag.append(tag_to_id_table[t])
    # np.save(open('temp', 'w'), l)
    return np.array(sentences), np.array(tags)


def get_batches(all_sentences, all_tags, id_to_word_table, embeddings, batch_size):
    sample_ids = random.sample(range(len(all_sentences)), batch_size)
    x_batch_ids = all_sentences[sample_ids]
    y_batch = all_tags[sample_ids]
    x_batch = []
    sentence_length = []
    for sentence in x_batch_ids:
        word_embeddings = []
        length = 0
        for word_id in sentence:
            if word_id == word_padding_id:
                word_embeddings.append(padding)
            else:
                # 理论上都能找到。
                word = id_to_word_table[word_id]
                word_embeddings.append(embeddings.get(word, unknown))
                # if embeddings.get(word) is None: print word
                length += 1
        sentence_length.append(length)

        x_batch.append(word_embeddings)
    # 得到每个句子的实际长度。

    return np.array(x_batch), np.array(y_batch), np.array(sentence_length), np.array(x_batch_ids)


def display_predict(sequence, viterbi_sequence, id_to_word_table, id_to_tag_table):
    '''
    :param sequence
    :return:
    '''
    print '=================prediction===================='
    for s, t in zip(sequence, viterbi_sequence):
        print id_to_word_table[s], '('+id_to_tag_table[t]+') ',
    print



if __name__ == '__main__':
    embeddings = load_word2vec_embedding()
    word_to_id_table, id_to_word_table, tag_to_id_table, id_to_tag_table = build_word_tag_tables()
    all_sentences, all_tags = get_sentences(word_to_id_table, tag_to_id_table)
    x_batch, y_batch,sequence_lengths,_ = get_batches(all_sentences, all_tags, id_to_word_table, embeddings, 64)

    print x_batch.shape, y_batch.shape
    print sequence_lengths[:5]
    print len(tag_to_id_table)
    print x_batch[0][0]
    # for e in all_sentences[0]:
    #     print id_to_word_table[e],
    for e in y_batch[0]:
        if e != 20945:
            print id_to_tag_table[e],

    print


