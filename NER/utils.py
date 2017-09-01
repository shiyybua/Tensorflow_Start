# -*- coding: utf-8 -*
import numpy as np
import jieba
import random
from collections import defaultdict

W2V_PATH = '../resource/wiki.zh.vec'
CORPUS_PATH = 'data_precess/retokenized_corpus.txt'
TEST_DATA_PATH = 'test.txt'
BUCKET = [(0,10),(10,20),(20,30),(30,40),(40,50),(50,100)]
unknown_embedding_path = 'unknown_embedding'
padding_embedding_path = 'padding_embedding'

embeddings_size = 300
unknown = np.load(open(unknown_embedding_path, 'r'))
padding = np.load(open(padding_embedding_path, 'r'))

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


def get_sentences(word_to_id_table, tag_to_id_table):
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

                # # 如果大于最大长度则截断
                # if len(sentence) > max_sequence:
                #     sentence = sentence[:max_sequence]
                #     tag = tag[:max_sequence]
                # # 否则填充padding
                # else:
                #     sentence += [word_padding_id] * (max_sequence - len(sentence))
                #     tag += [tag_padding_id] * (max_sequence - len(tag))

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


def group_by_sentences_padding(all_sentences, all_tags, bucket=BUCKET):
    '''
    :param all_sentences: 2维数组， [句子，单词]
    :param all_tags: 2维数组， [句子，单词对应的tag]
    :param bucket: e.g. [(0,10),(10,20),(20,30),(30,40),(40,50),(50,100)]
    :return: 字典{bucket_id：[sentences, tags]}
            bucket_id表示句子的上限
    三维数组[bucket_id, 句子, 单词], 三维数组[bucket_id, 句子, 单词对应的tag]
    '''
    assert len(all_sentences) == len(all_tags)
    group = defaultdict(list)
    for sentence, tag in zip(all_sentences, all_tags):
        sentence_len = len(sentence)
        tag_len = len(tag)
        assert sentence_len == tag_len
        if sentence_len >= BUCKET[-1][1]:
            # padding
            sentence = sentence[:BUCKET[-1][1]]
            tag = tag[:BUCKET[-1][1]]
            group[BUCKET[-1][1]].append((sentence, tag))
            continue
        for idx, (low, high) in enumerate(bucket):
            if low <= sentence_len < high:
                sentence += [word_padding_id] * (high - len(sentence))
                tag += [tag_padding_id] * (high - len(tag))
                group[high].append((sentence, tag))
                break

    return group


def get_batches(group, id_to_word_table, embeddings, batch_size):
    '''

    :param group: 函数group_by_sentences_padding返回的结果
    :param id_to_word_table:
    :param embeddings:
    :param batch_size:
    :param sample_group_id: 即是group id 也是句子的上限
    :return:
    '''
    sample_group_id = random.randint(0, len(group))
    all_sentences, all_tags = [], []
    for sentence, tag in group[sample_group_id]:
        all_sentences.append(sentence)
        all_tags.append(tag)
    all_sentences = np.array(all_sentences)
    all_tags = np.array(all_tags)
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

    return np.array(x_batch), np.array(y_batch), np.array(sentence_length), np.array(x_batch_ids), sample_group_id


def display_predict(sequence, viterbi_sequence, id_to_word_table, id_to_tag_table):
    '''
    :param sequence
    :return:
    '''
    print '=================prediction===================='
    for s, t in zip(sequence, viterbi_sequence):
        print id_to_word_table[s], '('+id_to_tag_table[t]+') ',
    print
    print ''.join([id_to_word_table[s] for s in sequence])


def tokenizer(sentence):
    return [word.encode('utf8') for word in jieba.cut(sentence)]


def get_data_from_files(embeddings, max_sequence=100):
    '''
    从文件中读取待测试的句子，把它们转换成词向量。
    :param embeddings: 预训练好的词向量字典。
    :return: padded data
    '''

    with open(TEST_DATA_PATH, 'r') as data:
        for line in data.readlines():
            actual_length = 0
            word_embedding = []
            line = line.strip()
            words = tokenizer(line)

            for w in words:
                emb = embeddings.get(w)
                if emb is None:
                    word_embedding.append(unknown)
                else:
                    word_embedding.append(emb)
                actual_length += 1

            # 如果大于最大长度则截断
            if len(word_embedding) > max_sequence:
                word_embedding = word_embedding[:max_sequence]
            # 否则填充padding
            else:
                word_embedding += [padding] * (max_sequence - len(word_embedding))
            yield np.array([word_embedding]), actual_length, words

if __name__ == '__main__':
    # embeddings = load_word2vec_embedding()
    print 'doing...'
    word_to_id_table, id_to_word_table, tag_to_id_table, id_to_tag_table = build_word_tag_tables()
    all_sentences, all_tags = get_sentences(word_to_id_table, tag_to_id_table)
    group = group_by_sentences_padding(all_sentences, all_tags)
    # x_batch, y_batch,sequence_lengths,_ = get_batches(group, id_to_word_table, embeddings, 64)
    # a = [k for k, v in group.items()]
    # print a
    # print sequence_lengths
    # print x_batch.shape
    # print y_batch.shape



