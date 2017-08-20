# -*- coding: utf-8 -*
import jieba

DATA_PATH = 'ner_trn'


def sent_targets():
    sentences = []
    tags = []
    uni_tags = set()
    with open(DATA_PATH, 'r') as corpus:
        sentence = []
        tag = []
        for line in corpus.readlines():
            line = line.strip()
            if line == '':
                sentences.append(sentence)
                tags.append(tag)
                sentence = []
                tag = []
            else:
                word, t = line.split(" ")
                sentence.append(word)
                tag.append(t)
                uni_tags.add(t)

    return sentences, tags, uni_tags


def retokenizer(path):
    '''
        标记不完全跟着merge走，等merge分词好了在重新标记。
    :return: 
    '''
    sentences, tags, _ = sent_targets()
    new_sentences, new_tags = [], []
    for sentence, tag in zip(sentences, tags):
        single_sentence = []
        single_tag = []
        temp_phrase = ''
        last_tag = ''
        last_type = ''
        for index, (w, t) in enumerate(zip(sentence, tag)):
            if t == 'O':
                BI, type = 'O', 'O'
            else:
                BI, type = t.split('-')
            # 如果begin-in的tag不一样了，说明要换词了，则把先前的存的词跟append进去。当然如果是首字符则不用操作。
            if BI != last_tag and temp_phrase != '' and BI != 'I':
                single_sentence.append(temp_phrase)
                single_tag.append(last_type)
                temp_phrase = ''

            temp_phrase += w
            last_tag = BI
            last_type = type

            if index == len(sentence) - 1:
                single_sentence.append(temp_phrase)
                single_tag.append(last_type)

        new_sentences.append(single_sentence)
        new_tags.append(single_tag)

    words_writer = []
    tag_writer = []
    for sentence, tag in zip(new_sentences, new_tags):
        words = []
        tags = []
        for phrase, t in zip(sentence, tag):
            # 第一个词是B后面都是I, O除外
            for index, w in enumerate(jieba.cut(phrase)):
                w = w.encode('utf8')
                if t != 'O':
                    if index == 0:
                        tags.append('B-'+t)
                    else:
                        tags.append('I-'+t)
                else:
                    tags.append(t)
                words.append(w)

        words_writer.append(words)
        tag_writer.append(tags)


    with open(path, 'w') as tokens:
        for x, y in zip(words_writer, tag_writer):
            for a, b in zip(x,y):
                tokens.write(a + ' ' + b + '\n')
            tokens.write('\n')


if __name__ == '__main__':
    operation = 'retokenizer'
    if operation == 'stat':
        sentences, tags, uni_tags = sent_targets()
        print len(sentences), len(tags)
        print uni_tags, len(uni_tags)
    elif operation == 'retokenizer':
        retokenizer('./retokenized_corpus.txt')



