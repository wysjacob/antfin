# -*- coding:utf8 -*-

import numpy as np
import io
import pandas as pd
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from utils.langconv import *

FEATURE_WORDS = set([u'花呗', u'借呗'])
HUA_BEI = set([u'花贝', u'花吧'])
JIE_BEI = set([u'借吧', u'借呗'])
MAX_SEQUENCE_LENGTH = 15

jieba.load_userdict('data/user_dict.txt')


class Vocab(object):
    def __init__(self, file, simplified=True, correct=True):
        _, _, _, self.q1_word, self.q2_word, self.label = self.get_data(file, simplified, correct)
        self.q_word = self.q1_word + self.q2_word
        self.embedding = 0
        self.word_index = {}
        self.nb_words = 0
        self.tokenizer = None

    def get_data(self, file, simplified=True, correct=True):
        df = pd.read_csv(file, header=None, sep='\t')
        index, q1, q2, label = df[0].tolist(), df[1].tolist(), df[2].tolist(), map(float, df[3].tolist())
        if simplified:
            q1 = list(map(self.cht_to_chs, q1))
            q2 = list(map(self.cht_to_chs, q2))
        if correct:
            q1 = list(map(self.correction, q1))
            q2 = list(map(self.correction, q2))
        q1_word = map(list, map(jieba.cut, q1))
        q2_word = map(list, map(jieba.cut, q2))

        def join_(l):
            return ' '.join(l).encode("utf-8").strip()

        q1_word = map(join_, q1_word)
        q2_word = map(join_, q2_word)

        return index, q1, q2, q1_word, q2_word, label

    def cht_to_chs(self, line):
        line = Converter('zh-hans').convert(line.decode("utf-8"))
        line.encode('utf-8')
        return line

    def correction(self, q):
        for word in FEATURE_WORDS:
            if word in q:
                return q
        for word in HUA_BEI:
            q = q.replace(word, u'花呗')
        for word in JIE_BEI:
            q = q.replace(word, u'借呗')
        return q

    def load_embedding(self, path):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.q_word)
        self.word_index = self.tokenizer.word_index
        print("Words in index: %d" % len(self.word_index))
        embeddings_index = {}
        fin = io.open('data/sgns.merge.word', 'r', encoding='utf-8', newline='\n', errors='ignore')
        for i, line in enumerate(fin):
            if i == 1200000:
                break
            tokens = line.rstrip().split(' ')
            embeddings_index[tokens[0]] = list(map(float, tokens[1:]))
        self.nb_words = len(self.word_index)
        self.embedding = np.zeros([self.nb_words + 1, 300])
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word.decode('utf-8'))
            if embedding_vector is not None:
                self.embedding[i] = embedding_vector
        print('Null word embeddings: %d' % np.sum(np.sum(self.embedding, axis=1) == 0))

    def to_sequence(self, question, padding=True):
        seq = self.tokenizer.texts_to_sequences(question)
        if padding:
            seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        return seq



if __name__ == '__main__':
    vocab = Vocab('data/data_all.csv')
    vocab.load_embedding('data/sgns.merge.word')
