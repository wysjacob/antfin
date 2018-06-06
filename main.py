# /usr/bin/env python
# -*- coding:utf-8 -*-

import pickle
import sys
import numpy as np
import datetime, time
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import train_test_split
import jieba
from vocab import Vocab
from max_bag_embedding_model import create_model

EMBEDDING_PATH = 'data/sgns.merge.word'
MODEL_WEIGHTS_FILE = 'saved_models/question_pairs_weights.h5'
jieba.load_userdict('data/user_dict.txt')

'''
# for overview function
FEATURE_WORDS = set([u'花呗', u'借呗'])
KE_FU = set([u'人工', u'客服'])
# 之前用来看数据的各种分布，暂时可以忽略
def overview(index, q1, q2, label):
    both, either, neither = [], [], []
    for i in range(len(q1)):
        q1_set = set(jieba.cut(q1[i]))
        q2_set = set(jieba.cut(q2[i]))
        intersection = q1_set & q2_set
        union = q1_set | q2_set
        if intersection & KE_FU:
            both.append((i,label[i]))
        elif union & KE_FU:
            either.append((i, label[i]))
        else:
            neither.append((i, label[i]))
    print(len(both), len(either), len(neither))
    # feature_words:(37815, 1351, 180)  KE_FU:(25, 37, 39284)
    both_pos = len([x for x in both if x[1] == 1])
    both_neg = len([x for x in both if x[1] == 0])
    eigher_pos = len([x for x in either if x[1] == 1])
    either_neg = len([x for x in either if x[1] == 0])
    print(both_pos, both_neg, eigher_pos, either_neg)

    for item in both:
        print(item[0])
        print(q1[item[0]])
        print(q2[item[0]])
        print(label[item[0]])
    print('-------------------------')
    for item in either:
        print(item[0])
        print(q1[item[0]])
        print(q2[item[0]])
        print(label[item[0]])
'''


def prepare():
    # To create vocab, and save to vocab.data.
    # All training data will used for create it as default.
    vocab = Vocab('data/data_all.csv')
    vocab.load_embedding(EMBEDDING_PATH)
    with open('vocab.data', 'wb') as fout:
        pickle.dump(vocab, fout)


def train():
    # training function
    with open('vocab.data', 'rb') as fin:
        vocab = pickle.load(fin)
    q1_data = vocab.to_sequence(vocab.q1_word)
    q2_data = vocab.to_sequence(vocab.q2_word)
    labels = np.array(vocab.label, dtype=int)
    print('Shape of question1 data tensor:', q1_data.shape)
    print('Shape of question2 data tensor:', q2_data.shape)
    print('Shape of label tensor:', labels.shape)
    x = np.stack((q1_data, q2_data), axis=1)
    y = labels
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=1317)
    q1_train = x_train[:, 0]
    q2_train = x_train[:, 1]
    q1_test = x_test[:, 0]
    q2_test = x_test[:, 1]
    model = create_model()

    print(model.summary())
    print("Starting training at", datetime.datetime.now())
    t0 = time.time()
    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_acc', save_best_only=True)]
    history = model.fit([q1_train, q2_train],
                        y_train,
                        epochs=10,
                        validation_split=0.1,
                        verbose=2,
                        batch_size=32,
                        callbacks=callbacks)
    t1 = time.time()
    print("Training ended at", datetime.datetime.now())
    print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
    max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
    print('Maximum accuracy at epoch', '{:d}'.format(idx + 1), '=', '{:.4f}'.format(max_val_acc))

    model.load_weights(MODEL_WEIGHTS_FILE)
    '''
    loss, accuracy = model.evaluate([q1_test, q2_test], y_test, verbose=0)
    print('loss = {0:.4f}, accuracy = {1:.4f}'.format(loss, accuracy))
    '''
    # check f1
    predict = model.predict([q1_test, q2_test])
    predict = map(round, predict)
    from sklearn import metrics
    print(metrics.f1_score(y_test, predict, average='weighted'))
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_test, predict))




def final_predict(inpath, outpath):
    # predict function
    model = create_model()
    model.load_weights('saved_models/question_pairs_weights.h5')
    with open('vocab.data', 'rb') as fin:
        vocab = pickle.load(fin)
    linenos, q1, q2 = [], [], []
    with open(inpath, 'r') as fin:
        for line in fin:
            lineno, sen1, sen2, _ = line.strip().split('\t')
            words1 = ' '.join([w for w in jieba.cut(sen1) if w.strip()])
            words2 = ' '.join([w for w in jieba.cut(sen2) if w.strip()])
            q1.append(words1.encode('utf-8'))
            q2.append(words2.encode('utf-8'))
            linenos.append(lineno)
    q1_predict = vocab.to_sequence(q1)
    q2_predict = vocab.to_sequence(q2)
    label_predict = model.predict([q1_predict, q2_predict])
    label_predict = list(map(str, map(int, map(round, label_predict))))

    with open(outpath, 'w') as fout:
        for i, item in enumerate(label_predict):
            fout.write(linenos[i] + '\t'+item+'\n')




if __name__ == '__main__':
    prepare()
    # train()
    # final_predict(sys.argv[1], sys.argv[2])





