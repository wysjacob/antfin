# -*- coding:utf-8 -*-

import pickle
import sys
import os
import numpy as np
import datetime, time
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from vocab import Vocab
from model import max_embedding, cnn_lstm_f1, bilstm

EMBEDDING_PATH = 'data/sgns.merge.char'
MODEL_WEIGHTS_FILE = 'saved_models/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'


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
    q1_data = vocab.to_sequence(vocab.q1_char)
    q2_data = vocab.to_sequence(vocab.q2_char)
    labels = np.array(vocab.label, dtype=int)
    print('Shape of question1 data tensor:', q1_data.shape)
    print('Shape of question2 data tensor:', q2_data.shape)
    print('Shape of label tensor:', labels.shape)
    x = np.stack((q1_data, q2_data), axis=1)
    y = labels
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1317)
    q1_train = x_train[:, 0]
    q2_train = x_train[:, 1]
    q1_test = x_test[:, 0]
    q2_test = x_test[:, 1]
    model = cnn_lstm_f1()

    print(model.summary())
    print("Starting training at", datetime.datetime.now())
    t0 = time.time()
    callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=False)]

    pos_rate = float(np.sum(labels)) / len(labels)
    neg_rate = 1 - pos_rate
    cw = {0: 1/neg_rate, 1: 1/pos_rate}

    history = model.fit([q1_train, q2_train],
                        y_train,
                        epochs=40,
                        validation_split=0.001,
                        verbose=2,
                        batch_size=40,
                        callbacks=callbacks,
                        class_weight=cw
                        )
    t1 = time.time()
    print("Training ended at", datetime.datetime.now())
    print("Minutes elapsed: %f" % ((t1 - t0) / 60.))
    min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
    print('Min loss at epoch', '{:d}'.format(idx + 1), '=', '{:.4f}'.format(min_val_loss))

    path = 'saved_models/'

    def get_f1(matrix):
        aa = float(matrix[1][1]) / (matrix[1][1] + matrix[0][1])
        bb = float(matrix[1][1]) / (matrix[1][1] + matrix[1][0])
        return 2 / (1 / aa + 1 / bb)

    for file in os.listdir(path):
        if file == '.DS_Store':
            continue
        file_path = os.path.join(path, file)

        model.load_weights(file_path)

        # check f1
        print(file_path, ':')
        predict = model.predict([q1_test, q2_test])
        predict = map(round, predict)

        matrix = confusion_matrix(y_test, predict)
        print(matrix)
        print('f1:', get_f1(matrix))


def final_predict(inpath, outpath, bagging=False):

    with open('vocab.data', 'rb') as fin:
        vocab = pickle.load(fin)
    linenos, q1, q2 = [], [], []
    with open(inpath, 'r') as fin:
        for line in fin:
            lineno, sen1, sen2 = line.strip().split('\t')
            sen1 = vocab.cht_to_chs(sen1)
            sen2 = vocab.cht_to_chs(sen2)
            sen1 = vocab.correction(sen1)
            sen2 = vocab.correction(sen2)
            chars1 = ' '.join([w for w in sen1 if w.strip()])
            chars2 = ' '.join([w for w in sen2 if w.strip()])
            q1.append(chars1.encode('utf-8'))
            q2.append(chars2.encode('utf-8'))
            linenos.append(lineno)
    q1_predict = vocab.to_sequence(q1)
    q2_predict = vocab.to_sequence(q2)



    def classify(score, threshold=0.45):
        ret = 1 if score > threshold else 0
        return str(ret)

    if not bagging:
        model = cnn_lstm_f1()
        model.load_weights('saved_models/question_pairs_weights.h5')
        label_predict = model.predict([q1_predict, q2_predict])
        label_predict = list(map(classify, label_predict))
    else:
        label_predict = list(map(str, bagging_predict(q1_predict, q2_predict, mode='vote')))

    with open(outpath, 'w') as fout:
        for i, item in enumerate(label_predict):
            fout.write(linenos[i] + '\t'+item+'\n')


def bagging_predict(q1_predict, q2_predict, mode='vote'):
    '''

    :param model: vote: 4 model4 vote predict. score: sum of 4 models score
    :return: predict results
    '''
    models = []
    model = max_embedding()
    model.load_weights('saved_models/max1.h5')
    models.append(model)

    model = max_embedding()
    model.load_weights('saved_models/max2.h5')
    models.append(model)

    model = cnn_lstm_f1()
    model.load_weights('saved_models/lstm3.h5')
    models.append(model)

    model = cnn_lstm_f1()
    model.load_weights('saved_models/lstm4.h5')
    models.append(model)

    def classify(score, threshold=0.5):
        ret = 1 if score >= threshold else 0
        return ret

    if mode == 'vote':
        vote = [0.0 for _ in range(len(q1_predict))]
        for model in models:
            predict = model.predict([q1_predict, q2_predict])
            label_predict = list(map(classify, predict))
            vote = [i+j for i,j in zip(vote, label_predict)]
        final_predict = [classify(i, 2.0) for i in vote]

    if mode == 'score':
        score = [0.0 for _ in range(len(q1_predict))]
        for model in models:
            predict = model.predict([q1_predict, q2_predict])
            score = [i+j for i,j in zip(score, predict)]
        final_predict = [classify(i, 2.0) for i in score]

    return final_predict


if __name__ == '__main__':
    prepare()
    train()
    #final_predict('fin.txt', 'fout.txt', bagging=True)
    final_predict(sys.argv[1], sys.argv[2])




