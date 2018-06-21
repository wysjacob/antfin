# -*- coding:utf8 -*-

import pickle
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization,Flatten
from keras.layers.embeddings import Embedding
from keras import backend as K


def create_model():
    dropoutrate=0.25
    from keras.layers import Merge,Conv1D,MaxPool1D
    with open('vocab.data', 'rb') as fin:
        vocab = pickle.load(fin)

    question1 = Input(shape=(15,))
    question2 = Input(shape=(15,))

    q1 = Embedding(vocab.nb_words + 1,
                   300,
                   weights=[vocab.embedding],
                   input_length=15,
                   trainable=False)(question1)
    q1=Dropout(dropoutrate)(q1)
    q1 = TimeDistributed(Dense(300, activation='relu'))(q1)
    q1=Dropout(dropoutrate)(q1)
    # q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(300,))(q1)
    q1 = Conv1D(128, 3, activation='relu')(q1)
    q1=Dropout(dropoutrate)(q1)
    q1 = MaxPool1D()(q1)
    # q1 = Dropout(dropoutrate)(q1)

    q2 = Embedding(vocab.nb_words + 1,
                   300,
                   weights=[vocab.embedding],
                   input_length=15,
                   trainable=False)(question2)
    q2=Dropout(dropoutrate)(q2)
    q2 = TimeDistributed(Dense(300, activation='relu'))(q2)
    q2=Dropout(dropoutrate)(q2)
    # q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(300,))(q2)
    q2 = Conv1D(128, 3, activation='relu')(q2)
    q2=Dropout(dropoutrate)(q2)
    q2 = MaxPool1D()(q2)
    # q2 = Dropout(dropoutrate)(q2)


    merged = Merge(mode='dot', dot_axes=[1, 1])([q1, q2])
    merged =Flatten()(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(dropoutrate)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(dropoutrate)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(128, activation='relu')(merged)
    merged = Dropout(dropoutrate)(merged)
    merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2], outputs=is_duplicate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
