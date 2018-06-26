# -*- coding:utf8 -*-

import pickle
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras import backend as K
MAX_SEQUENCE_LENGTH = 25
DROPOUT_RATE = 0.2

def max_embedding():
    with open('vocab.data', 'rb') as fin:
        vocab = pickle.load(fin)

    question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

    q1 = Embedding(vocab.nb_words + 1,
                   300,
                   weights=[vocab.embedding],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(question1)
    q1 = TimeDistributed(Dense(300, activation='relu'))(q1)
    q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(300,))(q1)

    q2 = Embedding(vocab.nb_words + 1,
                   300,
                   weights=[vocab.embedding],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(question2)
    q2 = TimeDistributed(Dense(300, activation='relu'))(q2)
    q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(300,))(q2)

    merged = concatenate([q1, q2])
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT_RATE)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT_RATE)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT_RATE)(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(200, activation='relu')(merged)
    merged = Dropout(DROPOUT_RATE)(merged)
    merged = BatchNormalization()(merged)

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2], outputs=is_duplicate)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

