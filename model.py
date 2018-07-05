# -*- coding:utf8 -*-

import pickle
from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense, Lambda, concatenate, Dropout, BatchNormalization, LSTM, Conv1D
from keras.layers.embeddings import Embedding
from keras import backend as K
from layers import Position_Embedding
MAX_SEQUENCE_LENGTH = 15
DROPOUT_RATE = 0

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

def cnn_lstm_f1():
    with open('vocab.data', 'rb') as fin:
        vocab = pickle.load(fin)

    question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

    q1 = Embedding(vocab.nb_words + 1,
                   300,
                   weights=[vocab.embedding],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(question1)

    q2 = Embedding(vocab.nb_words + 1,
                   300,
                   weights=[vocab.embedding],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(question2)

    f_rnn = LSTM(30, return_sequences=True, implementation=1)
    b_rnn = LSTM(30, return_sequences=True, implementation=1, go_backwards=True)

    pos = Position_Embedding(mode='concat')

    qf_rnn = f_rnn(q1)
    qb_rnn = b_rnn(q1)

    # q1_rnn = merge([qf_rnn, qb_rnn], mode='concat', concat_axis=-1)
    q1_rnn = concatenate([qf_rnn, qb_rnn], axis=-1)
    q1_rnn = pos(q1_rnn)
    q1_rnn = BatchNormalization()(q1_rnn)

    af_rnn = f_rnn(q2)
    ab_rnn = b_rnn(q2)
    # q2_rnn = merge([af_rnn, ab_rnn], mode='concat', concat_axis=-1)
    q2_rnn = concatenate([af_rnn, ab_rnn], axis=-1)
    q2_rnn = pos(q2_rnn)
    q2_rnn = BatchNormalization()(q2_rnn)

    # cnn
    cnns = [Conv1D(kernel_size=kernel_size,
                   filters=100,
                   activation='tanh',
                   padding='same') for kernel_size in [1, 2, 3, 5]]
    # qq_cnn = merge([cnn(question_pool) for cnn in cnns], mode='concat')
    q1_cnn = concatenate([cnn(q1_rnn) for cnn in cnns], axis=-1)
    # q2_cnn = merge([cnn(answer_pool) for cnn in cnns], mode='concat')
    q2_cnn = concatenate([cnn(q2_rnn) for cnn in cnns], axis=-1)

    maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
    maxpool.supports_masking = True
    q1_pool = Dropout(DROPOUT_RATE)(maxpool(q1_cnn))
    q2_pool = Dropout(DROPOUT_RATE)(maxpool(q2_cnn))
    merged1 = Dense(100, activation='relu')(q1_pool)
    merged2 = Dense(100, activation='relu')(q2_pool)
    merged = concatenate([merged1, merged2])

    is_duplicate = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[question1, question2], outputs=is_duplicate)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def bilstm():
    with open('vocab.data', 'rb') as fin:
        vocab = pickle.load(fin)

    question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))
    # feature_input = Input(shape=(feature_num,))
    # feature = Dense(32, activation='relu')(feature_input)
    # feature = Dropout()(feature)
    # feature = Dropout(DROPOUT_RATE)(feature)
    # feature = Dense(64, activation='relu')(feature)

    f_rnn = LSTM(256, return_sequences=True, consume_less='mem')
    b_rnn = LSTM(256, return_sequences=True, consume_less='mem', go_backwards=True)
    q1 = Embedding(vocab.nb_words + 1,
                   300,
                   weights=[vocab.embedding],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(question1)
    # q1 = TimeDistributed(Dense(300, activation='relu'))(q1)
    # q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(300,))(q1)
    q1_f = f_rnn(q1)
    q1_b = b_rnn(q1)
    q1 = concatenate([q1_f, q1_b], axis=-1)
    q1 = TimeDistributed(Dense(300, activation='relu'))(q1)
    q1 = Lambda(lambda x: K.max(x, axis=1), output_shape=(300,))(q1)



    q2 = Embedding(vocab.nb_words + 1,
                   300,
                   weights=[vocab.embedding],
                   input_length=MAX_SEQUENCE_LENGTH,
                   trainable=False)(question2)
    # q2 = TimeDistributed(Dense(300, activation='relu'))(q2)
    # q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(300,))(q2)
    q2_f = f_rnn(q2)
    q2_b = b_rnn(q2)
    q2 = concatenate([q2_f, q2_b], axis=-1)
    q2=Dense(128)(q2)
    q2 = TimeDistributed(Dense(300, activation='relu'))(q2)
    q2 = Lambda(lambda x: K.max(x, axis=1), output_shape=(300,))(q2)

    merged = concatenate([q1, q2])
    # merged=concatenate([merged,feature])
    merged = Dropout(DROPOUT_RATE)(merged)
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
