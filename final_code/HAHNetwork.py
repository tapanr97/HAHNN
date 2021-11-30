import pickle

import gensim
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import regularizers
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy.random import seed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GRU

from constants import term_embd_category, recurrent_neural_network_type, lr
from dataLoadUtilities import process_embeddings, scale_text
from AttentionLayer import Attention as Att
from tcn import TCN


class HAHNetwork:
    def __init__(self):
        self.LARGEST_SEN_SIZE = 0
        self.LARGEST_SEN_ENUM = 0
        self.LENGTH_OF_VOCAB = 0
        self.term_embd = None
        self.model = None
        self.term_att_final = None
        self.term_splitter = None
        self.no_of_labels = 2

    def prepare(self, no_of_labels=2, embd_size=200, embd_location=False):

        l2_reg = regularizers.l2(0.001)

        embd_w = np.random.normal(0, 1, (len(self.term_splitter.word_index) + 1, embd_size))

        if embd_location is not None:

            if term_embd_category == 'from_scratch':
                # FastText
                f_n = '../fasttext_model.txt'
                term_embd_m = gensim.models.FastText.load(f_n)

                embd_position = term_embd_m.wv
                embd_2d_array = np.zeros((len(self.term_splitter.word_index) + 1, embd_size))
                for term, x in self.term_splitter.word_index.items():
                    try:
                        embd_vec = embd_position[term]
                        if embd_vec is not None:
                            embd_2d_array[x] = embd_vec
                    except Exception as e:
                        continue
            else:
                embd_size = 300
                embd_2d_array = process_embeddings(self.term_splitter.word_index)

            embd_w = embd_2d_array

        input_shape = Input(shape=(self.LARGEST_SEN_SIZE,), dtype='int32', name="input_1")

        embd_instruct = True

        if term_embd_category == 'pre_trained':
            embd_instruct = False

        embd_term_series = Embedding(
            self.LENGTH_OF_VOCAB,
            embd_size,
            weights=[embd_w],
            input_length=self.LARGEST_SEN_SIZE,
            trainable=embd_instruct,
            mask_zero=False,
            name='word_embeddings', )(input_shape)

        dropout = Dropout(0.2)(embd_term_series)
        f_shapes = [3, 4, 5]
        cnn_list = []
        for f_shape in f_shapes:
            conv = Conv1D(filters=64, kernel_size=f_shape, padding='same', activation='relu')(dropout)
            pool = MaxPool1D(f_shape)(conv)
            cnn_list.append(pool)

        append = Concatenate(axis=1)(cnn_list)
        # append = TCN(nb_filters=64, kernel_size=6, dilations=[1, 2, 4, 8, 16, 32, 64])(dropout)

        if recurrent_neural_network_type == 'GRU':
            dropout = Dropout(0.1)(append)
            # term_e = Bidirectional(CuDNNGRU(50, return_sequences=True))(dropout)
            term_e = Bidirectional(GRU(50, return_sequences=True))(dropout)
        else:
            term_e = Bidirectional(
                LSTM(50, return_sequences=True, dropout=0.2))(embd_term_series)

        fnn_for_term = Dense(
            100,
            activation='relu',
            name='dense_transform_word',
            kernel_regularizer=l2_reg)(term_e)

        term_att = Model(
            input_shape, Att(name="word_attention")(fnn_for_term))

        self.term_att_final = term_att

        term_att.summary()

        input_shape_2 = Input(shape=(self.LARGEST_SEN_ENUM, self.LARGEST_SEN_SIZE), dtype='int32', name="input_2")

        term_atts = TimeDistributed(term_att)(input_shape_2)

        if recurrent_neural_network_type == 'GRU':
            dropout = Dropout(0.1)(term_atts)
            # sen_e = Bidirectional(CuDNNGRU(50, return_sequences=True))(dropout)
            sen_e = Bidirectional(GRU(50, return_sequences=True))(dropout)
        else:
            sen_e = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.2))(
                term_atts)

        fnn_for_sen = Dense(
            100,
            activation='relu',
            name='dense_transform_sentence',
            kernel_regularizer=l2_reg)(sen_e)

        txt_att = Att(name="sentence_attention")(fnn_for_sen)

        sentiment = Dense(no_of_labels, activation='softmax')(txt_att)

        term_embd_m = Model(input_shape_2, sentiment)
        term_embd_m.call = tf.function(term_embd_m.call)
        term_embd_m.summary()

        opt = Adam(learning_rate=lr, decay=0.0001)

        term_embd_m.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        return term_embd_m

    def get_tokenizer_filename(self, base_name):
        return base_name + '.tokenizer'

    def adjust_content(self, content):
        self.term_splitter = Tokenizer(filters='"()*,-/;[\]^_`{|}~', oov_token='UNK')
        sen_list = []
        largest_cnt = 0
        lagest_len = 0
        for c in content:
            s_cnt = len(c)
            if s_cnt > largest_cnt:
                largest_cnt = s_cnt
            for s in c:
                s_size = len(s)
                if s_size > lagest_len:
                    lagest_len = s_size
                sen_list.append(s)

        self.LARGEST_SEN_ENUM = min(largest_cnt, 15)
        self.LARGEST_SEN_SIZE = min(lagest_len, 50)

        self.term_splitter.fit_on_texts(sen_list)
        self.LENGTH_OF_VOCAB = len(self.term_splitter.word_index) + 1
        self.make_opposite_term_position()

    def make_opposite_term_position(self):
        self.reverse_word_index = {value: position for position, value in self.term_splitter.word_index.items()}

    def enc_content(self, content):
        out = np.zeros((len(content), self.LARGEST_SEN_ENUM, self.LARGEST_SEN_SIZE))
        for position, c in enumerate(content):
            single_out = np.array(pad_sequences(
                self.term_splitter.texts_to_sequences(c),
                maxlen=self.LARGEST_SEN_SIZE))[:self.LARGEST_SEN_ENUM]
            out[position][-len(single_out):] = single_out
        return out

    def store_t_model(self, location, current_e):
        if current_e == 0:
            st = {
                'tokenizer': self.term_splitter,
                'maxSentenceCount': self.LARGEST_SEN_ENUM,
                'maxSentenceLength': self.LARGEST_SEN_SIZE,
                'vocabularySize': self.LENGTH_OF_VOCAB
            }
            pickle.dump(st, open(location, "wb"))

    def process_final_model(self, x_t, y_t,
                            batch_size=16,
                            epochs=1,
                            embd_size=200,
                            embd_location=False,
                            tokenizer_dir='saved_models',
                            base_name=None, ):

        self.adjust_content(x_t)
        self.model = self.prepare(
            no_of_labels=y_t.shape[-1],
            embd_size=200,
            embd_location=embd_location)
        x_t_enc = self.enc_content(x_t)
        callbacks = [
            ReduceLROnPlateau(),
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.store_t_model(
                    os.path.join(tokenizer_dir,
                                 self.get_tokenizer_filename(base_name)), epoch))
        ]

        if base_name:
            callbacks.append(
                ModelCheckpoint(
                    filepath=os.path.join(tokenizer_dir, base_name),
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    save_weights_only=False,
                )
            )

        trained_model = self.model.fit(
            x=x_t_enc,
            y=y_t,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_split=0.1,
            shuffle=True)

        # self.model.save("trained_model")

        print(trained_model.history.keys())

        plt.plot(trained_model.history['accuracy'])
        plt.plot(trained_model.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(trained_model.history['loss'])
        plt.plot(trained_model.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def in_enc(self, _in, log=False):
        _in = np.array(_in)
        if not _in.shape:
            _in = np.expand_dims(_in, 0)
        out = np.array([scale_text(sen) for sen in _in])
        return self.enc_content(out)
