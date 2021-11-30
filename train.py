import codecs
import logging
import pickle
import random

import gensim
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras import regularizers
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from numpy.random import seed
from sklearn.manifold import TSNE
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

#
# import en_core_web_sm

# en_core_web_sm = spacy.load('en_core_web_md')

tf.compat.v1.disable_eager_execution()

dataset = "imdb"  # @param ["yelp", "imdb"]

word_embedding_type = "from_scratch"  # @param ["from_scratch", "pre_trained"]
word_vector_model = "fasttext"  # @param ["fasttext"]
rnn_type = "LSTM"  # @param ["LSTM", "GRU"]
learning_rate = 0.001
epochs = 8
batch_size = 64

nltk.download('punkt')
nltk.download('stopwords')

# from tensorflow import set_random_seed
os.environ['PYTHONHASHSEED'] = str(1024)
tf.random.set_seed(1024)
seed(1024)
np.random.seed(1024)
random.seed(1024)


def clean_str(string_):
    string_ = re.sub(r"\'s", " \'s", string_)
    string_ = re.sub(r"\'ve", " \'ve", string_)
    string_ = re.sub(r"n\'t", " n\'t", string_)
    string_ = re.sub(r"\'re", " \'re", string_)
    string_ = re.sub(r"\'d", " \'d", string_)
    string_ = re.sub(r"\'ll", " \'ll", string_)
    string_ = re.sub(r",", " , ", string_)
    string_ = re.sub(r"!", " ! ", string_)
    string_ = re.sub(r"\(", " \( ", string_)
    string_ = re.sub(r"\)", " \) ", string_)
    string_ = re.sub(r"\?", " \? ", string_)
    string_ = re.sub(r"\s{2,}", " ", string_)

    cleanr = re.compile('<.*?>')

    # string = re.sub(r'\d+', '', string)
    string_ = re.sub(cleanr, '', string_)
    # string = re.sub("'", '', string)
    # string = re.sub(r'\W+', ' ', string)
    string_ = string_.replace('_', '')

    return string_.strip().lower()


def load_subword_embedding_300d(word_index):
    print('load_subword_embedding...')
    embeddings_index = {}
    f = codecs.open("wiki-news-300d-1M-subword.vec", encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('found %s word vectors' % len(embeddings_index))

    # embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []

    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:

            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)

    return embedding_matrix


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)


nlp = spacy.load('en_core_web_md')

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
          '≤', '‡', '√', '#', '—–']


def clean_puncts(x):
    x = str(x)
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def remove_stopwords(text):
    text = str(text)
    ## Convert words to lower case and split them
    text = text.lower().split()

    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    text = " ".join(text)

    return text


def normalize(text):
    text = text.lower().strip()
    doc = nlp(text)
    filtered_sentences = []
    for sentence in doc.sents:
        sentence = clean_puncts(sentence)
        sentence = clean_str(sentence)
        # sentence = remove_stopwords(sentence)
        filtered_sentences.append(sentence)
    return filtered_sentences


def create_fasttext(embed_dim, data):
    filename = './fasttext_model.txt'

    if not os.path.isfile(filename):
        print('create_fasttext...')
        sent_lst = []

        for doc in data['text']:
            doc = clean_str(doc)
            sentences = nltk.tokenize.sent_tokenize(doc)
            for sent in sentences:
                word_lst = [w for w in nltk.tokenize.word_tokenize(sent) if w.isalnum()]
                sent_lst.append(word_lst)

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        fasttext_model = gensim.models.FastText(
            word_ngrams=1,
            sentences=sent_lst,
            size=embed_dim,
            workers=os.cpu_count(),
            window=1)
        fasttext_model.save("./fasttext_model.txt")


def load_data_yelp(path, train_ratio=1, size=400000):
    with open(path) as f:
        reviews = f.read().strip().split("\n")

    df = pd.DataFrame([json.loads(review) for review in reviews])

    text_tokens = []
    for row in tqdm(df['text']):
        text_tokens.append(normalize(row))

    df['text_tokens'] = text_tokens
    del text_tokens

    vector_dim = 200
    if word_embedding_type == 'from_scratch':
        create_fasttext(vector_dim, df)

    ###

    X = df['text_tokens'].values
    Y = pd.get_dummies(df['stars']).values

    return X, Y


def load_data_imdb(path, size=49000, train_ratio=1):
    df = pd.read_csv(path, nrows=size, usecols=['text', 'sentiment'])

    vector_dim = 200

    if word_embedding_type == 'from_scratch':
        # Fasttext
        create_fasttext(vector_dim, df)

    ###

    text_tokens = []
    for row in tqdm(df['text']):
        text_tokens.append(normalize(row))

    df['text_tokens'] = text_tokens

    del text_tokens
    ###

    X = df['text_tokens'].values
    Y = pd.get_dummies(df['sentiment']).values
    return X, Y


class Attention(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = 38
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], 1)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self._trainable_weights = [self.W, self.b, self.u]
        super(Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None, **kwargs):
        uit = K.tanh(K.dot(x, self.W) + self.b)
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)
        ait = K.exp(ait)

        if mask is not None:
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        ait = K.expand_dims(ait)
        weighted_input = x * ait

        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class HAHNetwork:
    def __init__(self):
        self.model = None
        self.MAX_SENTENCE_LENGTH = 0
        self.MAX_SENTENCE_COUNT = 0
        self.VOCABULARY_SIZE = 0
        self.word_embedding = None
        self.model = None
        self.word_attention_model = None
        self.tokenizer = None
        self.class_count = 2

    def build_model(self, n_classes=2, embedding_dim=200, embeddings_path=False):

        l2_reg = regularizers.l2(0.001)

        embedding_weights = np.random.normal(0, 1, (len(self.tokenizer.word_index) + 1, embedding_dim))

        if embeddings_path is not None:

            if word_embedding_type == 'from_scratch':
                # FastText
                filename = './fasttext_model.txt'
                model = gensim.models.FastText.load(filename)

                embeddings_index = model.wv
                embedding_matrix = np.zeros((len(self.tokenizer.word_index) + 1, embedding_dim))
                for word, i in self.tokenizer.word_index.items():
                    try:
                        embedding_vector = embeddings_index[word]
                        if embedding_vector is not None:
                            embedding_matrix[i] = embedding_vector
                    except Exception as e:
                        # print(str(e))
                        continue
            else:
                embedding_dim = 300
                embedding_matrix = load_subword_embedding_300d(self.tokenizer.word_index)

            embedding_weights = embedding_matrix

        sentence_in = Input(shape=(self.MAX_SENTENCE_LENGTH,), dtype='int32', name="input_1")

        embedding_trainable = True

        if word_embedding_type == 'pre_trained':
            embedding_trainable = False

        embedded_word_seq = Embedding(
            self.VOCABULARY_SIZE,
            embedding_dim,
            weights=[embedding_weights],
            input_length=self.MAX_SENTENCE_LENGTH,
            trainable=embedding_trainable,
            # mask_zero=True,
            mask_zero=False,
            name='word_embeddings', )(sentence_in)

        dropout = Dropout(0.2)(embedded_word_seq)
        filter_sizes = [3, 4, 5]
        convs = []
        for filter_size in filter_sizes:
            conv = Conv1D(filters=64, kernel_size=filter_size, padding='same', activation='relu')(dropout)
            pool = MaxPool1D(filter_size)(conv)
            convs.append(pool)

        concatenate = Concatenate(axis=1)(convs)

        if rnn_type == 'GRU':
            # word_encoder = Bidirectional(CuDNNGRU(50, return_sequences=True, dropout=0.2))(concatenate)
            dropout = Dropout(0.1)(concatenate)
            word_encoder = Bidirectional(CuDNNGRU(50, return_sequences=True))(dropout)
        else:
            word_encoder = Bidirectional(
                LSTM(50, return_sequences=True, dropout=0.2))(embedded_word_seq)

        dense_transform_word = Dense(
            100,
            activation='relu',
            name='dense_transform_word',
            kernel_regularizer=l2_reg)(word_encoder)

        # word attention
        attention_weighted_sentence = Model(
            sentence_in, Attention(name="word_attention")(dense_transform_word))

        self.word_attention_model = attention_weighted_sentence

        attention_weighted_sentence.summary()

        # sentence-attention-weighted document scores

        texts_in = Input(shape=(self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH), dtype='int32', name="input_2")

        attention_weighted_sentences = TimeDistributed(attention_weighted_sentence)(texts_in)

        if rnn_type == 'GRU':
            # sentence_encoder = Bidirectional(GRU(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.2))(attention_weighted_sentences)
            dropout = Dropout(0.1)(attention_weighted_sentences)
            sentence_encoder = Bidirectional(CuDNNGRU(50, return_sequences=True))(dropout)
        else:
            sentence_encoder = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.2))(
                attention_weighted_sentences)

        dense_transform_sentence = Dense(
            100,
            activation='relu',
            name='dense_transform_sentence',
            kernel_regularizer=l2_reg)(sentence_encoder)

        # sentence attention
        attention_weighted_text = Attention(name="sentence_attention")(dense_transform_sentence)

        prediction = Dense(n_classes, activation='softmax')(attention_weighted_text)

        model = Model(texts_in, prediction)
        model.call = tf.function(model.call)
        model.summary()

        optimizer = Adam(learning_rate=learning_rate, decay=0.0001)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        return model

    def get_tokenizer_filename(self, saved_model_filename):
        return saved_model_filename + '.tokenizer'

    def fit_on_texts(self, texts):
        self.tokenizer = Tokenizer(filters='"()*,-/;[\]^_`{|}~', oov_token='UNK')
        all_sentences = []
        max_sentence_count = 0
        max_sentence_length = 0
        for text in texts:
            sentence_count = len(text)
            if sentence_count > max_sentence_count:
                max_sentence_count = sentence_count
            for sentence in text:
                sentence_length = len(sentence)
                if sentence_length > max_sentence_length:
                    max_sentence_length = sentence_length
                all_sentences.append(sentence)

        self.MAX_SENTENCE_COUNT = min(max_sentence_count, 15)
        self.MAX_SENTENCE_LENGTH = min(max_sentence_length, 50)

        self.tokenizer.fit_on_texts(all_sentences)
        self.VOCABULARY_SIZE = len(self.tokenizer.word_index) + 1
        self.create_reverse_word_index()

    def create_reverse_word_index(self):
        self.reverse_word_index = {value: key for key, value in self.tokenizer.word_index.items()}

    def encode_texts(self, texts):
        encoded_texts = np.zeros((len(texts), self.MAX_SENTENCE_COUNT, self.MAX_SENTENCE_LENGTH))
        for i, text in enumerate(texts):
            encoded_text = np.array(pad_sequences(
                self.tokenizer.texts_to_sequences(text),
                maxlen=self.MAX_SENTENCE_LENGTH))[:self.MAX_SENTENCE_COUNT]
            encoded_texts[i][-len(encoded_text):] = encoded_text
        return encoded_texts

    def save_tokenizer_on_epoch_end(self, path, epoch):
        if epoch == 0:
            tokenizer_state = {
                'tokenizer': self.tokenizer,
                'maxSentenceCount': self.MAX_SENTENCE_COUNT,
                'maxSentenceLength': self.MAX_SENTENCE_LENGTH,
                'vocabularySize': self.VOCABULARY_SIZE
            }
            pickle.dump(tokenizer_state, open(path, "wb"))

    def train(self, train_x, train_y,
              batch_size=16,
              epochs=1,
              embedding_dim=200,
              embeddings_path=False,
              saved_model_dir='saved_models',
              saved_model_filename=None, ):

        self.fit_on_texts(train_x)
        self.model = self.build_model(
            n_classes=train_y.shape[-1],
            embedding_dim=200,
            embeddings_path=embeddings_path)
        encoded_train_x = self.encode_texts(train_x)
        callbacks = [
            ReduceLROnPlateau(),
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: self.save_tokenizer_on_epoch_end(
                    os.path.join(saved_model_dir,
                                 self.get_tokenizer_filename(saved_model_filename)), epoch))
        ]

        # if saved_model_filename:
        #     callbacks.append(
        #         ModelCheckpoint(
        #             filepath=os.path.join(saved_model_dir, saved_model_filename),
        #             monitor='val_accuracy',
        #             save_best_only=True,
        #             save_weights_only=False,
        #         )
        #     )
        history = self.model.fit(
            x=encoded_train_x,
            y=train_y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            validation_split=0.1,
            shuffle=True)

        self.model.save("trained_model")

        # Plot
        print(history.history.keys())

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def encode_input(self, x, log=False):
        x = np.array(x)
        if not x.shape:
            x = np.expand_dims(x, 0)
        texts = np.array([normalize(text) for text in x])
        return self.encode_texts(texts)


if __name__ == '__main__':
    YELP_DATA_PATH = 'yelp_reviews_sampling.json'
    IMDB_DATA_PATH = 'imdb_reviews.csv'
    SAVED_MODEL_DIR = 'saved_models'
    SAVED_MODEL_FILENAME = 'model.h5'

    if dataset == 'yelp':
        if not os.path.isfile("yelpTraining.npy"):
            (X, Y) = load_data_yelp(path=YELP_DATA_PATH, size=400000)
            X_arr = np.array(X)
            np.save("yelpTraining.npy", X_arr)

            Y_arr = np.array(Y)
            np.save("yelpLabels.npy", Y_arr)
        else:
            (X, Y) = np.load("yelpTraining.npy", allow_pickle=True), np.load("yelpLabels.npy", allow_pickle=True)
    else:
        if not os.path.isfile("trainingData.npy"):
            (X, Y) = load_data_imdb(path=IMDB_DATA_PATH, size=49000)

            X_arr = np.array(X)
            np.save("trainingData.npy", X_arr)

            Y_arr = np.array(Y)
            np.save("labels.npy", Y_arr)
        else:
            (X, Y) = np.load("trainingData.npy", allow_pickle=True), np.load("labels.npy", allow_pickle=True)

    limit = 200
    vector_dim = 200

    # Fasttext
    filename = './fasttext_model.txt'
    model = gensim.models.FastText.load(filename)
    words = []
    embedding = np.array([])
    i = 0
    for word in model.wv.vocab:
        if i == limit: break

        words.append(word)
        embedding = np.append(embedding, model[word])
        i += 1

    embedding = embedding.reshape(limit, vector_dim)
    tsne = TSNE(n_components=2)
    low_dim_embedding = tsne.fit_transform(embedding)

    plot_with_labels(low_dim_embedding, words)

    # K.clear_session()
    # tf.compat.v1.reset_default_graph()
    model = HAHNetwork()
    model.train(X, Y, batch_size=64, epochs=8, embeddings_path=True, saved_model_dir=SAVED_MODEL_DIR,
                saved_model_filename=SAVED_MODEL_FILENAME)
