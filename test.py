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
from IPython.display import HTML, display

#
# import en_core_web_sm

# en_core_web_sm = spacy.load('en_core_web_md')

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
        model.summary()

        optimizer = Adam(lr=learning_rate, decay=0.0001)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        return model

    def get_tokenizer_filename(self, saved_model_filename):
        return saved_model_filename + '.tokenizer'

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

    def encode_input(self, x, log=False):
        x = np.array(x)
        if not x.shape:
            x = np.expand_dims(x, 0)
        texts = np.array([normalize(text) for text in x])
        return self.encode_texts(texts)

    def predict(self, x):
        encoded_x = self.encode_texts(x)
        return self.model.predict(encoded_x)

    def activation_maps(self, text, websafe=False):
        normalized_text = normalize(text)

        encoded_text = self.encode_input(text)[0]

        # get word activations

        hidden_word_encoding_out = Model(
            inputs=self.word_attention_model.input,
            outputs=self.word_attention_model.get_layer('dense_transform_word').output)

        hidden_word_encodings = hidden_word_encoding_out.predict(encoded_text)

        word_context = self.word_attention_model.get_layer('word_attention').get_weights()[0]

        dot = np.dot(hidden_word_encodings, word_context)

        # u_wattention = encoded_text*np.exp(np.squeeze(dot))
        u_wattention = encoded_text

        if websafe:
            u_wattention = u_wattention.astype(float)

        nopad_encoded_text = encoded_text[-len(normalized_text):]
        nopad_encoded_text = [list(filter(lambda x: x > 0, sentence)) for sentence in nopad_encoded_text]
        reconstructed_texts = [[self.reverse_word_index[int(i)]
                                for i in sentence] for sentence in nopad_encoded_text]
        nopad_wattention = u_wattention[-len(normalized_text):]
        nopad_wattention = nopad_wattention / np.expand_dims(np.sum(nopad_wattention, -1), -1)
        nopad_wattention = np.array([attention_seq[-len(sentence):]
                                     for attention_seq, sentence in zip(nopad_wattention, nopad_encoded_text)])
        word_activation_maps = []
        for i, text in enumerate(reconstructed_texts):
            word_activation_maps.append(list(zip(text, nopad_wattention[i])))

        hidden_sentence_encoding_out = Model(inputs=self.model.input,
                                             outputs=self.model.get_layer('dense_transform_sentence').output)
        hidden_sentence_encodings = np.squeeze(
            hidden_sentence_encoding_out.predict(np.expand_dims(encoded_text, 0)), 0)
        sentence_context = self.model.get_layer('sentence_attention').get_weights()[0]
        u_sattention = np.exp(np.squeeze(np.dot(hidden_sentence_encodings, sentence_context), -1))
        if websafe:
            u_sattention = u_sattention.astype(float)
        nopad_sattention = u_sattention[-len(normalized_text):]

        nopad_sattention = nopad_sattention / np.expand_dims(np.sum(nopad_sattention, -1), -1)

        activation_map = list(zip(word_activation_maps, nopad_sattention))

        return activation_map

    def load_weights(self, saved_model_dir, saved_model_filename, saved_model_dir2):
        with CustomObjectScope({'Attention': Attention}):
            print(os.path.join(saved_model_dir, saved_model_filename))
            self.model = load_model(saved_model_dir2)
            self.word_attention_model = self.model.get_layer('time_distributed').layer
            tokenizer_path = os.path.join(
                saved_model_dir, self.get_tokenizer_filename(saved_model_filename))
            tokenizer_state = pickle.load(open(tokenizer_path, "rb"))
            self.tokenizer = tokenizer_state['tokenizer']
            self.MAX_SENTENCE_COUNT = tokenizer_state['maxSentenceCount']
            self.MAX_SENTENCE_LENGTH = tokenizer_state['maxSentenceLength']
            self.VOCABULARY_SIZE = tokenizer_state['vocabularySize']
            self.create_reverse_word_index()


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


def clean_str(string):
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    cleanr = re.compile('<.*?>')

    # string = re.sub(r'\d+', '', string)
    string = re.sub(cleanr, '', string)
    # string = re.sub("'", '', string)
    # string = re.sub(r'\W+', ' ', string)
    string = string.replace('_', '')

    return string.strip().lower()


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


tf.compat.v1.disable_eager_execution()


def main():
    graph = tf.compat.v1.get_default_graph()

    text = "I absolutely love Daughters of the Night Sky. " \
           "This is a book that popped up as a free or low-cost Amazon special deal. " \
           "I don't often succumb to these offers, but the brief description and, to be honest, " \
           "the cover art intrigued me. I am so glad I did. I would give this book six stars if I could."
    ntext = normalize(text)
    model = HAHNetwork()
    model.load_weights('./saved_models', './model.h5', './trained_model')

    with graph.as_default():
        activation_maps = model.activation_maps(text, websafe=True)
        preds = model.predict([ntext])[0]
        prediction = np.argmax(preds).astype(float)
        data = {'activations': activation_maps, 'normalizedText': ntext, 'prediction': prediction}
        print("Activations map:")
        print(json.dumps(data))
        display(HTML(
            """<div style="height: 400px">
            <script async src="//jsfiddle.net/luisfredgs/buLaor1x/81/embed/result"></script>
            </div>"""),
                display_id=True)


if __name__ == '__main__':
    main()
