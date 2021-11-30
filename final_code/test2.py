import pickle
import random

import numpy as np
import tensorflow as tf
from IPython.display import HTML, display
from keras import backend as K
from keras import initializers
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from numpy.random import seed

import seaborn as sns
import matplotlib.pyplot as plt

from dataLoadUtilities import *

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

os.environ['PYTHONHASHSEED'] = str(1024)
tf.random.set_seed(1024)
seed(1024)
np.random.seed(1024)
random.seed(1024)


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
        texts = np.array([scale_text(text) for text in x])
        return self.encode_texts(texts)

    def predict(self, x):
        encoded_x = self.encode_texts(x)
        return self.model.predict(encoded_x)

    def activation_maps(self, text, websafe=False):
        normalized_text = scale_text(text)

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

    def load_weights(self, saved_model_dir, saved_model_filename):
        with CustomObjectScope({'Attention': Attention}):
            print(os.path.join(saved_model_dir, saved_model_filename))
            self.model = load_model(os.path.join(saved_model_dir, saved_model_filename))
            self.word_attention_model = self.model.get_layer('time_distributed').layer
            tokenizer_path = os.path.join(
                saved_model_dir, self.get_tokenizer_filename(saved_model_filename))
            tokenizer_state = pickle.load(open(tokenizer_path, "rb"))
            self.tokenizer = tokenizer_state['tokenizer']
            self.MAX_SENTENCE_COUNT = tokenizer_state['maxSentenceCount']
            self.MAX_SENTENCE_LENGTH = tokenizer_state['maxSentenceLength']
            self.VOCABULARY_SIZE = tokenizer_state['vocabularySize']
            self.create_reverse_word_index()


def main():
    graph = tf.compat.v1.get_default_graph()

    text = "When I saw the elaborate DVD box for this and the dreadful Red Queen figurine, " \
           "I felt certain I was in for a big disappointment, but surprise, surprise, I loved it. " \
           "Convoluted nonsense of course and unforgivable that such a complicated denouement should be " \
           "rushed to the point of barely being able to read the subtitles, " \
           "let alone take in the ridiculous explanation. These quibbles apart, however, " \
           "the film is a dream. Fabulous ladies in fabulous outfits in wonderful settings and the whole " \
           "thing constantly on the move and accompanied by a wonderful Bruno Nicolai score. " \
           "He may not be Morricone but in these lighter pieces he might as well be so. " \
           "Really enjoyable with lots of colour, plenty of sexiness, " \
           "some gory kills and minimal police interference. Super."
    ntext = scale_text(text)
    model = HAHNetwork()
    model.load_weights('../saved_models', './model.tf')

    with graph.as_default():
        activation_maps = model.activation_maps(text, websafe=True)
        preds = model.predict([ntext])[0]
        prediction = np.argmax(preds).astype(float)

        _data = []
        _text = []

        mx_len = 0
        for i in range(len(activation_maps)):
            mx_len = max(mx_len, len(activation_maps[i][0]))

        mx_len /= 2

        for i in range(len(activation_maps)):
            l = len(activation_maps[i][0])
            for j in range(0, l, 5):
                arr = [] * 5
                labels = [] * 5
                if j + 5 <= l:
                    for k in range(j, j + 5):
                        arr.append(activation_maps[i][0][k][1])
                        labels.append(activation_maps[i][0][k][0])
                else:
                    for k in range(j, l):
                        arr.append(activation_maps[i][0][k][1])
                        labels.append(activation_maps[i][0][k][0])
                    for k in range(l, j + 5):
                        arr.append(0)
                        labels.append("")
                _data.append(np.array(arr))
                _text.append(np.array(labels))

        _data = np.array(_data)
        _text = np.array(_text)

        fig, ax = plt.subplots()
        sns.set(font_scale=1)
        ax = sns.heatmap(_data, annot=_text, fmt="", cmap="Reds", linewidths=.5, ax=ax)
        # ax.text(0.5, 0.8, 'Test', color='red',
        #         bbox=dict(facecolor='none', edgecolor='red'))

        plt.show()

        data = {'activations': activation_maps, 'normalizedText': ntext, 'prediction': prediction}
        print("Activations map:")
        print(json.dumps(data))


if __name__ == '__main__':
    main()
