import codecs
import logging

import gensim
import nltk
import pandas as pd
from keras.callbacks import *
from keras.models import *
from tqdm import tqdm

from constants import term_embd_category
from utilityFunctions import scale_text, refine_text


def process_embeddings(term_ind):
    print('process embeddings')
    ind = {}
    codec = codecs.open("wiki-news-300d-1M-subword.vec", encoding='utf-8')
    for ln in tqdm(codec):
        values = ln.rstrip().rsplit(' ')
        term = values[0]
        cf = np.asarray(values[1:], dtype='float32')
        ind[term] = cf
    codec.close()
    print('Total %s term vectors' % len(ind))

    print('Building embed 2d array...')
    terms_absent = []

    embd_2d_array = np.zeros((len(term_ind) + 1, 300))

    for term, x in term_ind.items():
        embd_arr = ind.get(term)
        if (embd_arr is not None) and len(embd_arr) > 0:

            embd_2d_array[x] = embd_arr
        else:
            terms_absent.append(term)

    return embd_2d_array


def build_word_embd_model(size, information):
    f_name = './fasttext_model.txt'

    if not os.path.isfile(f_name):
        print('building word embd model...')
        description_list = []

        for info in information['text']:
            info = refine_text(info)
            descriptions = nltk.tokenize.sent_tokenize(info)
            for single_sentence in descriptions:
                term_list = [term for term in nltk.tokenize.word_tokenize(single_sentence) if term.isalnum()]
                description_list.append(term_list)

        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        word_embeddings_model = gensim.models.FastText(
            word_ngrams=1,
            sentences=description_list,
            size=size,
            workers=os.cpu_count(),
            window=1)
        word_embeddings_model.save("./fasttext_model.txt")


def get_movie_reviews_yelp(json_location, train_ratio=1, size=400000):
    with open(json_location) as f:
        data = f.read().strip().split("\n")

    json_data_frame = pd.DataFrame([json.loads(data_instance) for data_instance in data])

    content_symbols = []
    for line in tqdm(json_data_frame['text']):
        content_symbols.append(scale_text(line))

    json_data_frame['text_tokens'] = content_symbols
    del content_symbols

    vec_size = 200
    if term_embd_category == 'from_scratch':
        build_word_embd_model(vec_size, json_data_frame)

    attributes = json_data_frame['text_tokens'].values
    labels = pd.get_dummies(json_data_frame['stars']).values

    return attributes, labels


def get_movie_reviews_imdb(csv_location, size=49000, train_ratio=1):
    csv_data_frame = pd.read_csv(csv_location, nrows=size, usecols=['text', 'sentiment'])

    vec_size = 200

    if term_embd_category == 'from_scratch':
        build_word_embd_model(vec_size, csv_data_frame)

    content_symbols = []
    for line in tqdm(csv_data_frame['text']):
        content_symbols.append(scale_text(line))

    csv_data_frame['text_tokens'] = content_symbols

    del content_symbols

    attributes = csv_data_frame['text_tokens'].values
    labels = pd.get_dummies(csv_data_frame['sentiment']).values
    return attributes, labels
