import random

import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.random import seed
from sklearn.manifold import TSNE

from HAHNetwork import HAHNetwork
from constants import *
from dataLoadUtilities import *

tf.compat.v1.disable_eager_execution()

nltk.download('punkt')
nltk.download('stopwords')

os.environ['PYTHONHASHSEED'] = str(1024)
tf.random.set_seed(1024)
seed(1024)
np.random.seed(1024)
random.seed(1024)


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


if __name__ == '__main__':
    YELP_DATA_PATH = '../yelp_reviews_sampling.json'
    IMDB_DATA_PATH = '../imdb_reviews.csv'
    SAVED_MODEL_DIR = '../saved_models'
    SAVED_MODEL_FILENAME = 'model.tf'

    if dataset == 'yelp':
        if not os.path.isfile("../yelpTraining.npy"):
            (X, Y) = get_movie_reviews_yelp(json_location=YELP_DATA_PATH, size=400000)
            X_arr = np.array(X)
            np.save("../yelpTraining.npy", X_arr)

            Y_arr = np.array(Y)
            np.save("../yelpLabels.npy", Y_arr)
        else:
            (X, Y) = np.load("../yelpTraining.npy", allow_pickle=True), np.load("../yelpLabels.npy", allow_pickle=True)
    else:
        if not os.path.isfile("../trainingData.npy"):
            (X, Y) = get_movie_reviews_imdb(csv_location=IMDB_DATA_PATH, size=49000)

            X_arr = np.array(X)
            np.save("../trainingData.npy", X_arr)

            Y_arr = np.array(Y)
            np.save("../labels.npy", Y_arr)
        else:
            (X, Y) = np.load("../trainingData.npy", allow_pickle=True), np.load("../labels.npy", allow_pickle=True)

    limit = 200
    vector_dim = 200

    # Fasttext
    filename = '../fasttext_model.txt'
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
    model.process_final_model(X, Y, batch_size=64, epochs=8, embd_location=True, tokenizer_dir=SAVED_MODEL_DIR,
                              base_name=SAVED_MODEL_FILENAME)
