import sys
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

from rnn import homework

del [
    tf.app,
    tf.compat,
    tf.contrib,
    tf.errors,
    tf.gfile,
    tf.graph_util,
    tf.image,
    tf.layers,
    tf.logging,
    tf.losses,
    tf.metrics,
    tf.python_io,
    tf.resource_loader,
    tf.saved_model,
    tf.sdca,
    tf.sets,
    tf.summary,
    tf.sysconfig,
    tf.test
]

sys.modules['keras'] = None


def validate_homework():
    global num_words

    num_words = 10000
    (train_X, train_y), (test_X, test_y) = imdb.load_data(num_words=num_words, seed=42, start_char=0, oov_char=1, index_from=2)

    # validate for small dataset
    train_X_mini = train_X[:100]
    train_y_mini = train_y[:100]
    test_X_mini = test_X[:100]
    test_y_mini = test_y[:100]

    pred_y = homework(train_X_mini, train_y_mini, test_X_mini)
    true_y = test_y_mini.tolist()

    print(f1_score(true_y, pred_y, average='macro'))


def score_homework():
    global num_words
    num_words = 10000

    (train_X, train_y), (test_X, test_y) = imdb.load_data(num_words=num_words, seed=42, start_char=0, oov_char=1, index_from=2)

    pred_y = homework(train_X, train_y, test_X)
    true_y = test_y.tolist()

    print(f1_score(true_y, pred_y, average='macro'))


if __name__ == '__main__':
    validate_homework()
    # score_homework()
