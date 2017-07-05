from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

from tens import homework

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
    tf.test,
    tf.train
]

def load_mnist():
    mnist = fetch_mldata('MNIST original')
    mnist_X, mnist_y = shuffle(mnist.data.astype('float32'),
                               mnist.target.astype('int32'), random_state=42)

    mnist_X = mnist_X / 255.0

    return train_test_split(mnist_X, mnist_y,
                test_size=0.2,
                random_state=42)

def validate_homework():
    train_X, test_X, train_y, test_y = load_mnist()

    # validate for small dataset
    train_X_mini = train_X[:10000]
    train_y_mini = train_y[:10000]
    test_X_mini = test_X[:10000]
    test_y_mini = test_y[:10000]

    pred_y = homework(train_X_mini, train_y_mini, test_X_mini)
    print(f1_score(test_y_mini, pred_y, average='macro'))

def score_homework():
    train_X, test_X, train_y, test_y = load_mnist()
    pred_y = homework(train_X, train_y, test_X)
    print(f1_score(test_y, pred_y, average='macro'))

if __name__ == '__main__':
    # validate_homework()
    score_homework()
