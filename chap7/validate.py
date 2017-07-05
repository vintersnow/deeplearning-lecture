from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
from cnn import homework

import numpy as np
import tensorflow as tf

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


def load_mnist():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    mnist_X = np.r_[mnist.train.images, mnist.test.images]
    mnist_y = np.r_[mnist.train.labels, mnist.test.labels]
    return train_test_split(mnist_X, mnist_y, test_size=0.2, random_state=42)


def validate_homework():
    train_X, test_X, train_y, test_y = load_mnist()
    train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
    test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

    # validate for small dataset
    n = 1000
    train_X_mini = train_X[:n]
    train_y_mini = train_y[:n]
    test_X_mini = test_X[:n]
    test_y_mini = test_y[:n]

    pred_y = homework(train_X_mini, train_y_mini, test_X_mini)
    print(f1_score(np.argmax(test_y_mini, 1), pred_y, average='macro'))


def score_homework():
    train_X, test_X, train_y, test_y = load_mnist()
    train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
    test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

    pred_y = homework(train_X, train_y, test_X)
    print(f1_score(np.argmax(test_y, 1), pred_y, average='macro'))


if __name__ == '__main__':
    # validate_homework()
    score_homework()
