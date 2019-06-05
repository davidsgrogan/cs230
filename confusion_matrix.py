#!/usr/bin/env python3

import numpy as np
import keras.utils
from sklearn.model_selection import train_test_split
import sklearn.metrics

import matplotlib.pyplot as plt

import glob
import time
import os

# We should instead plot a recall bar 1x20 where each square is a different
# color.
# sklearn.metrics.recall_score

# This function is from https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.clim(0, 1)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
#    plt.show()
    plt.close()


def generate_confusion_matrix(model, X_dev, y_dev):
    predictions = model.predict(X_dev, verbose=1)
    print("predictions.shape = ", predictions.shape)
    # This shape is m x number of speakers
    assert predictions.shape == y_dev.shape
    # I think it will be m X num_speakers
    predictions = predictions.argmax(axis=1)
    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_dev.argmax(axis=1), predictions)
    assert (confusion_matrix.shape[0] == confusion_matrix.shape[1] ==
            y_dev.shape[1]), "%s %s" % (confusion_matrix.shape, y_dev.shape)
#    print(confusion_matrix)
    plot_confusion_matrix(confusion_matrix, range(1, 21),
                          cmap=plt.get_cmap('Greens'), normalize=True)
    # Save these to disk in case we want to mess with the confusion matrix
    # directly.
    np.save("X_dev", X_dev)
    np.save("y_dev", y_dev)



if __name__ == '__main__':
    saved_model_name = "baseline_model.h5"
    print("loading model from", saved_model_name)
    model = keras.models.load_model(saved_model_name)
    # Training the model will save these files.
    X_dev = np.load("X_dev.npy")
    y_dev = np.load("y_dev.npy")
    generate_confusion_matrix(model, X_dev, y_dev)
