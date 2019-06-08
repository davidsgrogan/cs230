#!/usr/bin/env python3
#%%
import numpy as np
import keras.utils
from sklearn.model_selection import train_test_split
import sklearn.metrics
import librosa
import matplotlib.pyplot as plt

import glob
import time
import os
import random
import sys

top_20 = [
    # This is speaker 1 and the list is always in this order.
    "nowitcanbetold",
    "unspokensermons",
    "littlewomen",
    "artcookerymadeplaineasy1784",
    "historymathematics",
    "mysticalcityofgod1",
    "newtestament",
    "uncletomscabin",
    "seapower",
    "wildwales",
    "originofspecies",
    "mysteries",
    "1001nacht2",
    "rkopis",
    "innocentsabroad",
    "woutertjepieterse",
    "geschichtedespeloponnesischenkriegs",
    "vanityfair",
    "jerusalemrevelationsr",
    "worldenglishbible",
]

# We should instead plot a recall bar 1x20 where each square is a different
# color.
# sklearn.metrics.recall_score

# This function is from https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
#%%

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

# Returns shape (num_training_examples_desired, audio_samples_per_training_example)

#%%
def get_raw_dataset_from_mp3s_for_one_speaker(file_prefix,
                                              num_training_examples_desired,
                                              directory,
                                              audio_samples_per_training_example=int(
                                                  22050/2),
                                              expected_sample_rate=22050):
    this_speaker_glob = os.path.abspath(directory + os.sep + file_prefix + "_*")
    list_of_mp3s_for_one_speaker = glob.glob(this_speaker_glob)
    random.shuffle(list_of_mp3s_for_one_speaker)
    assert len(list_of_mp3s_for_one_speaker) > 0, this_speaker_glob
    speaker_start_time = time.time()
    samples_so_far_for_this_speaker = 0
    network_inputs = np.zeros(
        (num_training_examples_desired, audio_samples_per_training_example))
    for mp3 in list_of_mp3s_for_one_speaker:
        print("loading ", mp3)
        audio_time_series, sampling_rate = librosa.core.load(mp3, sr=None)
        assert sampling_rate == expected_sample_rate, (
            "frame_rate was %d, not %d" % (sampling_rate, expected_sample_rate))
        assert len(
            audio_time_series.shape) == 1, "It wasn't mono: %s" % audio_time_series.shape
        num_audio_samples_we_use = audio_samples_per_training_example * \
            int(audio_time_series.shape[0] /
                audio_samples_per_training_example)
        training_samples_from_this_file = audio_time_series[:num_audio_samples_we_use].reshape(
            (-1, audio_samples_per_training_example))
        number_training_samples_we_have = training_samples_from_this_file.shape[0]
        number_training_samples_we_need = num_training_examples_desired - \
            samples_so_far_for_this_speaker
        if number_training_samples_we_have > number_training_samples_we_need:
            training_samples_from_this_file = training_samples_from_this_file[
                :number_training_samples_we_need, :]
        network_inputs[samples_so_far_for_this_speaker:samples_so_far_for_this_speaker +
                       training_samples_from_this_file.shape[0], :] = training_samples_from_this_file
        samples_so_far_for_this_speaker += training_samples_from_this_file.shape[0]
        if samples_so_far_for_this_speaker >= num_training_examples_desired:
            assert samples_so_far_for_this_speaker == num_training_examples_desired, samples_so_far_for_this_speaker
            print("finished book %s, which took %d seconds" %
                  (file_prefix, (time.time() - speaker_start_time)))
            return network_inputs
    assert 0 == 1, "We ran out of mp3s for %s %d %d. Ended up with %d" % (
        this_speaker_glob, audio_samples_per_training_example, num_training_examples_desired, samples_so_far_for_this_speaker)

# returns tuple of training_data and labels
def generate_raw_dataset_from_mp3s_in_parallel(num_speakers, minutes_per_speaker,
                                               directory):
    from multiprocessing import Pool
    pool = Pool()
    # Note to self: partial fills in arguments starting from the left.
    print("starting %d processes" % pool._processes)
    results = []
    examples_per_speaker = 60*2*minutes_per_speaker
    try:
        results = pool.starmap(get_raw_dataset_from_mp3s_for_one_speaker, zip(
            top_20[:num_speakers],
            num_speakers*[examples_per_speaker],
            num_speakers*[directory]))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        sys.exit()
    pool.close()
    training_set = np.zeros(
        (num_speakers * examples_per_speaker, int(22050/2)))
    labels = np.full((num_speakers * examples_per_speaker, num_speakers), -1)
    for speaker_index, results in enumerate(results):
        print("concatenating results for speaker", speaker_index)
        start_index = speaker_index * examples_per_speaker
        end_index = start_index + examples_per_speaker
        training_set[start_index:end_index, :] = results
        labels[start_index:end_index, :] = keras.utils.to_categorical(speaker_index,
              num_classes=num_speakers)
    return (training_set, labels)

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
    # Save these to disk in case we want to alter the confusion matrix aesthetics.
    np.save("X_dev", X_dev)
    np.save("y_dev", y_dev)


if __name__ == '__main__':

    training_set, labels = generate_raw_dataset_from_mp3s_in_parallel(num_speakers=20,
                                                                      minutes_per_speaker=60,
                                                                      directory="noisy_top_20/")
    print(training_set.shape)
    print(training_set)
    training_set = np.expand_dims(training_set, axis=-1)
    print(training_set.shape)
    print(training_set)
    print(labels.shape)
    print(labels)

    saved_model_name = "monster_model.h5"
    print("loading model from", saved_model_name)
    model = keras.models.load_model(saved_model_name)
    # Training the model will save these files.
#    X_dev = np.load("X_dev.npy")
    #y_dev = np.load("y_dev.npy")
    generate_confusion_matrix(model, training_set, labels)
