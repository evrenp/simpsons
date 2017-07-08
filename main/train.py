import numpy as np
import cv2
import os
import gc
import matplotlib.pyplot as plt
import pickle
import h5py
import glob
import time
import logging
from random import shuffle
from collections import Counter
from keras import backend
from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam

from main.constants import MODEL_PATH
from main.data_preprocessing import load_data, DataSet
from main.keras_models import create_model_four_conv, create_six_conv_model

LOGGER = logging.getLogger(__name__)


def lr_schedule(epoch):
    lr = 0.01
    return lr * (0.1 ** int(epoch / 10))


def train(data, model_id='six_conv', load_weights=True, batch_size=20, steps_per_epoch=4, n_epochs=10,
          save_history=False):
    assert batch_size < 0.5 * data.x_train.shape[0], 'batch_size is too large'

    model = create_compiled_model(data=data, model_id=model_id, load_weights=load_weights)

    # data generator
    data_generator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    data_generator.fit(data.x_train)
    model_weights_path = os.path.join(MODEL_PATH, 'weights_{}_{}.hdf5'.format(data.id, model_id))
    checkpoint = ModelCheckpoint(model_weights_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [LearningRateScheduler(lr_schedule), checkpoint]
    with backend.get_session():
        history = model.fit_generator(data_generator.flow(data.x_train, data.y_train,
                                                          batch_size=batch_size),
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=n_epochs,
                                      validation_data=(data.x_test, data.y_test),
                                      callbacks=callbacks_list)

    if save_history:
        history_dict = {key: getattr(history, key) for key in ['history', 'epoch', 'params']}
        with open(os.path.join(MODEL_PATH, 'history_{}_{}.pkl'.format(data.id, model_id)), "wb") as handle:
            pickle.dump(history_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model, history


def create_compiled_model(data, model_id, load_weights=False):

    if model_id == 'six_conv':
        model, opt = create_six_conv_model(input_shape=data.x_train.shape[1:], n_classes=data.n_classes)
    elif model_id == 'four_conv':
        model, opt = create_model_four_conv(input_shape=data.x_train.shape[1:], n_classes=data.n_classes)
    else:
        raise ValueError('unknown model_id {}'.format(model_id))

    # load trained weights
    if load_weights:
        model_weights_path = os.path.join(MODEL_PATH, 'weights_{}_{}.hdf5'.format(data.id, model_id))
        if os.path.exists(model_weights_path):
            model.load_weights(model_weights_path)
        else:
            print('File {} with trained weights does not exist'.format(model_weights_path))

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    model_id = 'six_conv'
    data = load_data(data_id='small')
    batch_size = 10
    steps_per_epoch = int(data.x_train.shape[0] / batch_size)
    model, history = train(data=data, model_id='six_conv', load_weights=True, batch_size=batch_size,
                           steps_per_epoch=steps_per_epoch,
                           n_epochs=20, save_history=True)
