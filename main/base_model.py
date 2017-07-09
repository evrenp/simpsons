import os
import pickle
import logging
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from main.constants import MODEL_PATH
from keras.models import Model
from keras.optimizers import Optimizer


LOGGER = logging.getLogger(__name__)


class BaseModel(object):

    def __init__(
            self,
            model_id='default',
            input_shape=(64, 64, 3),
            n_classes=3,
            model_path=MODEL_PATH,
    ):
        self.model_id = model_id
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.model_path = model_path

    def _get_keras_model_and_optimizer(self):
        return Model(), Optimizer()

    @staticmethod
    def lr_schedule(epoch):
        lr = 0.01
        return lr * (0.1 ** int(epoch / 10))

    def train(
            self,
            data,
            batch_size=32,
            steps_per_epoch=3,
            n_epochs=2,
            load_weights=False,
            save_history=False,
    ):

        assert self.input_shape == data.x_train.shape[1:]
        assert self.n_classes == data.n_classes
        assert batch_size < 0.5 * data.x_train.shape[0]

        model, optimizer = self._get_keras_model_and_optimizer()
        if load_weights:
            model_weights_path = os.path.join(self.model_path, 'weights_{}.hdf5'.format(self.model_id))
            if os.path.exists(model_weights_path):
                model.load_weights(model_weights_path)
                print('Loading trained weights from file {}'.format(model_weights_path))
            else:
                print('File {} with trained weights does not exist'.format(model_weights_path))
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

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
        model_weights_path = os.path.join(self.model_path, 'weights_{}.hdf5'.format(self.model_id))
        checkpoint = ModelCheckpoint(model_weights_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [LearningRateScheduler(self.lr_schedule), checkpoint]
        history = model.fit_generator(data_generator.flow(data.x_train, data.y_train,
                                                          batch_size=batch_size),
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=n_epochs,
                                      validation_data=(data.x_test, data.y_test),
                                      callbacks=callbacks_list)

        if save_history:
            history_dict = {key: getattr(history, key) for key in ['history', 'epoch', 'params']}
            history_file_path = os.path.join(self.model_path, 'history_{}.pkl'.format(self.model_id))
            with open(history_file_path, "wb") as handle:
                pickle.dump(history_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Saving history to file {}'.format(history_file_path))

        return history

    def predict(
            self,
            x,
            batch_size=32,
            verbose=0,
    ):
        assert self.input_shape == x.shape[1:]

        model, optimizer = self._get_keras_model_and_optimizer()
        model_weights_path = os.path.join(self.model_path, 'weights_{}.hdf5'.format(self.model_id))
        if os.path.exists(model_weights_path):
            model.load_weights(model_weights_path)
            print('Loading trained weights from file {}'.format(model_weights_path))
        else:
            raise ValueError('File {} with trained weights does not exist'.format(model_weights_path))
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        y_hat = model.predict(x=x, batch_size=batch_size, verbose=verbose)
        assert y_hat.shape[1] == self.n_classes
        return y_hat
