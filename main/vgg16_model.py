import numpy as np
import os
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16

from main.base_model import BaseModel


class Vgg16(BaseModel):
    """
    This implements a static VGG16 model plus a trainable top model.
    See also: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    """

    def _get_vgg16_model(self):
        model = VGG16(
            include_top=False, weights='imagenet', input_shape=self.input_shape, classes=None, pooling=None
        )
        return model

    def _get_top_model(self, features_shape):
        model = Sequential()
        model.add(Flatten(input_shape=features_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _train_top_model(self, data, n_epochs, batch_size, save_history):
        with open(os.path.join(self.model_path, '{}_vgg16_features_{}.pkl'.format(data.id, 'train')), 'rb') as f:
            features_train = pickle.load(f)
        with open(os.path.join(self.model_path, '{}_vgg16_features_{}.pkl'.format(data.id, 'test')), 'rb') as f:
            features_test = pickle.load(f)

        model = self._get_top_model(features_shape=features_train.shape[1:])

        history = model.fit(features_train, data.y_train,
                            epochs=n_epochs,
                            batch_size=batch_size,
                            validation_data=(features_test, data.y_test))
        model.save_weights(self.model_weights_path)
        if save_history:
            self._save_history(history=history)

        return history

    def _save_bottleneck_features(self, data, model):
        assert np.min(data.x_train.shape[1:3]) >= 48

        for x, data_type in zip([data.x_train, data.x_test], ['train', 'test']):
            file_path = os.path.join(self.model_path, '{}_vgg16_features_{}.pkl'.format(data.id, data_type))
            if os.path.exists(file_path):
                print('Features {} already exist'.format(file_path))
            else:
                print('Predicting features for {} {}...'.format(data.id, data_type))
                features = model.predict(x, batch_size=32)
                print('Saving features to {}'.format(file_path))
                with open(file_path, "wb") as handle:
                    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train(
            self,
            data,
            batch_size=32,
            steps_per_epoch=3,
            n_epochs=2,
            load_weights=False,
            save_history=False,
    ):
        assert data.id == 'big', 'Currently, vgg only works for big data'
        assert self.input_shape == data.x_train.shape[1:]
        assert self.n_classes == data.n_classes
        assert batch_size < 0.5 * data.x_train.shape[0]

        vgg_model = self._get_vgg16_model()
        self._save_bottleneck_features(data=data, model=vgg_model)
        history = self._train_top_model(data=data, batch_size=batch_size, n_epochs=n_epochs, save_history=save_history)
        return history

    def predict(
            self,
            x,
            batch_size=32,
            verbose=0,
    ):
        assert self.input_shape == x.shape[1:]

        print('Predicting with vgg_model...')
        vgg_model = self._get_vgg16_model()
        features_hat = vgg_model.predict(x, batch_size=batch_size, verbose=verbose)

        print('Predicting with top_model...')
        top_model = self._get_top_model(features_shape=features_hat.shape[1:])
        top_model.load_weights(self.model_weights_path)
        y_hat = top_model.predict(x=features_hat, batch_size=batch_size, verbose=verbose)

        assert y_hat.shape[1] == self.n_classes
        return y_hat


if __name__ == '__main__':
    from main.data_preprocessing import load_data, DataSet
    from main.constants import DATA_PATH

    data = load_data(data_id='big', data_path=DATA_PATH)

    model = Vgg16(input_shape=data.x_train.shape[1:], n_classes=data.n_classes, model_id='vgg_001')

    history = model.train(data=data, batch_size=32, n_epochs=50, save_history=True)

    y_test_hat = model.predict(data.x_test)
    labels_num_test_hat = data.label_binarizer.inverse_transform(y_test_hat)

    print(labels_num_test_hat)
