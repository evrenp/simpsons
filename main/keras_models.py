import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from main.base_model import BaseModel


class FourConv(BaseModel):
    def _get_keras_model_and_optimizer(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes))
        model.add(Activation('softmax'))
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        return model, opt


class SixConv(BaseModel):
    def _get_keras_model_and_optimizer(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        return model, opt


if __name__ == '__main__':
    from main.data_preprocessing import load_data, DataSet
    from main.constants import DATA_PATH

    data = load_data(data_id='small', data_path=DATA_PATH)

    model = SixConv(input_shape=data.x_train.shape[1:], n_classes=data.n_classes, model_id='six_a')
    # model = FourConv(input_shape=data.x_train.shape[1:], n_classes=data.n_classes, model_id='four_a')

    history = model.train(data=data, batch_size=10, load_weights=True, save_history=True)

    y_test_hat = model.predict(data.x_test)
    labels_num_test_hat = data.label_binarizer.inverse_transform(y_test_hat)

    print(labels_num_test_hat)