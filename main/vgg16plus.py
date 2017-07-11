import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from main.constants import DATA_PATH, MODEL_PATH
from main.data_preprocessing import load_data



def save_bottleneck_features(data_id='medium', data_path=DATA_PATH):

    data = load_data(data_id=data_id)

    model = applications.VGG16(
        include_top=False, weights='imagenet', input_shape=data.x_train.shape[1:], classes=None, pooling=None
    )

    features_train = model.predict(data.x_train, batch_size=32, verbose=10)
    with open(os.path.join(data_path, '{}_vgg16_features_train.pkl'.format(data_id)), "wb") as handle:
        pickle.dump(features_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    features_test = model.predict(data.x_test, batch_size=32, verbose=10)
    with open(os.path.join(data_path, '{}_vgg16_features_test.pkl'.format(data_id)), "wb") as handle:
        pickle.dump(features_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(features_train.shape)
    print(features_test.shape)

    # generator = datagen.flow_from_directory(
    #     train_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=False)
    # bottleneck_features_train = model.predict_generator(
    #     generator, nb_train_samples // batch_size)
    # np.save(open('bottleneck_features_train.npy', 'w'),
    #         bottleneck_features_train)
    #
    # generator = datagen.flow_from_directory(
    #     validation_data_dir,
    #     target_size=(img_width, img_height),
    #     batch_size=batch_size,
    #     class_mode=None,
    #     shuffle=False)
    # bottleneck_features_validation = model.predict_generator(
    #     generator, nb_validation_samples // batch_size)
    # np.save(open('bottleneck_features_validation.npy', 'w'),
    #         bottleneck_features_validation)


def plot_pca(data_id='medium', data_path=DATA_PATH):

    data = load_data(data_id=data_id)

    # model = applications.VGG16(
    #     include_top=False, weights='imagenet', input_shape=data.x_train.shape[1:], classes=None, pooling=None
    # )

    with open(os.path.join(data_path, '{}_vgg16_features_train.pkl'.format(data_id)), 'rb') as f:
        features_train = pickle.load(f)

    with open(os.path.join(data_path, '{}_vgg16_features_test.pkl'.format(data_id)), 'rb') as f:
        features_test = pickle.load(f)

    from sklearn.decomposition import PCA

    fig, ax = plt.subplots(figsize=(8, 8))

    pca = PCA(n_components=2)
    x_transform = pca.fit_transform(features_train[:, 0, 0, :])
    ax.scatter(x_transform[:, 0], x_transform[:, 1], c=data.labels_num_train)
    plt.show()

    print(pca.explained_variance_ratio_)


def train_top(data_id='big', data_path=DATA_PATH, model_path=MODEL_PATH):

    data = load_data(data_id=data_id)

    with open(os.path.join(data_path, '{}_vgg16_features_train.pkl'.format(data_id)), 'rb') as f:
        features_train = pickle.load(f)

    with open(os.path.join(data_path, '{}_vgg16_features_test.pkl'.format(data_id)), 'rb') as f:
        features_test = pickle.load(f)

    model = Sequential()
    model.add(Flatten(input_shape=features_train.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(features_train, data.y_train,
              epochs=10,
              batch_size=32,
              validation_data=(features_test, data.y_test))

    print(history)

    model.save_weights(os.path.join(MODEL_PATH, 'some_exp.hdf5'))


if __name__ == '__main__':
    # medium shapes
    # (3200, 2, 2, 512)
    # (800, 2, 2, 512)
    # big
    # (8000, 2, 2, 512)
    # (2000, 2, 2, 512)
    # save_bottleneck_features(data_id='big')
    # plot_pca(data_id='medium')
    train_top(data_id='medium')