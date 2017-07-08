import numpy as np
import cv2
import os
import logging
import pickle
from sklearn.model_selection import train_test_split
from collections import namedtuple
from sklearn.preprocessing import LabelBinarizer

from main.constants import NUM_2_CHARACTER, DATA_PATH

DataSet = namedtuple('DataSet', [
    'x_train',
    'x_test',
    'y_train',
    'y_test',
    'labels_num_train',
    'labels_num_test',
    'labels_char_train',
    'labels_char_test',
    'n_classes',
    'image_shape',
    'id',
    'label_binarizer'
])
LOGGER = logging.getLogger(__name__)


def create_data(
        character_path=os.path.join(DATA_PATH, 'characters'),
        n_classes=10,
        n_samples_per_class=1000,
        test_size=0.2,
        image_shape=(64, 64),
        seed=0,
        data_id='some_data_id',
):
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
    )
    image_list, character_list, character_num_list = [], [], []
    n_created_classes = 0
    for character_num, character in NUM_2_CHARACTER.items():
        files = os.listdir(os.path.join(character_path, character))
        if len(files) >= n_samples_per_class:
            LOGGER.info('Generating data for {}...'.format(character))
            np.random.seed(seed)
            for file in np.random.choice(files, n_samples_per_class):
                image = cv2.imread(os.path.join(character_path, character, file))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, image_shape)
                image = image.astype('float32') / 255.
                image_list.append(image)
                character_list.append(character)
                character_num_list.append(character_num)
            n_created_classes += 1
        else:
            LOGGER.info('Skipping data for {}...'.format(character))
        if n_created_classes == n_classes:
            break
    if n_created_classes < n_classes:
        LOGGER.info('Could only create {} classes instead of {}'.format(n_created_classes, n_classes))

    x = np.stack(image_list, axis=0)

    label_binarizer = LabelBinarizer()

    y = label_binarizer.fit_transform(character_num_list).astype('float32')

    args = list(train_test_split(x, y, character_num_list, character_list, test_size=test_size, random_state=seed)) + \
           [n_created_classes, image_shape, data_id, label_binarizer]
    return DataSet(*args)


def load_data(data_id='small', data_path=DATA_PATH):
    """Load data

    Args:
        data_id (str): data id
        data_path (os.path.Path):

    Returns:
        DataSet
    """
    with open(os.path.join(data_path, '{}_data.pkl'.format(data_id)), 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    # save small data
    data = create_data(n_classes=3, n_samples_per_class=20, test_size=0.4, seed=1, image_shape=(32, 32), data_id='small')
    with open(os.path.join(DATA_PATH, 'small_data.pkl'), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # save medium data
    # data = create_data(n_classes=10, n_samples_per_class=200, test_size=0.2, data_id='medium')
    # with open(os.path.join(DATA_PATH, 'medium_data.pkl'), "wb") as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # # save big data
    # data = create_data(n_classes=18, n_samples_per_class=1000, test_size=0.2, data_id='big')
    # with open(os.path.join(DATA_PATH, 'big_data.pkl'), "wb") as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


