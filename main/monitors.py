import os
import seaborn
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import pandas as pd
import pickle

from main.data_preprocessing import load_data, DataSet
from main.train import create_compiled_model
from main.constants import NUM_2_CHARACTER, MODEL_PATH


def print_classification_report(data_id, model_id):
    data = load_data(data_id=data_id)
    model = create_compiled_model(data=data, model_id=model_id, load_weights=True)
    labels_num_test_hat = data.label_binarizer.inverse_transform(model.predict(data.x_test))
    print(sklearn.metrics.classification_report(y_true=data.labels_num_test, y_pred=labels_num_test_hat,
                                                target_names=list(NUM_2_CHARACTER.values()), digits=4))


def plot_history(data_id, model_id):
    # load history
    with open(os.path.join(MODEL_PATH, 'history_{}_{}.pkl'.format(data_id, model_id)), 'rb') as f:
        history = pickle.load(f)

    df = pd.DataFrame.from_dict(history['history'])
    df.rename(columns={'val_loss': 'validation loss', 'acc': 'accuracy', 'val_acc': 'validation accuracy'},
              inplace=True)
    df.index.name = 'Epoch'

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    df[['loss', 'validation loss']].plot(ax=axes[0])
    df[['accuracy', 'validation accuracy']].plot(ax=axes[1])
    fig.suptitle(model_id, fontsize=12)
    return fig


def plot_confusion_matrix(data_id, model_id):
    data = load_data(data_id=data_id)
    model = create_compiled_model(data=data, model_id=model_id, load_weights=True)
    labels_num_test_hat = data.label_binarizer.inverse_transform(model.predict(data.x_test))

    # make figure
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(bottom=0.3, left=0.3)
    classes = list(NUM_2_CHARACTER.values())
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    ax.set_title('data_id={}, model_id={}'.format(data_id, model_id))
    cax = ax.imshow(sklearn.metrics.confusion_matrix(data.labels_num_test, labels_num_test_hat),
                    interpolation='nearest', cmap=plt.cm.coolwarm)
    fig.colorbar(cax)
    return fig


if __name__ == '__main__':
    print_classification_report(data_id='small', model_id='six_conv')

    fig = plot_history(data_id='small', model_id='six_conv')
    plt.show()

    fig = plot_confusion_matrix(data_id='small', model_id='six_conv')
    plt.show()
