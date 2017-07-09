import os
import seaborn
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import pandas as pd
import pickle

from main.constants import NUM_2_CHARACTER, DATA_PATH

from main.data_preprocessing import load_data, DataSet
from main.keras_models import SixConv


class ModelMonitor(object):
    def __init__(
            self,
            data,
            model,
    ):
        self.data = data
        self.model = model

    def print_classification_report(self):
        labels_num_test_hat = self.data.label_binarizer.inverse_transform(self.model.predict(self.data.x_test))
        print(sklearn.metrics.classification_report(y_true=self.data.labels_num_test, y_pred=labels_num_test_hat,
                                                    target_names=list(NUM_2_CHARACTER.values()), digits=4))

    def plot_history(self):
        # load history
        with open(os.path.join(self.model.model_path, 'history_{}.pkl'.format(self.model.model_id)), 'rb') as f:
            history = pickle.load(f)

        df = pd.DataFrame.from_dict(history['history'])
        df.rename(columns={'val_loss': 'validation loss', 'acc': 'accuracy', 'val_acc': 'validation accuracy'},
                  inplace=True)
        df.index.name = 'Epoch'

        # make figure
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        df[['loss', 'validation loss']].plot(ax=axes[0])
        df[['accuracy', 'validation accuracy']].plot(ax=axes[1])
        fig.suptitle(self.model.model_id, fontsize=12)
        return fig

    def plot_confusion_matrix(self):
        labels_num_test_hat = self.data.label_binarizer.inverse_transform(self.model.predict(self.data.x_test))

        # make figure
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(bottom=0.3, left=0.3)
        classes = list(NUM_2_CHARACTER.values())
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
        ax.set_title('data_id={}, model_id={}'.format(self.data.id, self.model.model_id))
        cax = ax.imshow(sklearn.metrics.confusion_matrix(self.data.labels_num_test, labels_num_test_hat),
                        interpolation='nearest', cmap=plt.cm.coolwarm)
        fig.colorbar(cax)
        return fig


if __name__ == '__main__':
    data = load_data(data_id='small', data_path=DATA_PATH)

    model = SixConv(input_shape=data.x_train.shape[1:], n_classes=data.n_classes, model_id='six_a')

    monitor = ModelMonitor(data=data, model=model)

    monitor.print_classification_report()
    _ = monitor.plot_confusion_matrix()
    plt.show()
    _ = monitor.plot_history()
    plt.show()
