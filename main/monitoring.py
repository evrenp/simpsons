import os
import seaborn
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import pandas as pd
import pickle

from main.constants import NUM_2_CHARACTER, DATA_PATH


class ModelMonitor(object):
    def __init__(
            self,
            data,
            model,
    ):
        self.data = data
        self.model = model

        # final prediction on test data
        self.labels_num_test_hat = self.data.label_binarizer.inverse_transform(self.model.predict(self.data.x_test))

        # get class_labels for plotting
        self.class_labels = [NUM_2_CHARACTER[i] for i in self.data.label_binarizer.classes_]

    def print_classification_report(self, digits=2):
        print(sklearn.metrics.classification_report(y_true=self.data.labels_num_test, y_pred=self.labels_num_test_hat,
                                                    target_names=self.class_labels, digits=digits))

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
        fig.suptitle('data_id={}, model_id={}'.format(self.data.id, self.model.model_id), fontsize=12)
        return fig

    def plot_confusion_matrix(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.subplots_adjust(bottom=0.3, left=0.3)
        ticks = np.arange(len(self.class_labels))
        ax.set(
            xlabel='Actual character',
            ylabel='Predicted character',
            xticks=ticks,
            yticks=ticks,
            xticklabels=self.class_labels,
            yticklabels=self.class_labels,
            title='data_id={}, model_id={}'.format(self.data.id, self.model.model_id),
        )
        plt.xticks(rotation=90)
        cax = ax.imshow(sklearn.metrics.confusion_matrix(self.data.labels_num_test, self.labels_num_test_hat),
                        interpolation='nearest', cmap=plt.cm.coolwarm)
        cbar = fig.colorbar(cax)
        cbar.ax.set_ylabel('Number of events', rotation=90)
        return fig


if __name__ == '__main__':
    from main.keras_models import FourConv, SixConv
    from main.vgg16_model import Vgg16
    from main.data_preprocessing import load_data, DataSet

    # data = load_data(data_id='big', data_path=DATA_PATH)
    data = load_data(data_id='big', data_path=DATA_PATH)

    # model = SixConv(input_shape=data.x_train.shape[1:], n_classes=data.n_classes, model_id='six_conv_001')
    # model = FourConv(input_shape=data.x_train.shape[1:], n_classes=data.n_classes, model_id='four_conv_001')
    model = Vgg16(input_shape=data.x_train.shape[1:], n_classes=data.n_classes, model_id='vgg_001')

    monitor = ModelMonitor(data=data, model=model)

    monitor.print_classification_report()
    _ = monitor.plot_confusion_matrix()
    plt.show()
    _ = monitor.plot_history()
    plt.show()
