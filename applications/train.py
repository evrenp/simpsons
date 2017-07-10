import os
# suppress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import click
from main.data_preprocessing import load_data, DataSet
from main.constants import DATA_PATH
from main.keras_models import SixConv

CONFIGS = {
    'small': {
        'model': SixConv(input_shape=(32, 32, 3), n_classes=3, model_id='six_a'),
        'train_args': {
            'data': load_data(data_id='small', data_path=DATA_PATH),
            'load_weights': True,
            'batch_size': 10,
            'steps_per_epoch': 3,
            'n_epochs': 2,
            'save_history': True,
        }
    },
    'six_conv_big': {
        'model': SixConv(input_shape=(64, 64, 3), n_classes=10, model_id='six_conv_big'),
        'train_args': {
            'data': load_data(data_id='big', data_path=DATA_PATH),
            'load_weights': False,
            'batch_size': 32,
            'steps_per_epoch': 25,
            'n_epochs': 40,
            'save_history': True,
        }
    },
}


@click.command()
@click.argument('config_key', default='small')
def main(config_key):
    assert config_key in CONFIGS.keys(), '{} not in available configs'.format(config_key)
    config = CONFIGS[config_key]
    _ = config['model'].train(**config['train_args'])


if __name__ == '__main__':
    main()
