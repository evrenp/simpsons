import click
from main.train import train_model
from main.data_preprocessing import DataSet

CONFIGS = {
    'small': {
        'data_id': 'small',
        'model_id': 'six_conv',
        'load_weights': True,
        'batch_size': 10,
        'steps_per_epoch': 3,
        'n_epochs': 20,
        'save_history': True,
    }
}


@click.command()
@click.argument('config_id', default='small')
def main(config_id):
    assert config_id in CONFIGS.keys(), '{} not in available configs'.format(config_id)
    train_model(**CONFIGS[config_id])


if __name__ == '__main__':
    main()
