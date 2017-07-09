import click

from main.data_preprocessing import create_data

CONFIGS = {
    'small': {
        'n_classes': 3,
        'n_samples_per_class': 20,
        'test_size': 0.4,
        'image_shape': (32, 32),
        'data_id': 'small',
    },
    'medium': {
        'n_classes': 10,
        'n_samples_per_class': 400,
        'test_size': 0.2,
        'image_shape': (64, 64),
        'data_id': 'medium',
    },
    'big': {
        'n_classes': 18,
        'n_samples_per_class': 1000,
        'test_size': 0.2,
        'image_shape': (64, 64),
        'data_id': 'big',
    }
}


@click.command()
@click.argument('keys', nargs=-1)
def main(keys):
    for key in keys:
        _ = create_data(**CONFIGS[key], seed=0, save=True)


if __name__ == '__main__':
    main()
