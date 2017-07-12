import pytest
import os
from main.data_preprocessing import create_data, load_data
from main.keras_models import FourConv, SixConv
from main.monitoring import ModelMonitor
from main.constants import PROJECT_PATH

TEST_VAR_PATH = os.path.join(PROJECT_PATH, 'tests', 'var')
TEST_DATA_PATH = os.path.join(TEST_VAR_PATH, 'data')
TEST_FIGURE_PATH = os.path.join(TEST_VAR_PATH, 'figures')
TEST_MODEL_PATH = os.path.join(TEST_VAR_PATH, 'models')
for path in [TEST_VAR_PATH, TEST_DATA_PATH, TEST_FIGURE_PATH, TEST_MODEL_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)


@pytest.mark.functional
def test_create_data():
    _ = create_data(n_classes=3, n_samples_per_class=20, test_size=0.4, seed=1, image_shape=(32, 32),
                    data_id='small', data_path=TEST_DATA_PATH, save=True)


@pytest.fixture()
def data():
    return load_data(data_id='small', data_path=TEST_DATA_PATH)


@pytest.mark.functional
@pytest.mark.parametrize('model', [
    FourConv(input_shape=(32, 32, 3), n_classes=3, model_id='four_a',
             model_path=TEST_MODEL_PATH),
    SixConv(input_shape=(32, 32, 3), n_classes=3, model_id='six_a', model_path=TEST_MODEL_PATH),
])
def test_training_and_prediction(data, model):
    history = model.train(data=data, batch_size=10, load_weights=True, save_history=True)
    y_test_hat = model.predict(data.x_test)


@pytest.mark.functional
@pytest.mark.parametrize('model', [
    FourConv(input_shape=(32, 32, 3), n_classes=3, model_id='four_a',
             model_path=TEST_MODEL_PATH),
    SixConv(input_shape=(32, 32, 3), n_classes=3, model_id='six_a', model_path=TEST_MODEL_PATH),
])
def test_monitoring(data, model):
    monitor = ModelMonitor(data=data, model=model)
    monitor.print_classification_report()
    _ = monitor.plot_confusion_matrix()
    _ = monitor.plot_history()
    _ = monitor.plot_test_image_with_prediction()
