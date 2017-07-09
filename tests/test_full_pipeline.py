import pytest
import os
from main.data_preprocessing import create_data
from main.training import train_model
from main.monitoring import print_classification_report, plot_history, plot_confusion_matrix
from main.constants import PROJECT_PATH

TEST_VAR_PATH = os.path.join(PROJECT_PATH, 'tests', 'var')
TEST_DATA_PATH = os.path.join(TEST_VAR_PATH, 'data')
TEST_FIGURE_PATH = os.path.join(TEST_VAR_PATH, 'figures')
TEST_MODEL_PATH = os.path.join(TEST_VAR_PATH, 'models')
for path in [TEST_VAR_PATH, TEST_DATA_PATH, TEST_FIGURE_PATH, TEST_MODEL_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)


@pytest.mark.functional
def test_full_pipeline():
    data_id = 'small'
    model_id = 'six_conv'
    _ = create_data(n_classes=3, n_samples_per_class=20, test_size=0.4, seed=1, image_shape=(32, 32),
                    data_id=data_id, data_path=TEST_DATA_PATH, save=True)

    model, history = train_model(data_id=data_id, model_id=model_id, load_weights=False, batch_size=10,
                                 steps_per_epoch=2,
                                 n_epochs=2, save_history=True, data_path=TEST_DATA_PATH, model_path=TEST_MODEL_PATH)

    fig = plot_history(data_id=data_id, model_id=model_id, model_path=TEST_MODEL_PATH)
    fig = plot_confusion_matrix(data_id=data_id, model_id=model_id, data_path=TEST_DATA_PATH,
                                model_path=TEST_MODEL_PATH)
    fig = plot_confusion_matrix(data_id=data_id, model_id=model_id, data_path=TEST_DATA_PATH,
                                model_path=TEST_MODEL_PATH)
    print_classification_report(data_id=data_id, model_id=model_id, data_path=TEST_DATA_PATH,
                                model_path=TEST_MODEL_PATH)
