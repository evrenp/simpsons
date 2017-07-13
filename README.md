# Simpsons Character Recognition
- This repo is about character recognition for simpson image data with different deep learning models.
- The simpsons image data can be found on kaggle:  https://www.kaggle.com/alexattia/the-simpsons-characters-dataset
- This code is partially based on https://github.com/alexattia/SimpsonRecognition. The following changes and extensions have been made:
    - Data is always treated as in-memory flexible toy data at custom shape and scale.
    - The `BaseModel` class provides an interface for any type of convolutional network model in the context of simpsons character recognition data. 
    - The `ModelMonitor` class provides an interface for monitoring any `BaseModel` instance.
    - Functional tests under `pytest` provide quick sanitiy checks for development.
    - Installation is improved via requirements and setup scripts.

## Installation
- Download simpson character images from https://www.kaggle.com/alexattia/the-simpsons-characters-dataset, unzip and move to
    `var/data/characters`

- Create conda environment (or extend existing one)
    ```bash
    conda create --name py35 python=3.5 --file requirements.txt
    ```

- Run setup.py for development
    ```bash
    source activate py35
    python setup.py develop
    ```

## How to use
- Create data sets at custom scale and shape with
    ```bash
    python applications/save_datasets.py <data_id>
    ```
    Note that a trained algorithm will then only work for this particular data. The `data_id` argument in `data_preprocessing/create_data` is used to identify data sets.

- Register a model and training procedure in `applications/train.py` and run with
    ```bash
    python applications/train.py <config_key>
    ```

- Results of training and other types of analysis are collected in `notebooks`.
    - Checkout `notebooks/data_overview.ipynb` for a quick overview on how to create data.
    - Checkout `notebooks/training_results.ipynb` for current results on training.
    - Checkout `notebooks/prediction_demo.ipynb` for a brief demo of prediction performance on test data.

## Functional testing
- Run functional tests with
    ```bash
    pytest -v -m functional
    ```
    
## Next development steps
- Run hyper-parameter search on cluster. 
- Refactor four-conv and six-conv models into generalized convolutional model.
- Fine-tune the full VGG16 model.
