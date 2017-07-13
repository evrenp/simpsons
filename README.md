# Simpsons Character Recognition
- This repo is about character recognition for simpson image data with different deep learning models.
- The simpsons image data can be found on kaggle:  https://www.kaggle.com/alexattia/the-simpsons-characters-dataset
- This code is partially based on https://github.com/alexattia/SimpsonRecognition
- The following changes and extensions have been made:
    - Data is always treated as in-memory flexible toy data at custom shape and scale.
    - The `BaseModel` class provides an interface for any type of convolutional network model in the context of simpsons character recognition data. 
    - The `ModelMonitor` class provides an interface for monitoring any `BaseModel` instance.
    - Functional tests under `pytest` provide quick sanitiy checks for development.
    - Installation is improved via requirements and setup scripts.

## Installation
- Download simpson character images from https://www.kaggle.com/alexattia/the-simpsons-characters-dataset, unzip and move to
    `var/data/characters`

- Create conda environment
    ```bash
    conda create --name py35 python=3.5 --file requirements.txt
    ```

- Run setup.py for development
    ```bash
    source activate py35
    python setup.py develop
    ```

## How to use
- Run data-preprocessing with
    ```bash
    python main/save_datasets.py small medium big
    ```

- Register a model and training procedure in `applications/train.py` and run with
    ```bash
    python applications/train.py <config_key>
    ```

- Results of training and other types of analysis are collected in `notebooks`. Checkout `notebooks/training_results.ipynb`

## Tests
- Run functional tests with
    ```bash
    pytest -v -m functional
    ```
    
## Next development steps
- Run hyper-parameter search on cluster. 
- Refactor four-conv and six-conv models into morphable model. 
- Solve data size inconsitency issue.
