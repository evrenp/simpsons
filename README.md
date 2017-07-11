# Simpsons Character Recognition
- Character recognition for simpson image data with deep learning
- This repo is a refactored version of https://github.com/alexattia/SimpsonRecognition
- List of refactorings:
    - Data comes as in-memory toy-data at three different scales: small, medium, big. small data is used for development and functional tests.
    - Model class provides interface for defining and running convolutional network models.
    - Monitor class provides interface for defining and running a model monitor. 
    - Functional tests provide quick sanitiy checks for development.
    - Installation and maintainability is improved via requirements and setup.

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

- Results of training and other types of analysis are collected in `notebooks`.

## Tests
- Run functional tests with
    ```bash
    pytest -v -m functional
    ```
