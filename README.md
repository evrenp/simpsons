# Simpsons Character Recognition
- Character recognition for simpson image data with deep learning
- This repo is a refactored version of https://github.com/alexattia/SimpsonRecognition

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
    python main/data_preprocessing.py
    ```

- Train a specified model with
    ```bash
    python applications/train.py small
    ```

## Tests
- Run functional tests with
    ```bash
    pytest -v -m functional
    ```
