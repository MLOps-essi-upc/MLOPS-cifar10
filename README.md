MLOPS-cifar-10
==============================

Image Classification with cifar10 dataset for MLOps course.



The dataset card is available under this link: [Dataset Card](https://github.com/MLOps-essi-upc/MLOPS-cifar10/blob/master/docs/Dataset_card_Cifar10.md)



The model card is available under this link: [Model Card](https://github.com/MLOps-essi-upc/MLOPS-cifar10/blob/master/docs/Image_Classification_model_card.md)

The cloud provider chosen by the team is Amazon Web Services.



Project Organization
------------

    ├── LICENSE
    ├── Dockerfile         <- The Dockerfile provides instructions for building a Docker image of the project's runtime environment.
    ├── data.dvc           <- Part of the Data Version Control (DVC) system. It serves as a pointer to a specific version of the project's data stored externally.
    ├── params.yaml        <- Contains configuration parameters and settings for the project.
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Contains the model card and dataset card.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks for testing.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented 



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
