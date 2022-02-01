[WIP] part_counting
==============================

Two approaches are proposed to counting parts in an RGBD image. The goal is to properly estimate the amount of parts, all of the same geometry, in a steel box. All images were generated through renders of simulations, using blender.

One approach is to use computer vision "traditional" techniques to achieve the estimate. Registration algorithms are used to "dig" the part's geometry from the surface measured (depth channel).

The other is to use straightforward deep learning (convolutional encoder with fully-connected decoder). The problem is easily framed as a superviserd learning, regression task.

Deep-learning model
------------

A baseline model was trained, both to have a performance reference and to test the training and inference routines. This model consisted of two convolutional layers (convolution->ReLU->MaxPooling) followed by two fully-connected  with dropout.

Then, a transfer learning + fine-tuning approach based on EfficientNet was implemented. The experiments are tracked in wandb (see [transfer learning](https://wandb.ai/brunompac/part-counting-transfer-learning/) and [fine-tuning](https://wandb.ai/brunompac/part-counting-fine-tuning) experiments).

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
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
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
