# Context-aware Additive Regularization Topic Modelling

The model capable of building topic vectors of words taking into account the context of the word.

## Installation

To install the package:

0. Install `git` and `poetry` using your package manager.

1. Clone the repository

```
git clone https://github.com/revit3d/topic-modelling-attention
```

2. Run installation

```
cd topic-modelling-attention
poetry install
```

3. Downloading additional data

This module uses nltk for preprocessing. nltk requires additional data to work.
You can install it by running the following script:
```
poetry run download-nltk-data
```
This command will download nltk data to `~/nltk_data` by default.
You can change the download path by setting `NLTK_DATA` environment variable.
Alternatively, you can download `punkt_tab` and `stopwords` nltk resources by yourself.
