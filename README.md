# Context-aware Additive Regularization Topic Modelling

The model capable of building topic vectors of words taking into account the context of the word.

## Installation

To install the package:

0. Install `git` and `poetry` using your package manager.

1. Clone the repository

```
git clone https://github.com/revit3d/topic-modelling-attention
cd topic-modelling-attention
```

2. Run installation

2.1 If you don't have an nvidia gpu or if you want a cpu-only install.

```
poetry install --with cpu
```

2.2 If you want cpu+gpu install.

```
poetry install --with gpu
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
