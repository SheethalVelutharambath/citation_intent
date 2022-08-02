# Citation Intent Classification
This project deals with classification of citation intent in scientific publications. We have developed various models for classification. We also have tried to predict the impact of POS tags on citation classification.

## Installation

The deployable codes in `code` directory have their own `requirements.txt` files. Run the following commands from their own respective directories:

```
pip install -r code/lstm/requirements.txt
pip install -r code/lstm_pos/requirements.txt
pip install -r code/perceptron/requirements.txt
pip install -r code/traditional_models/requirements.txt
```

## Directory Structure

```
├── code/
│   ├── lstm/
│   │   ├── lstm_predict.py
│   │   ├── lstm_train.py
│   │   ├── main.py
│   │   ├── README.md
│   │   └── requirements.txt
│   ├── lstm_pos/
│   │   ├── lstm_predict.py
│   │   ├── lstm_train.py
│   │   ├── main.py
│   │   ├── README.md
│   │   └── requirements.txt
│   ├── perceptron/
│   │   ├── evaluation.py
│   │   ├── perceptron.py
│   │   ├── preprocessing.py
│   │   └── requirements.txt
│   └── traditional_models/
│       ├── traditional_model.py
│       ├── traditional_pos_model.py
│       └── requirements.txt
├── data/
│   ├── dev.tsv
│   ├── test.tsv
│   └── train.tsv
└── notebooks/
    ├── bert.ipynb
    ├── BiLSTM Model.ipynb
    ├── BiLSTM POS Model.ipynb
    ├── Traditional Model.ipynb
    └── Traditional POS Model.ipynb
```

The Repository is structured as follows:

- The `code` directory contains python codes of the baseline perceptron classifier, traditional machine learning classifiers and the LSTM, BiLSTM classifiers for classification with and without POS tags.
-  The `notebooks` directory contains jupyter notebook files containing proof of concept of the above deployable codes.
-  The `data` directory contains the raw as well as preprocessed data used in the code implementations.

For the task of intent classification, a main architecture of BiLSTM and GloVe embeddings has been employed and been tested with and without POS tags for where the features were generated of the token_tag type.

Peformance of the models has been evaluated using the Accuracy, Precison, Recall and F1 score metrics.

# Packages and Libraries

The packages and libraries used can be found in the `requirements.txt` files in respective code directories.

# Contribitors

 * [Sheethal Velutharambath](https://github.com/SheethalVelutharambath)
 * [Tejaswi Choppa](https://github.com/choppa98)
