# Citation Intent Classification
classify citation intent in scientific publications

This repository has been authored by Tejaswi Choppa and Sheethal Velutharambath based on our models for checking if adding POS tags to the features improves the performance of the models or not.


The Repository is structured as follows:

- The code folder contains python codes of the Baseline PErceptron model, Traditional Machine Classifiers used for the code and the LSTM, BiLSTM codes for classification with and without POS tags
-  The Notebooks folder contains the jupyter notebooks for all the code implementations
-  The data folder contains the datsets as well as other forms of documents used in code implementations
-  Few codes may require softwares outside this environment to run the code 

For the task of intent classification, a main architecture of BiLSTM and GloVe embeddings has been employed and been tested with and without POS tags for where the features were generated of the token_tag type.

Peformance of the models has been evaluated using the Accuracy, Precison, Recall and F1 score metrics.

The Libraries mainly used in this work are:
-Tensorflow
-SkLearn
-nltk

