import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras_preprocessing.sequence import pad_sequences
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import re
from keras.preprocessing.text import Tokenizer
import spacy

def predict(predict_path):
    nlp = spacy.load('en_core_web_sm', disable = ["parser", "ner", "textcat", "tagger"])

    predict = pd.read_csv(predict_path, sep= '\t',names=["id", "explicit", "text", "label"])

    predict.drop(['id', 'explicit'], axis = 1, inplace = True)

    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')

    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

    def preprocess(text, stem=False):
        text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
        tokens = []
        for token in text.split():
            if token not in stop_words:
                if stem:
                    tokens.append(stemmer.stem(token))
                else:
                    tokens.append(token)
        return " ".join(tokens)

    predict["text1"] = predict["text"].apply(lambda x: preprocess(x))

    MAX_SEQUENCE_LENGTH = 100

    tokens = []
    for doc in tqdm(nlp.pipe(predict["text1"].values)):
        tokens.append(" ".join([n.text for n in doc]))
    predict["text1"] = tokens

    with open('../../data/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary Size :", vocab_size)

    x_predict = pad_sequences(tokenizer.texts_to_sequences(predict.text1),
                            maxlen = MAX_SEQUENCE_LENGTH)
    
    model = tf.keras.models.load_model('model.h5')


    y_pred = model.predict(x_predict, verbose=0)
    yhat = np.argmax(y_pred,axis=1)

    predict["predicted_label"] = yhat

    num_to_label = {0:"background", 1:"method", 2:"result"}

    def label_decoder(label):
        return num_to_label[label]

    predict["label"] = predict["lapredicted_labelbel"].apply(lambda x: label_decoder(x))

    pd.to_csv("../../data/prediction.csv")