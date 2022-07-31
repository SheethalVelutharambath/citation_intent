import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
import json
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

def train(train_path, test_path):
    nlp = spacy.load('en_core_web_sm', disable = ["parser", "ner", "textcat", "tagger"])

    train = pd.read_csv(train_path, sep= '\t',names=["id", "explicit", "text", "label"])
    test = pd.read_csv(test_path, sep= '\t', names=["id", "explicit", "text", "label"])

    train.drop(['id', 'explicit'], axis = 1, inplace = True)
    test.drop(['id', 'explicit'], axis = 1, inplace = True)

    label_to_num = {"background":0, "method":1, "result":2}

    def label_decoder(label):
        return label_to_num[label]

    train["label1"] = train["label"].apply(lambda x: label_decoder(x))
    test["label1"] = test["label"].apply(lambda x: label_decoder(x))

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

    train["text1"] = train["text"].apply(lambda x: preprocess(x))
    test["text1"] = test["text"].apply(lambda x: preprocess(x))

    MAX_SEQUENCE_LENGTH = 100

    tokens = []
    for doc in tqdm(nlp.pipe(train["text1"].values)):
        tokens.append(" ".join([n.text for n in doc]))
    train["text1"] = tokens

    tokens = []
    for doc in tqdm(nlp.pipe(test["text1"].values)):
        tokens.append(" ".join([n.text for n in doc]))
    test["text1"] = tokens

    train, val = train_test_split(train, test_size=0.1, random_state=2018)

    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~')

    tokenizer.fit_on_texts(train.text1)


    # saving
    with open('../../data/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary Size :", vocab_size)

    x_train = pad_sequences(tokenizer.texts_to_sequences(train.text1),
                            maxlen = MAX_SEQUENCE_LENGTH)
    x_val = pad_sequences(tokenizer.texts_to_sequences(val.text1),
                            maxlen = MAX_SEQUENCE_LENGTH)
    x_test = pad_sequences(tokenizer.texts_to_sequences(test.text1),
                           maxlen = MAX_SEQUENCE_LENGTH)

    print("Training X Shape:",x_train.shape)
    print("Validation X Shape:",x_val.shape)
    print("Testing X Shape:",x_test.shape)

    y_train = train['label1']
    y_val = val["label1"]
    y_test = test['label1']

    y_train_list = []
    for i in y_train:
        kd = np.zeros(3)
        kd[i]=1
        y_train_list.append(list(kd))

    y_val_list = []
    for i in y_val:
        kd = np.zeros(3)
        kd[i]=1
        y_val_list.append(list(kd))

    y_test_list = []
    for i in y_test:
        kd = np.zeros(3)
        kd[i]=1
        y_test_list.append(list(kd))

    y_train_list = np.array(y_train_list)
    y_val_list = np.array(y_val_list)
    y_test_list = np.array(y_test_list)


    GLOVE_EMB = '../../data/glove.6B.300d.txt'
    EMBEDDING_DIM = 300
    LR = 1e-3
    BATCH_SIZE = 1024
    EPOCHS = 50

    embeddings_index = {}

    f = open(GLOVE_EMB)
    for line in f:
      values = line.split()
      word = value = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' %len(embeddings_index))

    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                              EMBEDDING_DIM,
                                              weights=[embedding_matrix],
                                              input_length=MAX_SEQUENCE_LENGTH,
                                              trainable=False)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_sequences = embedding_layer(sequence_input)
    x = SpatialDropout1D(0.2)(embedding_sequences)
    x = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2))(x)
    outputs = Dense(3, activation='softmax')(x)
    model = tf.keras.Model(sequence_input, outputs)

    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    mc = tf.keras.callbacks.ModelCheckpoint(
        "../../data/model.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        save_freq="epoch",
    )

    model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train_list, batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(x_val, y_val_list), callbacks=[es, mc])

    test_acc = model.evaluate(x_test, y_test_list, verbose=0)
    print("Test Loss:", test_acc[0])
    print("Test Accuracy:", test_acc[1])

    y_pred = model.predict(x_test, verbose=0)
    yhat = np.argmax(y_pred,axis=1)
    y_true = np.argmax(y_test_list,axis = 1)

    print("Test Accuracy: ", accuracy_score(y_true,yhat))
    print("Test Precision: ", precision_score(y_true,yhat, average='macro'))
    print("Test Recall: ", recall_score(y_true,yhat, average='macro'))
    print("Test F1-Score: ", f1_score(y_true,yhat, average='macro'))
