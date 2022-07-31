import numpy as np
import pandas as pd
import gensim
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

train = pd.read_csv("/content/drive/MyDrive/daaaataaa_citae/tsv/train.tsv", sep= '\t',names=["id", "explicit", "text", "label"])
dev = pd.read_csv("/content/drive/MyDrive/daaaataaa_citae/tsv/dev.tsv", sep= '\t',names=["id", "explicit", "text", "label"])
test = pd.read_csv("/content/drive/MyDrive/daaaataaa_citae/tsv/test.tsv", sep= '\t', names=["id", "explicit", "text", "label"])

train.drop(train.columns[[0, 1]], axis = 1, inplace = True)

test.drop(test.columns[[0, 1]], axis = 1, inplace = True)

test.head()

import nltk
nltk.download('punkt')

import string
string.punctuation

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
#storing the puntuation free text
train['text_train']= train['text'].apply(lambda x:remove_punctuation(x))
train['labels_train'] = train['label'].replace(['background','method','result'],[0,1,2])
train.head()

#storing the puntuation free text
test['text_test']= test['text'].apply(lambda x:remove_punctuation(x))
test['labels_test'] = test['label'].replace(['background','method','result'],[0,1,2])
test.head()

#converting to lower case
train['text_train']= train['text_train'].apply(lambda x: x.lower())

test['text_test']= test['text_test'].apply(lambda x: x.lower())

test.head()

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

def tokenize(word):
   word = nltk.word_tokenize(word)
   return word

train['tokenized']= train['text_train'].apply(lambda x: tokenize(x))
test['tokenized']= test['text_test'].apply(lambda x: tokenize(x))

def remove_stopwords(texts):
    output= [i for i in texts if i not in stopwords]
    return output

#removing stop words
train['tokenized']= train['tokenized'].apply(lambda x:remove_stopwords(x))
test['tokenized']= test['tokenized'].apply(lambda x:remove_stopwords(x))

from nltk.stem import WordNetLemmatizer
#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

nltk.download('wordnet')
nltk.download('omw-1.4')
def lemmatizer(text):
  lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
  return lemm_text
train['stemmed']=train['tokenized'].apply(lambda x:lemmatizer(x))

test['stemmed']=test['tokenized'].apply(lambda x:lemmatizer(x))

test.head()

x_test = test.text.tolist()

x_text=train.text
x_text[0]

import nltk
nltk.download('averaged_perceptron_tagger')

x_tokenized = train.stemmed.tolist()
test_tokenized =test.stemmed.tolist()

#generating pos tags using nltk
pos_tags = [nltk.pos_tag(token) for token in x_tokenized]

pos_tags_test = [nltk.pos_tag(token) for token in test_tokenized]

#storing train POS tags to the list
tags = []
for i in pos_tags:
  tags.append([j[1] for j in i])

#storing test POS tags to the list
tags_test = []
for i in pos_tags_test:
  tags_test.append([j[1] for j in i])

from gensim.models import Word2Vec

import gensim

w2v_model = Word2Vec(min_count=1,     #word2vec embedding for POS tags
                 window=2,
                 size=30,
                 sample=1e-5,
                 alpha=0.01,
                 min_alpha=0.0007,
                 negative=0,
                 workers=2)

w2v_model.build_vocab(tags, progress_per=1)
w2v_model.train(tags, total_examples=w2v_model.corpus_count, epochs=3, report_delay=1)

# getting the vectors for the Pos_tags from w2v_model
my_dict = dict({})
for index, key in enumerate(w2v_model.wv.vocab):
    my_dict[key] = w2v_model.wv[key]

X_train =tags
y_train = np.array(train['labels_train'].tolist())

X_test =tags_test
y_test = np.array(test['labels_test'].tolist())

#building vectors for words in text from word2vec model
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += w2v_model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

n_dim = 30

#preparing training data
from sklearn.preprocessing import scale
train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in X_train])
train_vecs = scale(train_vecs)

#preparing testing data
from sklearn.preprocessing import scale
test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in X_test])
test_vecs = scale(test_vecs)

print(y_train.shape)
print(train_vecs.shape)

#SGD Classifier
from sklearn.linear_model import SGDClassifier
lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)
y_pred_sgd = lr.predict(test_vecs)
print(classification_report(y_test,y_pred_sgd))

print('accuracy_sgd:', accuracy_score(y_test,y_pred_sgd))
print('recall_sgd:' ,recall_score(y_test,y_pred_sgd, average='macro'))
print('precision_sgd:',precision_score(y_test,y_pred_sgd, average='macro'))
print('f1_score_sgd:',f1_score(y_test,y_pred_sgd, average='macro'))

from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(train_vecs, y_train)
y_pred_percep = clf.predict(test_vecs)
print(classification_report(y_test,y_pred_percep))

print('accuracy_perceptron:', accuracy_score(y_test,y_pred_percep))
print('recall_perceptron:' ,recall_score(y_test,y_pred_percep, average='macro'))
print('precision_perceptron:',precision_score(y_test,y_pred_percep, average='macro'))
print('f1_score_perceptron:',f1_score(y_test,y_pred_percep, average='macro'))

from sklearn.neighbors import KNeighborsClassifier
cl = KNeighborsClassifier(n_neighbors=5)
cl.fit(train_vecs, y_train)
y_pred_kn = cl.predict(test_vecs)
print(classification_report(y_test,y_pred_kn))


print('accuracy_knn:', accuracy_score(y_test,y_pred_kn))
print('recall_knn:' ,recall_score(y_test,y_pred_kn, average='macro'))
print('precision_knn:',precision_score(y_test,y_pred_kn, average='macro'))
print('f1_score_knn:',f1_score(y_test,y_pred_kn, average='macro'))

mod = SVC(kernel = 'linear', C=1)
mod.fit(train_vecs, y_train)
y_preds = clf.predict(test_vecs)

print('accuracy_SVM :',accuracy_score(y_test, y_preds))
print('recall_SVM:',recall_score(y_test, y_preds, average='macro'))
print('precision_SVM:',precision_score(y_test, y_preds, average='macro'))
print('f1score_SVM:',f1_score(y_test, y_preds, average='macro'))
