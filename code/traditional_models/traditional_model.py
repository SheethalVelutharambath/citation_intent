import numpy as np
import pandas as pd
import gensim
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

train = pd.read_csv("/content/drive/MyDrive/daaaataaa_citae/tsv/train.tsv", sep= '\t',names=["id", "explicit", "text", "label"])
dev = pd.read_csv("/content/drive/MyDrive/daaaataaa_citae/tsv/dev.tsv", sep= '\t')
test = pd.read_csv("/content/drive/MyDrive/daaaataaa_citae/tsv/test.tsv", sep= '\t', names=["id", "explicit", "text", "label"])

train.drop(train.columns[[0, 1]], axis = 1, inplace = True)
test.drop(test.columns[[0, 1]], axis = 1, inplace = True)

train.head()

import nltk
nltk.download('punkt')

import string
string.punctuation

#function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
#storing the puntuation free text
train['text1']= train['text'].apply(lambda x:remove_punctuation(x))
test['text'] = test['text'].apply(lambda x:remove_punctuation(x))
train['labels'] = train['label'].replace(['background','method','result'],[0,1,2])
test['label'] = test['label'].replace(['background','method','result'],[0,1,2])
test.head()

#storing the lower cased text
train['text1']= train['text1'].apply(lambda x: x.lower())
test['text'] = test['text'].apply(lambda x: x.lower())

train.head()

test.head()

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

#function for tokenization
def tokenize(word):
   word = nltk.word_tokenize(word)
   return word

#storing the tokenized text
train['tokenized']= train['text1'].apply(lambda x: tokenize(x))

#storing the tokenized text
test['tokenized']= test['text'].apply(lambda x: tokenize(x))

#function for remove stopwords
def remove_stopwords(texts):
    output= [i for i in texts if i not in stopwords]
    return output

#removed stop words from tokenized text
train['tokenized']= train['tokenized'].apply(lambda x:remove_stopwords(x))

#removed stop words from tokenized text
test['tokenized']= test['tokenized'].apply(lambda x:remove_stopwords(x))

from nltk.stem import WordNetLemmatizer
#defining the object for Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()

#download Wordnet through nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
#function for Lemmatization
def lemmatizer(text):
  lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
  return lemm_text
train['stemmed']=train['tokenized'].apply(lambda x:lemmatizer(x))

test['stemmed']=test['tokenized'].apply(lambda x:lemmatizer(x))

train.head()

x_test = test.text.tolist()

test.head()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer
td = TfidfVectorizer(max_features = 4500)
X = td.fit_transform(train['text1']).toarray()

X

td1 = TfidfVectorizer(max_features = 4500)
X_test = td1.fit_transform(test['text']).toarray()

y = train['labels'].tolist()

Y_test = test['label'].tolist()

#SGDClassifier
from sklearn.linear_model import SGDClassifier
lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(X, y)
preds = lr.predict(X_test)
print(classification_report(Y_test, preds))


print('accuracy score_SGD:', accuracy_score(Y_test, preds))
print('recall score_SGD:',recall_score(Y_test, preds,average = 'macro'))
print('precision score_SGD:',precision_score(Y_test, preds,average = 'macro'))
print('f1 score_SGD:',f1_score(Y_test, preds,average = 'macro'))

#KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
predictions = knn.predict(X_test)


print(classification_report(Y_test, predictions))
print('accuracy score_KNN:', accuracy_score(Y_test, predictions))
print('recall score_KNN:',recall_score(Y_test, predictions,average = 'macro'))
print('precision score_KNN:',precision_score(Y_test, predictions,average = 'macro'))
print('f1 score_KNN:',f1_score(Y_test, predictions,average = 'macro'))

#Perceptron
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X,y)
y_pred = clf.predict(X_test)



print(classification_report(Y_test, y_pred))
print('accuracy score_percep:', accuracy_score(Y_test, y_pred))
print('recall score_percep:',recall_score(Y_test, y_pred,average = 'macro'))
print('precision score_percep:',precision_score(Y_test, y_pred,average = 'macro'))
print('f1 score_percep:',f1_score(Y_test, y_pred,average = 'macro'))

#SVM classifier
mod = SVC(kernel = 'linear', C=1)
mod.fit(X,y)
y_preds = clf.predict(X_test)

print(classification_report(Y_test, y_preds))
print('accuracy score_svm:', accuracy_score(Y_test, y_preds))
print('recall score_svm:',recall_score(Y_test, y_preds,average = 'macro'))
print('precision score_svm:',precision_score(Y_test, y_preds,average = 'macro'))
print('f1 score_svm:',f1_score(Y_test, y_preds,average = 'macro'))
