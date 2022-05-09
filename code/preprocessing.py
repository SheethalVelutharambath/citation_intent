import os
from pathlib import Path
import string
import numpy as np
import nltk

# nltk.download('stopwords')
# stopwords = nltk.corpus.stopwords.words('english')
data_path = os.path.join(Path(__file__).parent.parent, "data")



def normalize(text):
    output = ""
    # convert newline and tab as string with space
    text = text.replace('\\n', ' ').replace('\\t', ' ').replace('et al.', '')
    for char in text.lower():
        if char in string.punctuation:
            continue
        elif char.isdigit():
            continue
        try:
            char.encode("ascii")
            output += char
        except UnicodeEncodeError:
            output += ''
    return output


def remove_stopwords(text):
    return [i for i in text if i not in stopwords_doc]


def read_file(filepath):
    doc = []
    with open(filepath, encoding='utf8') as file:
        for line in file:
            split_line = line.split("\t")
            doc.append(split_line[2])
    return doc


def read_stopwords(filepath):
    stopwords = []
    with open(filepath, encoding='utf8') as file:
        for word in file:
            split_word = word.split('\n')
            stopwords.append(split_word[0])
    return stopwords


class Perceptron:

    def __init__(self, learning_rate=0.01, iters=1000):
        self.lr = learning_rate
        self.iters = iters
        self.activation_func = self.act_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        boolvectors, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.iters):

            for i, xi in enumerate(X):
                output = np.dot(xi, self.weights) + self.bias
                y_predicted = self.activation_func(output)

                # Perceptron update rule
                update = self.lr * (y[i] - y_predicted)

                self.weights += update * xi
                self.bias += update

    def predict(self, X):
        output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(output)
        return y_predicted

    def act_func(self, x):
        return np.where(x >= 0, 1, 0)


def cross_valid(file, nfolds):
    file_split = list()
    file_copy = list(file)
    fold_size = int(len(file) / nfolds)
    for i in range(nfolds):
        fold = list()
        while len(fold) < fold_size:
            idx = randrange(len(file_copy))
            fold.append(file_copy.pop(index))
        file_split.append(fold)
    return file_split


if __name__ == "__main__":
    train_file_path = data_path + "/tsv/train.tsv"
    train_docs = read_file(train_file_path)
    stopwords_file_path = data_path + "/stop_words_english.txt"
    stopwords_doc = read_stopwords(stopwords_file_path)
    print(normalize(train_docs[0]))
    normalized_docs = [normalize(item) for item in train_docs]
    print(normalized_docs[:1])

    tokenized_docs = []
    all_terms = []
    for n_doc in normalized_docs:
        tokens = remove_stopwords(n_doc.split())
        all_terms.extend(tokens)
        tokenized_docs.append(tokens)

    #getting most 1000 frequent tokens
    from collections import Counter
    most_freq = dict(Counter(all_terms).most_common(1000))
    bool_vectors = []
    for sentence in normalized_docs:
        tok = sentence.split()
        sent_vec = []
        for token in most_freq:
            if token in tok:
                sent_vec.append(1)
            else:
                sent_vec.append(0)
        bool_vectors.append(sent_vec)
    bool_vectors = np.asarray(bool_vectors)
    print(bool_vectors)

