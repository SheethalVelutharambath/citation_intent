import os
from pathlib import Path
import string
import numpy as np
from tqdm import tqdm
from perceptron import Perceptron
from evaluation import *

# nltk.download('stopwords')
# stopwords = nltk.corpus.stopwords.words('english')
data_path = os.path.join(Path(__file__).parent.parent, "data")
from collections import Counter


class Preprocessor:

    def __init__(self):
        train_file_path = data_path + "/tsv/train.tsv"
        self.train_docs, self.label_doc = self.read_file(train_file_path)

        test_file_path = data_path + "/tsv/test.tsv"
        self.test_docs, self.test_label_doc = self.read_file(test_file_path)

        dev_file_path = data_path + "/tsv/dev.tsv"
        self.dev_docs, self.dev_label_doc = self.read_file(dev_file_path)

        stopwords_file_path = data_path + "/stop_words_english.txt"
        self.stopwords_doc = self.read_stopwords(stopwords_file_path)

        self.corpus = None

    def normalize(self, text):
        output = ""
        # convert newline and tab as string with space
        text = text.replace('\\n', ' ').replace('\\t', ' ').replace('et al.', '').replace('-', ' ').replace('(i)',
                                                                                                            ' ').replace(
            '(ii)', ' ').replace('/', ' ')
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

    def remove_stopwords(self, text):
        return [i for i in text if i not in self.stopwords_doc]

    def read_file(self, filepath):
        doc = []
        labels = []
        with open(filepath, encoding='utf8') as file:
            for line in file:
                split_line = line.split("\t")
                doc.append(split_line[2])
                labels.append(split_line[3].replace('\n', ''))
        return doc, labels

    def read_stopwords(self, filepath):
        stopwords = []
        with open(filepath, encoding='utf8') as file:
            for word in file:
                split_word = word.split('\n')
                stopwords.append(split_word[0])
        return stopwords

    def get_features_from_doc(self, docs, original_labels, train=True):
        # get tokenized documents
        normalized_docs = [self.normalize(item) for item in docs]
        tokenized_docs = []
        all_terms = []
        for n_doc in normalized_docs:
            tokens = pr.remove_stopwords(n_doc.split())
            all_terms.extend(tokens)
            tokenized_docs.append(tokens)

        # get one hot encoded labels
        one_hotted_labels = []
        for i in original_labels:
            idx = labels_index[i]
            on_ht = [0] * 3
            for j in range(3):
                if j == idx:
                    on_ht[j] = 1
            one_hotted_labels.append(on_ht)

        term_dict = []
        for i in normalized_docs:
            term_dict.append((i, len(i.split())))

        if train:
            freq = {}  # frequency of each word in all tweets
            for s in tokenized_docs:
                tokens = s
                for t in tokens:
                    if t not in freq.keys():
                        freq[t] = 1
                    else:
                        freq[t] += 1
            self.corpus = sorted(freq.keys())
            self.corpus = {k: v for k, v in freq.items() if v > 0}

        tf_values = []
        for tokens in tokenized_docs:
            frequency = {}
            for t in tokens:
                if t in self.corpus:
                    if t not in frequency.keys():
                        frequency[t] = 1
                    else:
                        frequency[t] += 1
            tf_values.append({k: frequency[k] / len(tokens) for k in frequency.keys()})

        idf_values = {}  # calculating the idf values for each word in all tweets
        for i in tqdm(self.corpus):
            doc_containing_word = 0
            for document in normalized_docs:
                if i in document:
                    doc_containing_word += 1
                idf_values[i] = np.log(len(normalized_docs) / (1 + doc_containing_word))

        tfidf_values = []
        for tf in tf_values:
            tf_idf = {}
            for key in tf.keys():
                tf_idf[key] = tf[key] * idf_values[key]
            tfidf_values.append(tf_idf)

        tfidf_model = []
        for i, j in tqdm(zip(tfidf_values, normalized_docs)):
            tfidf_list = []
            for word in sorted(self.corpus):
                if word in i.keys():
                    tfidf_list.append(i[word])
                else:
                    tfidf_list.append(0)
            tfidf_model.append((j, np.asarray(tfidf_list)))

        tf_idf_encoding = []
        for i in tfidf_model:
            tf_idf_encoding.append(i[1])
        return tf_idf_encoding, one_hotted_labels


if __name__ == "__main__":
    pr = Preprocessor()
    tf_idf_encoding_same, one_hotted_labels_same = pr.get_features_from_doc(pr.train_docs, pr.label_doc)
    percep = Perceptron(lr=10, itrs=2000).fit_onevsall(np.array(tf_idf_encoding_same), np.array(one_hotted_labels_same))

    ## evaluate on test dataset
    print("evaluating on test data file")
    test_tf_idf_encoding, _ = pr.get_features_from_doc(pr.test_docs, pr.test_label_doc, train=False)
    test_y_pred = percep.predict_ovr(np.array(test_tf_idf_encoding))
    test_gold_labels= [labels_index[v] for v in pr.test_label_doc]
    print(evaluate(test_gold_labels, test_y_pred))

    ## evaluate on dev dataset
    print("evaluating on dev data file")
    dev_tf_idf_encoding, _ = pr.get_features_from_doc(pr.dev_docs, pr.dev_label_doc, train=False)
    dev_y_pred = percep.predict_ovr(np.array(dev_tf_idf_encoding))
    dev_gold_labels= [labels_index[v] for v in pr.dev_label_doc]
    print(evaluate(dev_gold_labels, dev_y_pred))