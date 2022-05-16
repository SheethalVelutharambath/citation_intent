import os
from pathlib import Path
import string
import numpy as np
import nltk
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

        stopwords_file_path = data_path + "/stop_words_english.txt"
        self.stopwords_doc = self.read_stopwords(stopwords_file_path)

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

    def get_features_from_doc(self, docs, original_labels):
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

        most_freq = list(dict(Counter(all_terms).most_common(1000)).keys())
        doc_vectors = []
        doc_len = []
        for sentence in tokenized_docs:
            sent_vec = []
            for token in sentence:
                if token in most_freq:
                    idx = most_freq.index(token)
                    sent_vec.append(idx)
            doc_len.append(len(sent_vec))
            doc_vectors.append(sent_vec)
        desired_max_len = 50
        doc_vectors_same = []
        one_hotted_labels_same = []
        gold_labels_same = []
        for item, label, gold_label in zip(doc_vectors, one_hotted_labels, original_labels):
            if len(item) <= desired_max_len:
                for i in range(len(item), 50):
                    item.append(0)
                doc_vectors_same.append(item)
                one_hotted_labels_same.append(label)
                gold_labels_same.append(labels_index[gold_label])
        return doc_vectors_same, one_hotted_labels_same, gold_labels_same

    def get_norm_docs(self, docs):
        docs_norm = []
        for k in docs:
            norm = [(float(i) - 0) / (2000 - 0) for i in k]
            docs_norm.append(norm)
        return docs_norm


if __name__ == "__main__":
    pr = Preprocessor()
    doc_vectors, one_hotted_labels_same, gold_labels = pr.get_features_from_doc(pr.train_docs, pr.label_doc)
    print(Counter(gold_labels))
    doc_vectors_same_norm = pr.get_norm_docs(doc_vectors)

    train_doc_vectors_same_norm = doc_vectors_same_norm[:6500]
    train_one_hotted_labels_same = one_hotted_labels_same[:6500]
    train_gold_labels = gold_labels[:6500]

    test_doc_vectors_same_norm = doc_vectors_same_norm[6500:]
    test_gold_labels = gold_labels[6500:]

    percep = Perceptron(lr=0.5, itrs=100000).fit_onevsall(np.array(train_doc_vectors_same_norm),
                                                          np.array(train_one_hotted_labels_same))

    y_pred = percep.predict_ovr(np.array(test_doc_vectors_same_norm))
    print(evaluate(test_gold_labels, y_pred))

    ## evaluate on test dataset
    print("evaluating on test data file")
    test_doc_vectors, _, test_gold_labels = pr.get_features_from_doc(pr.test_docs, pr.test_label_doc)
    new_test_doc_vectors_same_norm = pr.get_norm_docs(test_doc_vectors)
    test_y_pred = percep.predict_ovr(np.array(new_test_doc_vectors_same_norm))
    print(evaluate(test_gold_labels, test_y_pred))


