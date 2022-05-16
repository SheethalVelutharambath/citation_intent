import sklearn.metrics as metrics

labels_index = {'background': 0, 'method': 1, 'result': 2}


def get_accuracy(pred, true):
    count = 0
    for i in range(len(pred)):
        if pred[i] == true[i]:
            count += 1
    return count / len(pred)


def evaluate(true, pred):
    assert len(true) == len(pred)
    precision_dict = {}
    recall_dict = {}
    fscore_dict = {}
    for label in labels_index.values():
        tp, tn, fn, fp = 0, 0, 0, 0
        for x, y in zip(true, pred):
            if x == label and y == label:
                tp += 1
            elif x != label and y != label:
                tn += 1
            elif x == label and y != label:
                fn += 1
            elif x != label and y == label:
                fp += 1
        precision = tp / (tp + fp + 0.0000000000001)
        recall = tp / (tp + fn + 0.0000000000001)
        fscore = 2 * precision * recall / (precision + recall + 0.0000000000001)

        precision_dict[label] = round(precision, 3)
        recall_dict[label] = round(recall, 3)
        fscore_dict[label] = round(fscore, 3)
    acc = get_accuracy(pred, true)
    print(f'Precision values = {precision_dict}')
    print(f'Recall values = {recall_dict}')
    print(f'Fscore values = {fscore_dict}')
    print(f'Accuracy = {acc}')
    return precision_dict, recall_dict, fscore_dict,acc
