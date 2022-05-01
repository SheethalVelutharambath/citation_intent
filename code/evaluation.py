import sklearn.metrics as metrics

target = [1, 2, 3]

def evaluate(true, pred):
    precision_dict = {}
    recall_dict = {}
    fscore_dict = {}
    for label in target:
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
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = 2 * precision * recall / (precision + recall)

        precision_dict[label] = round(precision, 3)
        recall_dict[label] = round(recall, 3)
        fscore_dict[label] = round(fscore, 3)

    return precision_dict, recall_dict, fscore_dict


## to test evaluation
correct = [1, 1, 2, 3, 2, 1, 1, 3, 1, 3]
predicted = [1, 2, 1, 2, 2, 3, 1, 3, 2, 1]
assert len(correct) == len(predicted)
print(metrics.classification_report(correct, predicted))


precision, recall, fscore = evaluate(correct, predicted)
print(f'Precision values = {precision}')
print(f'Recall values = {recall}')
print(f'Fscore values = {fscore}')
