import sys
from lstm_train import train
from lstm_predict import predict

mode="train"
data_path="../../data/train.tsv"
test_path="../../data/train.tsv"

try:
    mode = sys.argv[1]
    data_path = sys.argv[2]
    test_path = sys.argv[3]
except:
    pass

if(mode=="train"):
    train, test = train(data_path, test_path)
elif(mode=="predict"):
    predict(data_path)
print(train["text1"])
