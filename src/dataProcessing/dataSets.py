import sys, os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
path_src = path.split('src')[0]
sys.path.append(path_src + "src")#项目的根目录
from pyhanlp import HanLP

def readFileAndCountWeibo(fileName, label):
    dataList = []
    labelList = []
    with open(fileName, 'r') as f:
        line = f.readline()
        while line != '':
            line = line.split('kabukabu')[-1]
            line = line.split(' ')
            if len(line) == 1:
                line = f.readline()
                continue
            dataList.append(line)
            labelList.append(label)
            line = f.readline()
    return dataList, labelList

from sklearn.model_selection import train_test_split
def processWeibo():
    negCommentFile = path_src + r'/data/weiboSentiment/negative'
    data1, label1 = readFileAndCountWeibo(negCommentFile, 2)
    posCommentFile = path_src + r'/data/weiboSentiment/positive'
    data2, label2 = readFileAndCountWeibo(posCommentFile, 1)
    data = data1 + data2
    label = label1 + label2
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    return X_train, y_train, X_test, y_test

def readDataList(fileName):
    dataList = []
    labelList = []
    with open(fileName, 'r') as f:
        line = f.readline()
        while line != "":
            labelwords = line.split(" ")
            labelList.append(int(labelwords[0]))
            dataList.append(labelwords[1:])
            line = f.readline()
    return dataList, labelList

def wordCount4All(fileName):
    wordCountMap = {}
    with open(fileName, 'r') as f:
        line = f.readline()
        while line != "":
            words = line.split(" ")[1:]
            for word in words:
                wordCountMap[word] = wordCountMap.get(word, 1) + 1
            line = f.readline()
    wordCountList = sorted(wordCountMap.items(), key=lambda x: x[1], reverse=True)
    wordCountList = list(map(lambda x: x[0] + ' ' + str(x[1]), wordCountList))
    with open('wordlist.txt', 'w') as f:
        f.write("\n".join(wordCountList))
    return list(map(lambda x: x[0], wordCountList)), wordCountMap