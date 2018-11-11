# -*- coding:utf-8 -*-
# import scipy
import matplotlib.pyplot as plt
import numpy as np
import pylab
import sys
import os

path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import random, time

test = True
if test == True:
    trainSize = 2000
else:
    trainSize = -1


def showConfusionMatrix(myres, realLabels, classNum):
    predLabel = myres
    realLabel = realLabels

    confusionMatrix = np.zeros((classNum, classNum))
    for i in range(len(predLabel)):
        n = predLabel[i]
        m = realLabel[i]
        confusionMatrix[m - 1, n - 1] += 1
    #     print("???????")
    #     print("     ????")
    #     print("\t")#, end="\t")
    #     for line in list(range(1, classNum+1))[:13]:
    #         print(str(line) + "\t"),
    #     print()
    #     for i in range(classNum):
    #         print(str(i+1) + "\t"),
    #         for v in list(confusionMatrix[i,:])[:13]:
    #             print(str(v) + "\t"),
    #         print()
    for i in range(len(confusionMatrix)):
        a = list(confusionMatrix[i])
        print(' '.join(map(lambda x: str(int(x)), a)))


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


def trainNB(dataList, labelList, labelDict, wordCountMap):
    labelSet = labelDict
    numClass = len(labelSet)
    wordFreqMap = {}
    sampleNumEachClassMap = {}
    for label in labelList:
        sampleNumEachClassMap[label] = sampleNumEachClassMap.get(label, 0) + 1.0
    print(sampleNumEachClassMap)
    totalNumSample = 0.0
    for key in sampleNumEachClassMap:
        totalNumSample += sampleNumEachClassMap[key]
    classProbMap = {}
    classProbLList = np.zeros(numClass)
    for key in sampleNumEachClassMap:
        classProbMap[key] = np.log(sampleNumEachClassMap[key] / totalNumSample)
    # print(classProbMap)
    for label in labelDict:
        labelIndex = labelSet.index(label)
        classProbLList[labelIndex] = classProbMap[label]
    for i in range(len(dataList)):
        words = dataList[i]
        label = labelList[i]
        labelIndex = labelSet.index(label)
        for word in words:
            if len(word) < 2 or len(word) > 4:
                continue
            if word in wordFreqMap:
                wordFreqMap[word][labelIndex] += 1.0 / sampleNumEachClassMap[label]
            else:
                tempCount = np.zeros(numClass)
                tempCount += 0.0000001
                tempCount[labelIndex] += 1.0 / sampleNumEachClassMap[label]
                wordFreqMap[word] = tempCount
    #             print(word, wordFreqMap[word])
    #     print(wordFreqMap)
    wordsInEachClass = np.zeros(numClass)
    for key in wordFreqMap:
        wordsInEachClass += wordFreqMap[key]

    for key in wordFreqMap:
        wordFreqMap[key] = wordFreqMap[key] / wordsInEachClass
        wordFreqMap[key] = np.log(wordFreqMap[key] / wordCountMap[key])
    #         wordFreqMap[key] /= wordCountMap[key]**1.5
    #     print(wordFreqMap)
    return wordFreqMap, classProbLList


def predict(clf, labelSet, classProbLList, dataList, numClass=0):
    res = []
    aaa = []
    for words in dataList:
        probs = np.ones(numClass)
        for word in words:
            if word not in clf:
                probs += 0  # np.ones(numClass)
            else:
                probs += clf[word]
        probs += classProbLList
        indexClass = list(probs).index(max(probs))
        aaa.append(probs)
        pred = labelSet[indexClass]
        res.append(pred)
    return res, probs


def readFileAndCountWeibo(fileName, label):
    dataList = []
    labelList = []
    with open(fileName, 'r') as f:
        line = f.readline()
        # print(line)
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


import sklearn


def processWeibo():
    negCommentFile = r'negative'
    data1, label1 = readFileAndCountWeibo(negCommentFile, 2)
    posCommentFile = r'positive'
    data2, label2 = readFileAndCountWeibo(posCommentFile, 1)

    data = data1 + data2
    label = label1 + label2
    wordCountMap = {}
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(data, label, test_size=0.2)
    for line in X_train:
        for word in line:
            wordCountMap[word] = wordCountMap.get(word, 1) + 1
    return X_train, y_train, wordCountMap, X_test, y_test


import pickle

if __name__ == '__main__':
    #     goodWords, wordCountMap = wordCount4All('train_data_getWords.txt')
    #     dataList, labelList = readDataList('train_data_getWords.txt')
    #     a = [dataList, labelList, goodWords, wordCountMap]
    #     pickle.dump(a, open('data.pkl', 'wb'))
    #     [dataList, labelList, goodWords, wordCountMap] = pickle.load(open('data.pkl', 'rb'))
    #     testdataList, testlabelList = readDataList('test_data_getWords.txt')#
    dataList, labelList, wordCountMap, X_test, y_test = processWeibo()
    labelDict = list(set(labelList))

    clf, classProbLList = trainNB(dataList, labelList, labelDict, wordCountMap)
    print(len(dataList), classProbLList)
    res, aaa = predict(clf, labelDict, classProbLList, X_test, numClass=len(labelDict))
    print(labelDict)
    print(res[-10:], aaa, labelList[-10:])
    showConfusionMatrix(res, labelList, len(labelDict))