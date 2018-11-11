#使用pyspark的ml模块来构建分类器，实现虎扑用户的性别分类。
#用关联规则分析用户混迹的板块
import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
path_src = path.split('src')[0]
sys.path.append(path_src + "src")#项目的根目录
import time
from pymongo import MongoClient
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession,SQLContext
import numpy as np
import copy
from analysis.config import environment
import pandas as pd
from pyspark.ml.linalg import SparseVector, DenseVector,DenseMatrix, Vectors
from pyspark.mllib.regression import LabeledPoint


userStasticsCollection = "completeUserFeatureSample"
MONGO_IP, MONGO_PORT ,dbname, username, password = environment.MONGO_IP, environment.MONGO_PORT, \
                                                          environment.MONGO_DB_NAME, None, None
conn = MongoClient(MONGO_IP, MONGO_PORT)
db = conn[dbname]
# db.authenticate(username, password)
collection = db[userStasticsCollection]

def getWordFreqDocumentFreq(userWordFreqMapList, jobName = 'wordFreqDocumentFreq'):
    userWordFreqMapList = copy.deepcopy(userWordFreqMapList)
    wordFreqMap, ducumentFreqMap = {}, {}
    for userWordFreqMap in userWordFreqMapList:
        if 'gender' in userWordFreqMap:
            del userWordFreqMap['gender']
        for word in userWordFreqMap:
            if word in wordFreqMap:
                wordFreqMap[word] += userWordFreqMap[word]#把该用户的这个词语的频数累加上
                ducumentFreqMap[word] += 1#这个用户使用了这个词语，文档频率加一
            else:
                wordFreqMap[word] = 1
                ducumentFreqMap[word] = 1
    print(jobName, "原始词汇表的大小是", len(wordFreqMap))
    # 返回频率和文档频率都比较高的gram，作为较优特征
    noneWords1 = set(sorted(wordFreqMap.items(), key= lambda x: x[1],reverse=True)[-1000:])#低频词
    noneWords2 = set(sorted(ducumentFreqMap.items(), key= lambda x: x[1],reverse=True)[:1000])#文档频率低的词语
    TFIDFMAP = {}
    numDoc = len(userWordFreqMapList)
    for word in wordFreqMap:
        if word in noneWords1 or word in noneWords2:
            continue
        tf = wordFreqMap.get(word, 0)
        idf = numDoc/(ducumentFreqMap.get(word, 1))
        TFIDFMAP[word] = np.log(tf*idf)
    res = sorted(TFIDFMAP.items(), key= lambda x: x[1],reverse=True)[1000:-10000]
    res = set(map(lambda x: x[0], res))
    return res

def ngramFreatures(sampleSize, featureName="", one_hot=True):
    data = collection.find({}, {'_id':1, "gender": 1, featureName: 1})#从mongo中查询这个特征以及对应的性别标签
    dataList = []
    print("正在读取数据")
    count = 0
    vocab = {}
    wordCount = 0
    for line in data:
        print(line)
        if type(line[featureName][0])==list:
            tempData = {}
            for word in line[featureName]:
                tempData[word[0]] = word[1]
            line[featureName] = tempData
        line[featureName].update({'gender': line['gender']})
        dataList.append(line[featureName])
        for word in line[featureName]:
            if word not in vocab:
                vocab[word] = wordCount
                wordCount += 1
        count += 1
        if count == sampleSize:
            break
        print("读取数据的进度是", count, "/", sampleSize)
    vocabSize = len(vocab)
    # pickle.dump(dataList, open('data.pkl','wb'))
    # dataList = pickle.load(open('data.pkl','rb'))
    print(dataList[0])
    #首先对所有的gram进行一个简单筛选，把普及率低于一定阈值(几乎所有人都不用的),总的使用次数小于一定阈值(大家都用过，然而昙花一现的)
    print("正在初步筛选特征。")
    betterFeatureSet = getWordFreqDocumentFreq(dataList, jobName=featureName)
    betterFeatureSet.add("gender")
    #从通过初筛的所有gram中挑选使用率最高的10000个，进入下一步
    import time
    print("删除不优质的特征")
    for sample in dataList:
        for key in list(sample.keys()):
            if key not in betterFeatureSet:
                del sample[key]#删除不是优质特征的条目
    #dataList里现在是稀疏向量
    #将数据整理为spark可接受的形式
    import random
    random.shuffle(dataList)
    res = []
    for sample in dataList:
        gender = sample['gender']
        sample.pop('gender')
        if len(sample)<20:
            continue
        sparseV = []
        for key, value in sample.items():
            if one_hot==True:
                sparseV.append((vocab[key], 1))
            else:
                sparseV.append((vocab[key], value))
        # print(gender, sparseV)
        sparseV = sorted(sparseV, key=lambda x:x[0])
        res.append([gender, SparseVector(vocabSize, sparseV)])
    trainData, testData = res[:int(len(res)*0.8)], res[int(len(res)*0.8):]
    return trainData, testData, vocabSize

def showConfusionMatrix(predLabel, realLabel, classNum):
    print("真实标签", realLabel[:40])
    print("预测标签", predLabel[:40])
    confusionMatrix = np.zeros((classNum, classNum))
    for i in range(len(predLabel)):
        n = int(predLabel[i])
        m = int(realLabel[i])
        confusionMatrix[m, n] += 1
    print("展示混淆矩阵：")
    print("     预测标签")
    print(" ", end='\t')
    for line in list(range(1, classNum + 1)):
        print(str(line), end='\t')
    print()
    for i in range(classNum):
        print(str(i + 1), end='\t')
        for v in list(confusionMatrix[i, :]):
            print(v, end='\t')
        print()
    recallList = []
    precision = []
    confusionMatrix = np.array(confusionMatrix)
    sampleNumEachClass = np.sum(confusionMatrix, axis=1)
    sampleNumEachPredClass = np.sum(confusionMatrix, axis=0)
    for i in range(len(sampleNumEachClass)):
        recallList.append(int(100 * confusionMatrix[i, i] / (sampleNumEachClass[i] + 0.0000001)))
        precision.append(int(100 * confusionMatrix[i, i] / (sampleNumEachPredClass[i] + 0.0000001)))
    print("召回率是", recallList)
    print("精度是", precision)
    return {"recal": recallList, 'precision': precision}

from matplotlib import pyplot as plt
def plotDimensionWithF1Score(evalutionResult):
    def calFaScore(dataMap):
        recal, precision = dataMap['recal'][1], dataMap['precision'][1]
        return 2*recal*precision/(precision + recal + 0.00000000001)
    pList = [None for i in range(len(evalutionResult)*2)]
    labelList = []
    count = 0
    dimensionList = []
    for clfName,res in evalutionResult.items():
        f1soreListTraningSet = []
        f1soreListTestingSet = []
        dimensionList = []
        for key, value in sorted(res.items(), key=lambda x: x[0]):
            dimensionList.append(key)
            f1soreListTraningSet.append(calFaScore(value['trainingSet']))
            f1soreListTestingSet.append(calFaScore(value['testSet']))
        ax = plt.subplot(1,1,1)
        pList[count], = ax.plot(f1soreListTraningSet, marker='.')
        pList[count+1], = ax.plot(f1soreListTestingSet, marker='.')
        count += 2
        labelList += [clfName+" training set", clfName + ' testing set']
    plt.legend(handles = pList, labels = labelList)
    plt.xlabel("dimension size")
    plt.xticks(range(len(dimensionList)),dimensionList, rotation=45)
    plt.ylabel("F1 Score")
    plt.show()


# from pyspark.mllib.feature import ChiSqSelector
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.classification import LogisticRegression, \
    NaiveBayes, RandomForestClassifier, MultilayerPerceptronClassifier
if __name__ == '__main__':
    trainData, testData, vocabSize = ngramFreatures(-1, featureName="wordFreq")
