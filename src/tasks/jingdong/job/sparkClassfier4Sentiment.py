#使用pyspark的ml模块来构建分类器，实现虎扑用户的性别分类。
#用关联规则分析用户混迹的板块
import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
import time
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession,SQLContext
import numpy as np
import copy
import pandas as pd
from pyspark.ml.linalg import SparseVector, DenseVector,DenseMatrix, Vectors
from pyspark.mllib.regression import LabeledPoint


def getWordFreqDocumentFreq(userWordFreqMapList, jobName = ''):
    userWordFreqMapList = copy.deepcopy(userWordFreqMapList)
    wordFreqMap, ducumentFreqMap = {}, {}
    for userWordFreqMap in userWordFreqMapList:
        if 'sentiment' in userWordFreqMap:
            del userWordFreqMap['sentiment']
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

import pickle
def ngramFreatures(sampleSize, one_hot=True):
    with open('JingDongCommentsWordFreq.pkl', 'rb') as f:
        data = pickle.load(f)
    dataList = []
    print("正在读取数据")
    count = 0
    vocab = {}
    wordCount = 0
    for line in data:
        tempData = {}
        # if type(line[featureName][0])==list:
        for word in line['wordFreq']:
            tempData[word[0]] = word[1]
            # line[featureName] = tempData
        tempData['sentiment'] = line['sentiment']
        # line[featureName].update({'gender': line['gender']})
        dataList.append(tempData)
        for word in tempData:
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
    # print(dataList[:10])
    #首先对所有的gram进行一个简单筛选，把普及率低于一定阈值(几乎所有人都不用的),总的使用次数小于一定阈值(大家都用过，然而昙花一现的)
    print("正在初步筛选特征。")
    betterFeatureSet = getWordFreqDocumentFreq(dataList, jobName='sentiment')
    betterFeatureSet.add("sentiment")
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
    print(dataList[:10])
    res = []
    for sample in dataList:
        sentiment = sample['sentiment']
        sample.pop('sentiment')
        sparseV = []
        for key, value in sample.items():
            if one_hot==True:
                sparseV.append((vocab[key], 1))
            else:
                sparseV.append((vocab[key], value))
        # print(gender, sparseV)
        sparseV = sorted(sparseV, key=lambda x:x[0])
        res.append([sentiment, SparseVector(vocabSize, sparseV)])
    trainData, testData = res[:int(len(res)*0.8)], res[int(len(res)*0.8):]
    return trainData, testData, vocabSize, betterFeatureSet

def ngramFreatures_weibo(sampleSize, vocabSize, betterFeatureSet, one_hot=True):
    with open('WeiboWordFreq.pkl', 'rb') as f:
        data = pickle.load(f)
    dataList = []
    print("正在读取数据")
    count = 0
    vocab = {}
    wordCount = 0
    for line in data:
        tempData = {}
        # if type(line[featureName][0])==list:
        for word in line['wordFreq']:
            tempData[word[0]] = word[1]
            # line[featureName] = tempData
        tempData['sentiment'] = line['sentiment']
        # line[featureName].update({'gender': line['gender']})
        dataList.append(tempData)
        for word in tempData:
            if word not in vocab:
                vocab[word] = wordCount
                wordCount += 1
        count += 1
        if count == sampleSize:
            break
        print("读取数据的进度是", count, "/", sampleSize)
    # pickle.dump(dataList, open('data.pkl','wb'))
    # dataList = pickle.load(open('data.pkl','rb'))
    print("删除不优质的特征")

    for sample in dataList:
        for key in list(sample.keys()):
            if key not in betterFeatureSet:
                del sample[key]#删除不是优质特征的条目
    #dataList里现在是稀疏向量
    #将数据整理为spark可接受的形式
    import random
    random.shuffle(dataList)
    print(dataList[:10])
    res = []
    for sample in dataList:
        sentiment = sample['sentiment']
        sample.pop('sentiment')
        # if len(sample) < 3:
        #     continue
        sparseV = []
        for key, value in sample.items():
            if one_hot==True:
                sparseV.append((vocab[key], 1))
            else:
                sparseV.append((vocab[key], value))
        # print(gender, sparseV)
        sparseV = sorted(sparseV, key=lambda x:x[0])
        res.append([sentiment, SparseVector(vocabSize, sparseV)])
    # trainData, testData = res[:int(len(res)*0.8)], res[int(len(res)*0.8):]
    return res

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

    conf = SparkConf().setAppName("hupu_block")#配置spark任务的基本参数
    sc = SparkContext(conf=conf)#创建sc对象
    sqlcontext = SQLContext(sc)
    one_hot_list = ['one_hot_encoding']
    for one_hot_stat in one_hot_list:
        if_one_hot = True if one_hot_stat=='one_hot_encoding' else False
        #获取训练集和测试集
        trainData, testData, vocabSize, betterFeatureSet = ngramFreatures(-1,one_hot = False)
        #微博数据需要使用和京东数据一样的词汇表，得修改一下代码
        weiboTestData = ngramFreatures_weibo(-1,vocabSize, betterFeatureSet, one_hot=False)
        # print(trainData[:2])
        # time.sleep(20)
        #将本地数据放到集群中
        df_oriFeatures_train = sqlcontext.createDataFrame(trainData,['label', 'orifeatures']).repartition(30)
        df_oriFeatures_test = sqlcontext.createDataFrame(testData,['label', 'orifeatures']).repartition(30)
        df_weiboTestData = sqlcontext.createDataFrame(weiboTestData,['label', 'orifeatures']).repartition(30)
        # df_oriFeatures_train1 = df_oriFeatures_train.repartition(500)
        # df_oriFeatures_test1 = df_oriFeatures_test.repartition(50)
        #训练特征选择器
        evaluationMap = {'lrm':{}, 'nb':{}, 'mp': {}}
        for i in range(500, 50000, 500):
            x2selector = ChiSqSelector(numTopFeatures=i,featuresCol= 'orifeatures', outputCol="features")
            x2selector_model = x2selector.fit(df_oriFeatures_train)
            df_good_oriFeatures_train = x2selector_model.transform(df_oriFeatures_train).\
                                      select(['label', 'features'])#.repartition(30)
            df_good_oriFeatures_test = x2selector_model.transform(df_oriFeatures_test).\
                                      select(['label', 'features'])#.repartition(30)

            df_good_oriFeatures_weibo = x2selector_model.transform(df_weiboTestData).\
                                      select(['label', 'features'])#
            #挑选特征

            #训练分类器
            lrm = LogisticRegression(maxIter=20, regParam=0.1).fit(df_good_oriFeatures_train)
            nb = NaiveBayes().fit(df_good_oriFeatures_train)
            mp = MultilayerPerceptronClassifier(maxIter=100,
                      layers=[len(list(df_good_oriFeatures_train.select('features').take(1)[0].features)), 200,50,  2],
                          blockSize=1, seed=123).fit(df_good_oriFeatures_train)
            # rf = RandomForestClassifier(numTrees=50, maxDepth=10,
            #         featuresCol= 'features', labelCol="label", seed=42).\
            #           fit(df_good_oriFeatures_train)
            clfMap = {'lrm':lrm, 'nb':nb, 'mp': mp}

            #评估分类器
            for clfName, clf in clfMap.items():
                predRes = clf.transform(df_good_oriFeatures_train)
                res = predRes.rdd.map(lambda x: [x.label, x.prediction]).collect()
                predLabel = list(map(lambda x: x[1], res))
                realLabel = list(map(lambda x: x[0], res))
                # 计算混淆矩阵，准确率，召回率，精度
                evaluation = showConfusionMatrix(realLabel, predLabel, 2)
                evaluationMap[clfName][i] = {"trainingSet": evaluation}
                predRes1 = clf.transform(df_good_oriFeatures_test)
                res1 = predRes1.rdd.map(lambda x: [x.label, x.prediction]).collect()
                predLabel1 = list(map(lambda x: x[1],res1))
                realLabel1 = list(map(lambda x: x[0], res1))
                #计算混淆矩阵，准确率，召回率，精度
                evaluation1 = showConfusionMatrix(realLabel1, predLabel1, 2)
                evaluationMap[clfName][i]['testSet'] = evaluation1
                print("评估结果是", evaluationMap)
                import pickle
                pickle.dump(evaluationMap, open(one_hot_stat + '_evaluation.pkl', 'wb'))
        #data = pickle.load(open('one_hot_encoding_evaluation.pkl', 'rb'))
        #plotDimensionWithF1Score(data)