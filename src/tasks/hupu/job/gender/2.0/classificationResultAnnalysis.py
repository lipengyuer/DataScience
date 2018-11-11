# coding=utf8
import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
path_src = path.split('src')[0]
sys.path.append(path_src + "src")#项目的根目录
import pandas as pd
from analysis.config import environment
import numpy as np
from pymongo import MongoClient

MONGO_IP, MONGO_PORT ,dbname, username, password = environment.MONGO_IP, environment.MONGO_PORT, \
                                                   environment.MONGO_DB_NAME, None, None
conn = MongoClient(MONGO_IP, MONGO_PORT)
db = conn[dbname]
import pickle
clf = pickle.load(open('genderClassfier.pkl','rb'))

def addLine(path, line):
    with open(path, 'a+') as f:
        f.write(line)
def classificationResult(featureCollectionName = "", scale=True):
    def scala(x):
        res = []
        for i in range(len(x)):
            temp = []
            for n in x[i]:
                v = 1 if n>0 else 0
                temp.append(v)
            res.append(temp)
        return np.array(res)
    print("读取数据")
    collection = db[featureCollectionName]
    data = collection.find({})#.limit(100)#从mongo中查询这个特征以及对应的性别标签
    data = list(data)[5000:]
    df = pd.DataFrame(data)

    count = 0
    for i in range(len(df)):
        sampleData = df.loc[i]
        uid, gender = sampleData['uid'], sampleData['gender']
        # print(uid, gender, type(sampleData))
        dfClean = sampleData.drop(labels=['gender', '_id', 'uid'])
        # print(np.sum(dfClean.values))
        res = clf.predict([dfClean])
        # print(res)
        if res[0]!=gender:
            count += 1
            print(count, uid)
            addLine('errorInClassification.txt', str(uid)+"\n")


    #     X, Y = dfClean.values, df['gender']
    #     trainIndex = line[0]
    #     testIndex = line[1]
    #     trainX = X[trainIndex]
    # print(cmTotal)
    # print((cmTotal[0,0]+ cmTotal[1,1])/(sum(sum(cmTotal))))
    # print("召回率是", cmTotal[1, 1] / (cmTotal[1, 0] + cmTotal[1, 1]), "精度是", cmTotal[1, 1] / (cmTotal[0, 1] + cmTotal[1, 1]))


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold as kf
from sklearn.linear_model import LogisticRegression
import pickle
def testClassifier(featureCollectionName = "", scale=True):
    def scala(x):
        res = []
        for i in range(len(x)):
            temp = []
            for n in x[i]:
                v = 1 if n>0 else 0
                temp.append(v)
            res.append(temp)
        return np.array(res)
    print("读取数据")
    collection = db[featureCollectionName]
    data = collection.find({})#从mongo中查询这个特征以及对应的性别标签
    data = list(data)#[:500]
    df = pd.DataFrame(data)

    dfClean = df.drop(columns=['gender','_id', 'uid'])
    X, Y = dfClean.values, df['gender']
    if scale:
        X = scala(X)
    print("开始交叉验证")
    index =list(range(len(Y)))
    cmTotal = np.zeros((2, 2))
    count = 0
    clf = LogisticRegression(max_iter=50, solver='newton-cg', C=0.1)#基于词频0.75
    trainIndex = index[0:5000]
    testIndex = index[5000:]
    trainX = X[trainIndex]
    testX = X[testIndex]
    trainy = Y[trainIndex]
    testy = Y[testIndex]
    clf.fit(trainX, trainy)
    count += 1
    print(count,"训练集表现:")
    y_train = clf.predict(trainX)
    cm = confusion_matrix(trainy, y_train)
    print(cm)
    y_pred = clf.predict(testX)
    cm = confusion_matrix(testy, y_pred)
    cmTotal += np.array(cm)
    print(count,"测试集表现：")
    print(cm)
    pickle.dump(clf, open('genderClassfier.pkl','wb'))
    # dataList = pickle.load(open('genderClassfier.pkl','rb'))
    print(cmTotal)
    print((cmTotal[0,0]+ cmTotal[1,1])/(sum(sum(cmTotal))))
    print("召回率是", cmTotal[1, 1] / (cmTotal[1, 0] + cmTotal[1, 1]), "精度是", cmTotal[1, 1] / (cmTotal[0, 1] + cmTotal[1, 1]))


if __name__ == '__main__':
    # testClassifier(featureCollectionName='wordFreq', scale=True)
    classificationResult(featureCollectionName='wordFreq', scale=True)