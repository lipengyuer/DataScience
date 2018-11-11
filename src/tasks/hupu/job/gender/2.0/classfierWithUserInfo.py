# coding=utf8
import sys, os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
path_src = path.split('src')[0]
sys.path.append(path_src + "src")  # 项目的根目录
import pymongo
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble  import VotingClassifier, RandomForestClassifier

import runTime
from analysis.config import environment
# from analysis.algorithm.stacking import StackingClassifier
import numpy as np
from pymongo import MongoClient
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BaseNB
from sklearn.svm import SVC

userOriFeatureCollection = 'hupuUserInfo'
MONGO_IP, MONGO_PORT, dbname, username, password = environment.MONGO_IP, environment.MONGO_PORT, \
                                                   environment.MONGO_DB_NAME, None, None
conn = MongoClient(MONGO_IP, MONGO_PORT)
db = conn[dbname]
# db.authenticate(username, password)
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold as kf
import time

def time2timestamp(timestr):
    if not timestr:
        return int(time.time())
    if timestr=='今天':
        return int(time.time())
    if timestr=='昨天':
        return int(time.time()) - 3600*24
    if timestr=='前天':
        return int(time.time()) - 3600*24*2
    timeArray = time.strptime(timestr, "%Y-%m-%d")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


def testClassifierComplexFeatures():
    featureNames = ['lastLoginTime', 'registerTime', 'homeTeams', 'fans', 'theOrg', 'follow',
                    'location', 'uid', 'onlineTime', 'userNumCame',
                    'communityRPScore','HPLevel', 'gender']
    featureNames = set(featureNames)
    featureNamesMap = {}
    for name in featureNames:
        featureNamesMap[name] = 1
    featureNamesMap['_id'] = 0
    print("读取数据") # , db['bigram'],db['postagBigramFreq']]
    uids = db[userOriFeatureCollection].find({}, featureNamesMap)
    dataList = []
    count = 0
    progressCount = 0
    for line in uids:
        progressCount += 1
        print("正在读取第", progressCount, "个用户。", )
        # print(set(uid.keys())&featureNames)
        if len(set(line.keys())&featureNames)<7 or 'gender' not in line:
            continue
        # if line['gender']=='m' and len(dataList)>0 and dataList[-1]['gender']=='f':
        #     pass
        if line['gender']=='m' and random.uniform(0, 1) > 0.05:
            continue
        count += 1
        if 'communityRPScore' in line and line['communityRPScore'] !=None and line['communityRPScore']<0:
            line['communityRPBad'] = 1
        if 'userName' in line:
            line = len(line['userName'])
        if 'location' in line:
            line['location'] = 1
        if 'follow' in line:
            line['follow'] = len(line['follow'])
        if 'fans' in line:
            line['fans'] = len(line['fans'])
        if 'theOrg' in line:
            if line['theOrg']=='小黑屋住户':
                line['theOrgBlock'] = 1
            line['theOrg'] = 1
        if 'homeTeams' in line:
            line['homeTeams'] = len(line['homeTeams'])
        if 'registerTime' in line:
            line['registerTime'] = time2timestamp(line['registerTime'])-\
                                   time2timestamp('2003-1-1')
            line['registerTime'] = int(line['registerTime']/3600)
        if 'lastLoginTime' in line:
            line['lastLoginTime'] = int(time.time()) - time2timestamp(line['lastLoginTime'])
            line['lastLoginTime'] = int(line['lastLoginTime']/3600)
        # if count == 200000:
        #     break
        dataList.append(line)
    df = pd.DataFrame(dataList)
    dfClean = df[list(featureNames)].drop(columns=['gender', 'uid']).fillna(0)
    X, Y = dfClean.values, df['gender']
    print(Y)
    print("开始交叉验证")
    index = kf(n_splits=10, random_state=666).split(list(range(len(Y))))
    cmTotal = np.zeros((2, 2))
    count = 0
    for line in index:
        # clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=100)#基于词频0.75
        # clf = DecisionTreeClassifier()
        # clf = RandomForestClassifier()
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=666, n_jobs=6)
        #clf = MLPClassifier(hidden_layer_sizes=(50,10), max_iter=500, learning_rate_init=0.01)
        # clf = SVC(C=0.9)
        trainIndex = line[0]
        testIndex = line[1]
        trainX = X[trainIndex]
        testX = X[testIndex]
        trainy = Y[trainIndex]
        testy = Y[testIndex]
        # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        # featureProcessor = LinearDiscriminantAnalysis(n_components=150)
        # featureProcessor.fit(trainX, trainy)
        # trainX = featureProcessor.transform(trainX)
        # testX = featureProcessor.transform(testX)

        clf.fit(trainX, trainy)
        count += 1
        print(count, "训练集表现:")
        y_train = clf.predict(trainX)
        cm = confusion_matrix(trainy, y_train)
        print(cm)
        y_pred = clf.predict(testX)
        cm = confusion_matrix(testy, y_pred)
        cmTotal += np.array(cm)
        print(count, "测试集表现：")
        print(cm)
    print(cmTotal)
    print((cmTotal[0, 0] + cmTotal[1, 1]) / (sum(sum(cmTotal))))


if __name__ == '__main__':
    testClassifierComplexFeatures()