import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import otherData
import feature_engineering
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV

data_file = './data/happiness_train_complete_fix.csv'

def stastics(data):
    print(data.groupby(['happiness']).count())

import random
def get_rand(hap_v):
    hap_v = int(hap_v)
    if hap_v<3 or hap_v>4: return 1
    else: return random.uniform(0,1)

from data_augmentation import aug_data_age_group
def loadData(fileName):
    data = pd.read_csv(fileName)
    data = data.drop(['id'], axis=1)
    data = data[data['happiness']>0]
    y = data[['happiness']]
    data = data.drop(['happiness'], axis=1)
    x = feature_engineering.featureEngineering(data)
    info = x.describe()
    print(info)
    info.to_csv('./data/data_describe.csv', index=0)
    return x,y#trainX, testX, trainY, testY

def loadContestData(fileName):
    data = pd.read_csv(fileName)
    res = data[['id']]
    data = data.drop(['id'], axis=1)
    data = feature_engineering.featureEngineering(data)
    
    return res, data

def evaluation(pred, y):
    count = 0
    cost = 0
    for i in range(len(y)):
        if pred[i]==y[i][0]: count += 1
        cost += (pred[i]-y[i][0])**2
        #print(pred[i], y[i][0])
    cost /= len(y)
    #print(count/len(y))
    return count/len(y), cost

def findWhoClassifiedWrongly():
    data = pd.read_csv(data_file)
    print("数据总量是", len(data))
    data = data[data['happiness']>0]
    print("删除没有幸福感值后的数据量是", len(data))
    ids = data['id'].values
    data = data.drop(['id'], axis=1)
    trainY = data[['happiness']]
    data = data.drop(['happiness'], axis=1)
    trainX = feature_engineering.featureEngineering(data)
    print("删除没有幸福感值后的数据量是", len(data))
    clf = GradientBoostingRegressor(loss='ls', n_estimators=500, learning_rate=.05,
                                        max_depth=6,
                                         random_state=int(time.time()), max_features=0.3,\
                                         min_samples_leaf=30, subsample=0.3)
    clf.fit(trainX, trainY)
    
    real_labels = trainY.values
    labels = clf.predict(trainX)
    print("标签的数量是", len(labels))
    for i in range(min(200000, len(labels))):
        if abs(labels[i]-real_labels[i][0]>3):
            print(labels[i], real_labels[i], ids[i])
    trainacc, traincostn = evaluation(labels, real_labels)
    print(trainacc, traincostn)

    


import time
def KFoldTest(trainX, trainY):
    K=10
    kf = KFold(n_splits=K, random_state=int(time.time()))
    totalAcc = 0
    totalTrainingAcc = 0
    cost = 0
    trainingCost = 0
    count = 0
    for trainIndex, testIndex in kf.split(trainX):
        # print(trainX.size, trainY.size)

        trainInput, trainOutput = trainX.iloc[trainIndex].copy(), trainY.iloc[trainIndex].copy()
#         trainInput, trainOutput = aug_data_age_group(trainInput, trainOutput)
        print("扩增后的训练数据量是", len(trainInput))
#         print(trainInput['birth'])
        testInput, testOutput = trainX.iloc[testIndex], trainY.iloc[testIndex]

#         clf = RandomForestRegressor(n_estimators=200, max_depth=5, n_jobs=8, criterion='mse', \
#                                     max_features=0.3, bootstrap=False)
        # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
        #                          algorithm="SAMME",
        #                          n_estimators=200)
        # clf = GradientBoostingClassifier( n_estimators=500, learning_rate=.01, max_depth=10,
        #                                  random_state=nt(time.time()), max_features=0.3,\
        #                                  min_samples_leaf=20, subsample=0.9)
        clf = GradientBoostingRegressor(loss='huber', n_estimators=500, learning_rate=.01,
                                        max_depth=5,random_state=int(time.time()), max_features=0.3,\
                                         min_samples_leaf=20, subsample=0.3)
        clf.fit(trainInput, trainOutput.values)

        pred = clf.predict(trainInput)
        trainacc, traincostn = evaluation(pred, trainOutput.values)


        pred = clf.predict(testInput)
        acc, costn = evaluation(pred, testOutput.values)
        count += 1
        print(count, "training acc", trainacc, traincostn,trainOutput.size,  'testing acc', acc, costn ,testOutput.size)
        totalAcc += acc
        cost += costn
        totalTrainingAcc += trainacc
        trainingCost += traincostn
    print("k-fold crossvalidation:", totalAcc/K, 'cost is', cost/K)
    print("in training is ", totalTrainingAcc/K, trainingCost/K)

def gridSearch(trainX, trainY):

        parameters = {'loss':['ls'], 'n_estimators':[1000],
                      'learning_rate':[0.01, 0.008], 'max_depth':[10, 15, 20],
                    'random_state':[int(time.time())], 'max_features':[0.1,0.2,0.3],
                        'min_samples_leaf':[10,30], 'subsample':[0.9]}
        gbdt = GradientBoostingRegressor()
        #DT = DecisionTreeClassifier()
        clf = GridSearchCV(gbdt, parameters, n_jobs=-1, verbose=2, cv=5, scoring='neg_mean_squared_error')
        clf.fit(trainX, trainY.values)
        cv_result = pd.DataFrame.from_dict(clf.cv_results_)
        with open('cv_result.csv', 'w') as f:
            cv_result.to_csv(f)

def task(trainX, trainY):
    rootPath = './data/'
    clf = GradientBoostingRegressor(loss='ls', n_estimators=1000, learning_rate=.01,
                                        max_depth=6,
                                         random_state=int(time.time()), max_features=0.3,\
                                         min_samples_leaf=10, subsample=0.3)
    clf.fit(trainX, trainY)
    res, contData = loadContestData(rootPath + 'happiness_test_complete.csv')
    labels = clf.predict(contData)
    res['happiness'] = labels
    res = res[['id', 'happiness']]
    res.to_csv(rootPath + 'myRes.csv', index=0)

if __name__ == '__main__':
    trainX, trainY = loadData(data_file)
    KFoldTest(trainX, trainY)
#     task(trainX, trainY)

#     findWhoClassifiedWrongly()

    """happiness              ...                         
1           104        ...                      104
2           497        ...                      497
3          1159        ...                     1159
4          4818        ...                     4818
5          1410        ...                     1410
find happiness==4 at first ...
"""

