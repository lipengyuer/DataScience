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
import copy
from sklearn.ensemble.weight_boosting import AdaBoostRegressor
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.svm.classes import SVC, SVR

class Classifier():
    '''为不同年龄段的人分别训练模型，使用不同的模型去预测'''
    def __init__(self):
        base_clf =  GradientBoostingRegressor(loss='ls', n_estimators=200, learning_rate=.01,
                                        max_depth=5,
                                         random_state=int(time.time()), max_features=0.3,\
                                         min_samples_leaf=10, subsample=0.3)
#         base_clf = RandomForestRegressor(n_estimators=20, max_depth=7, criterion='mse', \
#                                     max_features=0.3)
        self.clf_map = {'-30': copy.deepcopy(base_clf), \
                        '30-40':copy.deepcopy(base_clf), \
                        '60-': copy.deepcopy(base_clf)}
        self.age_group_data = [[0,40, '-30'], [40, 65, '30-40'],[65, 100, '60-']]
        
    def determine_age_group(self, age):
        #判断年龄段，如果分组较多，需要换为二分查找
        for data in self.age_group_data:
            if data[0] <= age <data[1]:
                return data[2]
    
    def fit(self, X, Y):
        X, Y = copy.deepcopy(X), copy.deepcopy(Y)
        X['age_group'] = X['age'].apply(self.determine_age_group)
        X = X.reset_index(drop=True)
        X = X.reindex(list(range(len(Y))))
        Y = Y.reset_index(drop=True)
        Y = Y.reindex(list(range(len(Y))))
        for age_roup in self.clf_map:
            group_size = len(X[X['age_group']==age_roup])
            index_this_age_group = list(range(len(Y)))
            random.shuffle(index_this_age_group)
            index_this_age_group = index_this_age_group[:int(group_size*0.3)]
            X_this_age_group = X.iloc[index_this_age_group]
            Y_this_age_group = Y.iloc[index_this_age_group]
            X_this_age_group = X_this_age_group.drop(['age_group'], axis=1)
            #print(np.reshape(Y_this_age_group.values, [-1]))
            self.clf_map[age_roup].fit(X_this_age_group, np.reshape(Y_this_age_group.values, [-1]))
                        
            index_this_age_group = X[X['age_group']==age_roup].index.tolist()
            X_this_age_group = X.iloc[index_this_age_group]
            
            X_this_age_group = X_this_age_group.drop(['age_group'], axis=1)
            Y_this_age_group = Y.iloc[index_this_age_group]
#             print(Y_this_age_group)
            self.clf_map[age_roup].fit(X_this_age_group, np.reshape(Y_this_age_group.values, [-1]))
            

            
    def predict(self, input_data_df):
        ages = input_data_df['age'].values
        input_data_df = input_data_df.values
        result = []
        for i in range(len(input_data_df)):
            input_data = input_data_df[i, :]
            age_group = self.determine_age_group(ages[i])
            clf = self.clf_map[age_group]
            label = clf.predict([input_data])
            #print(label)
            result.append(label[0])
        #print(result)
        return result

import random
def get_rand(hap_v):
    hap_v = int(hap_v)
    if hap_v<3 or hap_v>4: return 1
    else: return random.uniform(0,1)
    
def loadData(fileName):
    data = pd.read_csv(fileName)
    data = data.drop(['id'], axis=1)
    data = data[data['happiness']>0]

    data = data.fillna(-1)
#     data['random'] = data['happiness'].apply(get_rand)
#     data = data[data['random']>0.5]
    y = data[['happiness']]
    data = data.drop(['happiness'], axis=1)

    x = feature_engineering.featureEngineering(data)
    info = x.describe()
    print(info)
    info.to_csv(rootPath + 'data_describe.csv', index=0)
    return x,y

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
from sklearn.tree import DecisionTreeClassifier
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
        count += 1
        # print(trainX.size, trainY.size)

        trainInput, trainOutput = trainX.iloc[trainIndex], trainY.iloc[trainIndex]
        # print(trainInput['edu_yr'])
        testInput, testOutput = trainX.iloc[testIndex], trainY.iloc[testIndex]
        weight = compute_class_weight('balanced',[1,2,3,4,5], list(map(lambda x: x[0], trainOutput.values)))
        weight = [[i+1, weight[i]] for i in range(len(weight))]
        weight = dict(weight)
        weight = {1:1/104, 2:1/497, 3:1/1159, 4:1/4818, 5:1/1410}
        for key in weight: weight[key] = weight[key]**0.55
        
        clf = Classifier()
        clf.fit(trainInput, trainOutput)

        pred = clf.predict(trainInput)
        trainacc, traincostn = evaluation(pred, trainOutput.values)

        pred = clf.predict(testInput)
        acc, costn = evaluation(pred, testOutput.values)
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


if __name__ == '__main__':
    rootPath = './data/'
    trainX, trainY = loadData(rootPath + 'happiness_train_complete.csv')
    #print(trainX)
    # trainX, trainY = loadData(rootPath + 'happiness_train_abbr.csv')
    # trainX = abs(trainX)
    # selector = SelectKBest(chi2, k=80)  # Ñ¡Ôñk¸ö×î¼ÑÌØÕ÷
    # selector.fit(trainX, trainY)
    # trainX = selector.transform(trainX)
    #gridSearch(trainX, trainY)
    KFoldTest(trainX, trainY)
#     clf = GradientBoostingRegressor(loss='huber', n_estimators=1000, learning_rate=.01,
#                                         max_depth=6,
#                                          random_state=int(time.time()), max_features=0.3,\
#                                          min_samples_leaf=30, subsample=0.3)
#     clf.fit(trainX, trainY)
#     res, contData = loadContestData(rootPath + 'happiness_test_complete.csv')
#     labels = clf.predict(contData)
#     res['happiness'] = labels
#     res = res[['id', 'happiness']]
#     res.to_csv(rootPath + 'myRes.csv', index=0)


    """happiness              ...                         
1           104        ...                      104
2           497        ...                      497
3          1159        ...                     1159
4          4818        ...                     4818
5          1410        ...                     1410
find happiness==4 at first ...
"""

