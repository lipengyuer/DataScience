import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from numpy.distutils.system_info import accelerate_info
    
def  dataProcess20190314():
    fileName = r"c:\\Users\\Administrator\\Desktop\\简单任务\回头客项目\\data_format2\\train_format2_sub.csv"
    fileName = r"c:\\Users\\Administrator\\Desktop\\简单任务\回头客项目\\data_format2\\train_format2.csv"

    df = pd.read_csv(fileName)
    df = df.fillna(-666)
#     df = df.iloc[369150:369200]
#     print(df.columns)
    df = df[df['user_id']!='user_id']
    df['user_id'] = df['user_id'].astype(int)
    df['age_range'] = df['age_range'].astype(int).apply(lambda x: 0 if x==-666 else x)
    df['gender'] = df['gender'].astype(int).apply(lambda x: -1 if x==-666 else x)
    df['merchant_id'] = df['merchant_id'].astype(int)
    df['label'] = df['label'].astype(int)
    df = df[df['merchant_id']!=-666]
    df = df[df['activity_log']!=-666]
    df['activity_log'] = df['activity_log'].apply(lambda x: len(str(x).split('#'))).astype(int)
    df = df[df['label']>=0]
    
    X = df[['activity_log', 'age_range', 'gender']]
    Y = df[['label']]
    return X, Y


def calAccuracy(inputData, labels, model):
    preds = model.predict(inputData)
    c = 0
    print(preds)
    for i in range(len(preds)):
        if preds[i]== labels[i] :
            c += 1
    return  c/len(preds), f1_score(preds, labels)
    
def KFoldValidation(X, Y, K=10):
    kfolder = KFold(n_splits=K)
    totalAcc, totalF1  = 0, 0
    for trainIndex, testIndex in kfolder.split(X, Y):
        trainX, trainY = X.iloc[trainIndex], Y.iloc[trainIndex]
        testX, testY = X.iloc[testIndex], Y.iloc[testIndex]
        clf = LogisticRegression()
#         clf = MLPClassifier()
#         clf = SVC()
        clf.fit(trainX, trainY)
        print("在训练集中:")
        calAccuracy(trainX, trainY.values, clf)
        print("在测试集中:")
        acc, f1 = calAccuracy(testX, testY.values, clf)
        totalAcc += acc
        totalF1 += f1

    print('f1 score is:', acc/K)
    print("准确率是", totalF1/K)


