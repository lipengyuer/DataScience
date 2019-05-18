import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from numpy.distutils.system_info import accelerate_info

def processLog(logStr, mode='work'):
    actions = logStr.split('#')
    actionsNum = len(actions)#行为个数
    freqMap = {'item_id_num':set() , 'category_id_num': set() , 'brand_id_num':set()  ,
                'time_stamp_num':set() , 'action_type_total_num':set() }
    actionTypeFreq = {'0_freq': 0, '1_freq': 0, '2_freq': 0, '3_freq': 0}
    nearBuyNum = 0#用户有几次购买或者接近购买
    time_stamp_set = set()
    for action in actions:
        s = action.split(':')
        if len(s)<5:
            continue
        [item_id, category_id, brand_id, time_stamp, action_type] = s
        if action_type in [1, 2, 3] and time_stamp not in time_stamp_set:
            nearBuyNum += 1
            time_stamp_set.add(time_stamp)
            
        freqMap['item_id_num'].add(item_id)
        freqMap['category_id_num'].add(category_id)
        freqMap['brand_id_num'].add(brand_id)
        freqMap['action_type_total_num'].add(action_type)
        freqMap['time_stamp_num'].add(time_stamp)
        actionTypeFreq[action_type + '_freq'] = actionTypeFreq.get(action_type+ '_freq', 0) + 1
    features = [actionsNum]
    names = ['actionsNum']
    for key in freqMap: 
        features.append(len(freqMap[key])/(actionsNum + 0.00000001))
        names.append(key)
    for key in actionTypeFreq: 
        features.append(actionTypeFreq[key]/(actionsNum + 0.00000001))
        names.append(key)
    features.append(nearBuyNum)
    names.append('nearBuyNum')
    #print("日志里的特征有",names)
    if mode=='work':
        return features
    else:
        return features, names

def featureEngineering(df):
    df = df.fillna(-666)
    df = df[df['label']>=0]
    df = df[df['user_id']!='user_id']
    df['user_id'] = df['user_id'].astype(int)
    df['age_range'] = df['age_range'].astype(int).apply(lambda x: 0 if x==-666 else x)
    df['gender'] = df['gender'].astype(int).apply(lambda x: -1 if x==-666 else x)
    df['merchant_id'] = df['merchant_id'].astype(int)
    df['label'] = df['label'].astype(int)
    df = df[df['merchant_id']!=-666]
    df = df[df['activity_log']!=-666]
    _, featuresFromLog = processLog(df['activity_log'].values[0], mode='test')
    activity_log = list(map(lambda x: processLog(x), df['activity_log'].values))
    df = df.reset_index(drop=True)
    df_log = pd.DataFrame(activity_log, columns=featuresFromLog)

    df = pd.concat([df, df_log], axis=1)
    dropFreatures = ['label', 'user_id', 'merchant_id', 'activity_log']
    X = df.drop(dropFreatures, axis=1)
    Y = df[['label']]
    return X, Y

def  dataProcess():
    fileName = r"c:\\Users\\Administrator\\Desktop\\简单任务\回头客项目\\data_format2\\train_format2_sub.csv"
#     fileName = r"c:\\Users\\Administrator\\Desktop\\简单任务\回头客项目\\data_format2\\train_format2.csv"
    df = pd.read_csv(fileName)
    X,Y = featureEngineering(df)
    return X, Y


def calAccuracy(inputData, labels, model):
    preds = model.predict(inputData)
    labels = list(map(lambda x: x[0], labels))
    #print('preds', preds)
    #print('labels', labels)
    auc = metrics.roc_auc_score(labels,preds)#验证集上的auc值
    print('AUC值是', auc)
    return auc

from sklearn import metrics
from sklearn import tree
from sklearn import ensemble
def KFoldValidation(X, Y, K=10):
    kfolder = KFold(n_splits=K)
    totalAUC  = 0
    count = 0
    for trainIndex, testIndex in kfolder.split(X, Y):
        count += 1
        trainX, trainY = X.iloc[trainIndex], Y.iloc[trainIndex]
        testX, testY = X.iloc[testIndex], Y.iloc[testIndex]
#         clf = LogisticRegression()
#         clf = tree.DecisionTreeClassifier()
        clf = ensemble.RandomForestRegressor(n_estimators=50, max_depth=6, max_features=0.5)
#         clf = ensemble.GradientBoostingRegressor(learning_rate=0.1, n_estimators=10, subsample =0.8)
        #clf = ensemble.AdaBoostRegressor()
#         clf = SVC()
        clf.fit(trainX, trainY)
        print(count, "在训练集中:")
        train_auc = calAccuracy(trainX, trainY.values, clf)
        print(count, "在测试集中:")
        test_auc = calAccuracy(testX, testY.values, clf)
        totalAUC += test_auc

    print('avg AUC is:', totalAUC/K)

def  getIDs():
    print("读取id数据。")
    id_set = set({})
    fileName = r"c:\Users\Administrator\Desktop\简单任务\回头客项目\sample_submission.csv"
    with open(fileName, 'r') as f:
        line = f.readline()
        line = f.readline()
        while line!='':
            #print(line)
            fields = line.split(',')
            user_id, merchant_id = fields[0], fields[1]
            id_set.add(user_id + '_' + merchant_id)
            line = f.readline()
    return id_set

def getFinalResult(clf):
    id_set = getIDs()
    fileName = r"c:\\Users\\Administrator\\Desktop\\简单任务\回头客项目\\data_format2\\test_format2.csv"
    data = []
    count = 0
    with open(fileName, 'r') as f:
        line = f.readline()
        line = f.readline()
        while line!='':
            #print(line)
            fields = line.replace('\n', '').split(',')
            #print(line)
            user_id, merchant_id, activity_log, age_range, gender = fields[0], fields[3], fields[-1], fields[1], fields[2]
            if user_id + '_' + merchant_id in id_set:
                if age_range=='': age_range='0'
                if gender=='': gender='0'
                data.append([int(user_id),int(age_range), int(gender),  int(merchant_id), -1,  activity_log])
                count += 1
                if count%10000==0:
                    print(count)
                    #break
            line = f.readline()
            
                
    df = pd.DataFrame(data, columns=['user_id', 'age_range','gender', 'merchant_id', 'label', 'activity_log'])
    df['label'] = df['label'].apply(lambda x: 2)
    X, _ = featureEngineering(df)
    #print(X)
    Y = clf.predict(X)
    Y = list(map(lambda x: [x], Y))
    df_Y = pd.DataFrame(Y, columns=['prob'])
    df = pd.concat([df, df_Y], axis=1)
    df = df[['user_id', 'merchant_id', 'prob']]
    df.to_csv('c:\\Users\\Administrator\\Desktop\\简单任务\回头客项目\\myRes.csv', index=0)
    
        
if __name__ == '__main__':
    print("处理数据")
    X, Y = dataProcess()
    print("开始交叉验证")
    KFoldValidation(X, Y)
#     print("开始训练模型")
#     clf = LogisticRegression()
# #     clf = ensemble.RandomForestRegressor(n_estimators=100, max_depth=8, max_features=0.4)
#     clf.fit(X, Y)
#     print("开始对目标进行计算。")
#     getFinalResult(clf)

    