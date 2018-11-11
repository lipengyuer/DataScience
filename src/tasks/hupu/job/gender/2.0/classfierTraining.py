# coding=utf8
import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
path_src = path.split('src')[0]
sys.path.append(path_src + "src")#项目的根目录
import pymongo
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

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


userOriFeatureCollection = runTime.ORI_USER_FEATURE_SAMPLE_COLLECTION
MONGO_IP, MONGO_PORT ,dbname, username, password = environment.MONGO_IP, environment.MONGO_PORT, \
                                                   environment.MONGO_DB_NAME, None, None
conn = MongoClient(MONGO_IP, MONGO_PORT)
db = conn[dbname]
# db.authenticate(username, password)

import sklearn as sk
import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold as kf

class StackingClassifier():

    def __init__(self):
        self.baseModels = {}
        self.oriBaseModels = {}
        self.metaModel = None
        self.oriMetaModel = None
        self.clfNames = []

    def setBaseModels(self, baseModels={}):  # 设置初级分类器
        """baseModels的元素是一个dict，比如{"clname": SVC()}"""
        self.baseModels = baseModels
        self.oriBaseModels = baseModels
        self.clfNames = list(baseModels.keys())

    def setMetaModel(self, metaModel=None):  # 设置次级分类器
        self.metaModel = metaModel
        self.oriMetaModel = metaModel

    def fit(self, XMap={}, Y=None):  # 训练分类器
        "XMap是一个dict,key为初级分类器的名字，value是对应的特征，要求是numpy.array"
        # 训练初级分类器
        for modelName in self.baseModels:
            self.baseModels[modelName].fit(XMap[modelName], Y)

        # 计算分类器的输出
        basePrediction = np.zeros((len(Y), 1))
        for modelName in self.clfNames:
            baseProbability = self.baseModels[modelName].predict_proba(XMap[modelName])
            basePrediction = np.concatenate((basePrediction, baseProbability), axis=1)
        basePrediction = basePrediction[:, 1:]
        # 训练次级分类器
        self.metaModel.fit(basePrediction, Y)

    def predict(self, XMap={}):
        featureNum = len(XMap[self.clfNames[0]])
        basePrediction = np.zeros((featureNum, 1))
        for modelName in self.baseModels:
            baseProbability = self.baseModels[modelName].predict_proba(XMap[modelName])
            basePrediction = np.concatenate((basePrediction, baseProbability), axis=1)
        basePrediction = basePrediction[:, 1:]
        result = self.metaModel.predict(basePrediction)
        return result

    def kFoldValidatoin(self, XMap, Y, k=10, classNum=2):
        randomIndex = random.sample(range(len(Y)), len(Y))
        for clfName in XMap:
            XMap[clfName] = XMap[clfName][randomIndex]
        y = Y[randomIndex]
        cmTotal = np.zeros((classNum, classNum))
        index = kf(n_splits=k, random_state=666).split(list(range(len(Y))))
        for line in index:
            trainIndex = line[0]
            testIndex = line[1]
            trainX, testX = {}, {}
            trainy = y[trainIndex]
            testy = y[testIndex]
            for clfName in XMap:
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                featureProcessor = LinearDiscriminantAnalysis(n_components=100)

                trainX[clfName] = XMap[clfName][trainIndex]
                testX[clfName] = XMap[clfName][testIndex]
                featureProcessor.fit(trainX[clfName], trainy)
                trainX[clfName] = featureProcessor.transform(trainX[clfName])
                testX[clfName] = featureProcessor.transform(testX[clfName])

            import copy
            self.baseModels = copy.deepcopy(self.oriBaseModels)
            self.metaModel = copy.deepcopy(self.oriMetaModel)
            self.fit(trainX, trainy)
            print("训练集表现:")
            y_train = self.predict(trainX)
            cm = confusion_matrix(trainy, y_train)
            print(cm)
            y_pred = self.predict(testX)
            cm = confusion_matrix(testy, y_pred)
            cmTotal += np.array(cm)
            print("测试集表现：")
            print(cm)
        print(cmTotal)
        return cmTotal

def querySimpleFeatures(featureName='', goodFeatureNameList = []):
    collection = db['completeUserFeatureSample']
    print("读取数据")
    data = collection.find({}, {'_id':1, 'uid': 1, "gender": 1, featureName: 1})#从mongo中查询这个特征以及对应的性别标签
    data = list(data)

    for line in data:
        line[featureName].update({'gender': line['gender'], 'uid': line['uid']})
    data = list(map(lambda x: x[featureName], data))
    df = pd.DataFrame(data).fillna(0)#每条记录的dict里没有的词语，pd会用nan来占据这个位置的value;没有出现的词语的频率就是0
    df = df[['gender', 'uid'] + goodFeatureNameList]if goodFeatureNameList!=[] else df#如果指定了优质特征，把这些特征提取出来
    meanV = df.mean(axis=0)
    print(meanV)
    data = df.to_dict('records')
    db[featureName].drop()#删除原表
    db[featureName].insert(data, check_keys=False)
    
def queryGramFreqFeatures(featureName='', goodFeatureNameList = []):
    if len(goodFeatureNameList)==0:
        print("没有指定优质特征。")
    #     return None
    goodFeatureNameSet = set(goodFeatureNameList + ['gender', 'uid'])#添加gender,以用于后面删除不好的特征
    collection = db['completeUserFeatureSample']
    data = collection.find({}, {'_id':1, "gender": 1,'uid':1, featureName: 1})#从mongo中查询这个特征以及对应的性别标签
    count = 0
    data = list(data)
    for line in data:
        count += 1
        line[featureName].update({'gender': line['gender'], 'uid': line['uid']})
    data = list(map(lambda x: x[featureName], data))
    if len(goodFeatureNameList)!=0:
        for sample in data:
            for key in list(sample.keys()):
                if key not in goodFeatureNameSet or type(key)==int or key=='$':
                    del sample[key]#删除不是优质特征的条目
    df = pd.DataFrame(data).fillna(0)
    db[featureName].drop()
    data = df.to_dict("records")
    print(featureName, "开始写数据")
    for line in data:
        db[featureName].insert(line, check_keys=False)

def testClassifierStacking(featureNames, scale=True):
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
    dataCursorList = [db['wordFreq'],  db['unigramFreq']]#, db['bigram'],db['postagBigramFreq']]
    uids = dataCursorList[0].find({}, {'uid': 1})
    dataList = []
    count = 0
    featureMap = {}
    for i in range(len(featureNames)):
        featureMap[featureNames[i]] = []
    for uid in uids:
        uid = uid['uid']
        count += 1
        if count == 2000:
            break
        print("正在读取第", count, "个用户。")
        for i in range(len(dataCursorList)):
            cursor = dataCursorList[i]
            line = cursor.find_one({'uid': uid})
            featureMap[featureNames[i]].append(line)
    dfList = []
    Y = None
    yList = []
    for key in featureMap:
        df = pd.DataFrame(featureMap[key])
        dfClean = df.drop(columns=['gender','_id', 'uid'])
        X, Y = dfClean.values, df['gender']
        yList.append(Y)
        if scale:
            X = scala(X)
        dfList.append(X)
    # print(yList[0][:100])
    # print(yList[1][:100])
    inputMap = {"DT": dfList[0], 'KNN': dfList[1]}#, 'LR1': dfList[2], 'LR2': dfList[3]}

    clf = StackingClassifier()
    clf.setBaseModels({
        "DT": LogisticRegression(max_iter=10),#决策树
         'KNN': LogisticRegression(max_iter=10),#最近邻
        #  "LR1": LogisticRegression(max_iter=1000, solver='lbfgs', C=100),
        # "LR2": LogisticRegression(max_iter=1000, solver='lbfgs', C=100)
                                              })
    clf.setMetaModel(LogisticRegression(max_iter=10))
    print("开始交叉验证")
    cmTotal = clf.kFoldValidatoin(inputMap, Y, k=10, classNum=2)
    print((cmTotal[0,0]+ cmTotal[1,1])/(sum(sum(cmTotal))))


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold as kf
from sklearn.ensemble  import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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
    index = kf(n_splits=10, random_state=666).split(list(range(len(Y))))
    cmTotal = np.zeros((2, 2))
    count = 0
    for line in index:
        # clf = LogisticRegression(max_iter=50, solver='newton-cg', C=0.1)#基于词频0.75
        clf = LogisticRegression(max_iter=50, solver='newton-cg', C=0.1, class_weight={0: 0.5, 1: 0.7})#基于词频0.75
        # clf = DecisionTreeClassifier()
        # clf = RandomForestClassifier()
        # clf = MLPClassifier(hidden_layer_sizes=(200,50))
        # clf = SVC(C=0.8)
        trainIndex = line[0]
        testIndex = line[1]
        trainX = X[trainIndex]
        testX = X[testIndex]
        trainy = Y[trainIndex]
        testy = Y[testIndex]
        # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        # featureProcessor = LinearDiscriminantAnalysis(n_components=500)
        # featureProcessor.fit(trainX, trainy)
        # trainX = featureProcessor.transform(trainX)
        # testX = featureProcessor.transform(testX)

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

def decideGenderBySpecialWords(freqMap, maleSpecialWords, femaleSpecialWords):
    words = set(freqMap.keys())
    for word in maleSpecialWords:
        if freqMap[word]>0:
            return 'male'
    for word in femaleSpecialWords:
        if freqMap[word]>0:
            return 'female'
    return "other"

from sklearn.preprocessing import OneHotEncoder
from analysis.algorithm import deepLearning
def testClassifierLSTM(featureCollectionName = "", scale=True):
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
    maleSpecialWords = [ '武器库',  'UFC','硬邦邦的', '龟头', '前臂', '尼玛比']
    maleSpecialWords = set(maleSpecialWords)
    femaleSpecialWords = ['小女子', '小宝贝', "美少年", '萌图', '治愈系', '防晒霜',
                          '萌系']
    femaleSpecialWords = set(femaleSpecialWords)
    data = list(data)#[]
    maleData, femaleData, otherData = [], [], []
    for line in data:
        gender = decideGenderBySpecialWords(line, maleSpecialWords, femaleSpecialWords)
        if gender=='male':
            maleData.append(line)
        elif gender=='female':
            femaleData.append(line)
        else:
            otherData.append(line)
    print("基于特殊词语判断性别，得到男性和女性个数分别是", len(maleData), len(femaleData))
    print("还剩下的用户数是", len(otherData))
    df = pd.DataFrame(otherData)
    dropFeatureNames = []
    import re

    for line in df.columns:
        if len(re.findall('[0-9]', line))>0:
            dropFeatureNames.append(line)

    # dfClean = df.drop(columns=['gender','_id', 'uid'])
    dfClean = df.drop(columns=dropFeatureNames + ['gender','_id', 'uid'])
    X, Y = dfClean.values, df['gender']
    if scale:
        X = scala(X)
    Y = np.array(Y).reshape(-1,1)
    X = X[:,:]
    # Y = list(map(lambda x: [x], Y))
    # Y = np.array(Y).reshape(-1,1)
    oneHotEncoder4Y = OneHotEncoder().fit(Y)
    print("开始交叉验证")
    index = kf(n_splits=10, random_state=666).split(list(range(len(Y))))
    cmTotal = np.zeros((2, 2))
    count = 0
    for line in index:
        num_feature = len(X[0])
        clf = deepLearning.LSTMClassifier(2, num_feature, learning_rate=1e-3,
                                          layer_num=1, hidden_size=100, timestep_size=100)
        clf.initGraph(ifDecrLR=True)
        trainIndex = line[0]
        testIndex = line[1]
        trainX = X[trainIndex]
        testX = X[testIndex]
        trainy = Y[trainIndex]
        testy = Y[testIndex]
        clf.initOneHotEncoder4Y(trainy)
        batch_ys = oneHotEncoder4Y.transform(trainy).todense().astype(np.float32)
        batch_xs = np.array(trainX).astype(np.float32)
        for i in range(50):
            print("这是第", count, "折,第", i, "轮训练。", len(trainy))
            stepsize = 100
            for j in range(0, len(trainy), stepsize):
                batch_ys = clf.oneHotEncode(trainy[j: j + stepsize, :])
                batch_xs = np.array(trainX[j: j + stepsize, :]).astype(np.float32)
                clf.fit(batch_xs, batch_ys)
            print("学习率是", clf.learning_rate, clf.global_step)
            batch_ys = clf.oneHotEncode(trainy)
            batch_xs = np.array(trainX).astype(np.float32)
            pred_train = clf.test(batch_xs, batch_ys)
            pred_train = list(map(lambda x:1 if x[0] < x[1] else 0, pred_train))
            pred_train= np.array(pred_train).reshape(-1, 1)
            # print(len(trainy), len(pred_train))
            cm = confusion_matrix(trainy, pred_train)
            print("训练集混淆矩阵", cm)
            count += 1
            batch_ys = clf.oneHotEncode(testy)
            batch_xs = np.array(testX).astype(np.float32)
            print(count, "测试集表现:")
            y_pred = clf.test(batch_xs, batch_ys)
            y_pred = list(map(lambda x: 1 if x[0] < x[1] else 0, y_pred))
            y_pred = np.array(y_pred).reshape(-1, 1)
            cm = confusion_matrix(testy, y_pred)
            print("测试集混淆矩阵", cm)
            print("召回率是", cm[1,1]/(cm[1,0] + cm[1,1]), "精度是", cm[1,1]/(cm[0,1] + cm[1,1]))
        break

def testClassifierComplexFeatures(featureNames = [], scale=True):
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
    dataCursorList = [db[name] for name in featureNames]#, db['bigram'],db['postagBigramFreq']]
    uids = dataCursorList[0].find({}, {'uid': 1})
    dataList = []
    count = 0
    for uid in uids:
        uid = uid['uid']
        count += 1
        # if count == 200:
        #     break
        print("正在读取第", count, "个用户。")
        dataTemp = {}
        flag = 0
        for i in range(len(dataCursorList)):
            cursor = dataCursorList[i]
            line = cursor.find_one({'uid': uid})
            if line==None:
                flag = 1
                break
            dataTemp.update(line)
        if flag==1:
            continue
        dataList.append(dataTemp)
    df = pd.DataFrame(dataList)
    dfClean = df.drop(columns=['gender','_id', 'uid'])
    X, Y = dfClean.values, df['gender']
    X = scala(X)
    print("开始交叉验证")
    index = kf(n_splits=10, random_state=666).split(list(range(len(Y))))
    cmTotal = np.zeros((2, 2))
    count = 0
    for line in index:
        # clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=100)#基于词频0.75
        # clf = DecisionTreeClassifier()
        # clf = RandomForestClassifier()
        # clf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=666)
        # clf = MLPClassifier(hidden_layer_sizes=(200,10))
        clf = SVC(C=0.9)
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
        print(count,"训练集表现:")
        y_train = clf.predict(trainX)
        cm = confusion_matrix(trainy, y_train)
        print(cm)
        y_pred = clf.predict(testX)
        cm = confusion_matrix(testy, y_pred)
        cmTotal += np.array(cm)
        print(count,"测试集表现：")
        print(cm)
    print(cmTotal)
    print((cmTotal[0,0]+ cmTotal[1,1])/(sum(sum(cmTotal))))

if __name__ == '__main__':
    # querySimpleFeatures(featureName='sentenceLengthFeatures')
    # testClassifier(featureCollectionName = 'sentenceLengthFeatures', scale=False)
    # querySimpleFeatures(featureName='punctuationMarkFreq')
    # testClassifier(featureCollectionName = 'punctuationMarkFreq')
    # querySimpleFeatures(featureName='functoinWordFreq')
    # testClassifier(featureCollectionName = 'functoinWordFreq')#0.62
    s = ""
    with open("goodFeature.txt", 'r') as f:
        s = f.readline()
    # queryGramFreqFeatures(featureName="wordFreq", goodFeatureNameList=s.split('kabukabu'))
    testClassifier(featureCollectionName='wordFreq', scale=True)#0.75
    # s = """あkabukabu三kabukabu专kabukabu东kabukabu个kabukabu中kabukabu临kabukabu为kabukabu乃kabukabu么kabukabu乐kabukabu乖kabukabu买kabukabu了kabukabu事kabukabu交kabukabu享kabukabu亲kabukabu仁kabukabu今kabukabu仓kabukabu仔kabukabu他kabukabu们kabukabu价kabukabu会kabukabu你kabukabu佳kabukabu信kabukabu俺kabukabu偶kabukabu傻kabukabu儿kabukabu兄kabukabu光kabukabu内kabukabu几kabukabu出kabukabu刀kabukabu分kabukabu判kabukabu利kabukabu制kabukabu力kabukabu加kabukabu努kabukabu卜kabukabu卡kabukabu卫kabukabu厄kabukabu去kabukabu又kabukabu友kabukabu双kabukabu叔kabukabu受kabukabu叮kabukabu叶kabukabu各kabukabu合kabukabu君kabukabu听kabukabu呀kabukabu呃kabukabu呆kabukabu员kabukabu呜kabukabu呢kabukabu咔kabukabu咩kabukabu咳kabukabu品kabukabu哇kabukabu哈kabukabu哎kabukabu哒kabukabu哟kabukabu哦kabukabu哭kabukabu哼kabukabu商kabukabu啊kabukabu啦kabukabu啧kabukabu啪kabukabu喂kabukabu喔kabukabu喜kabukabu喵kabukabu嗯kabukabu嗷kabukabu嘛kabukabu嘤kabukabu嘶kabukabu嘻kabukabu噔kabukabu噗kabukabu噢kabukabu回kabukabu团kabukabu囧kabukabu国kabukabu坛kabukabu型kabukabu垫kabukabu外kabukabu天kabukabu太kabukabu头kabukabu奖kabukabu女kabukabu她kabukabu好kabukabu妹kabukabu姐kabukabu姑kabukabu姚kabukabu娃kabukabu娘kabukabu婚kabukabu媒kabukabu嫂kabukabu子kabukabu存kabukabu孩kabukabu守kabukabu官kabukabu宝kabukabu实kabukabu宪kabukabu家kabukabu射kabukabu小kabukabu少kabukabu崴kabukabu差kabukabu巴kabukabu市kabukabu帅kabukabu希kabukabu帖kabukabu席kabukabu帮kabukabu幸kabukabu库kabukabu应kabukabu店kabukabu座kabukabu建kabukabu开kabukabu弟kabukabu张kabukabu弹kabukabu强kabukabu微kabukabu心kabukabu快kabukabu恋kabukabu恤kabukabu恭kabukabu息kabukabu您kabukabu情kabukabu惠kabukabu想kabukabu愉kabukabu我kabukabu战kabukabu所kabukabu手kabukabu打kabukabu技kabukabu投kabukabu抗kabukabu折kabukabu报kabukabu抱kabukabu拆kabukabu拜kabukabu持kabukabu挎kabukabu挡kabukabu换kabukabu掌kabukabu接kabukabu控kabukabu援kabukabu摸kabukabu撸kabukabu攒kabukabu支kabukabu收kabukabu攻kabukabu故kabukabu文kabukabu斜kabukabu新kabukabu日kabukabu旺kabukabu易kabukabu曼kabukabu朋kabukabu望kabukabu木kabukabu术kabukabu机kabukabu杀kabukabu李kabukabu村kabukabu来kabukabu杯kabukabu柜kabukabu框kabukabu梅kabukabu欢kabukabu款kabukabu正kabukabu此kabukabu步kabukabu比kabukabu气kabukabu汉kabukabu江kabukabu油kabukabu治kabukabu波kabukabu注kabukabu泪kabukabu淘kabukabu渣kabukabu满kabukabu火kabukabu炜kabukabu然kabukabu煤kabukabu照kabukabu爱kabukabu爽kabukabu片kabukabu牙kabukabu犯kabukabu猛kabukabu王kabukabu玩kabukabu球kabukabu瓦kabukabu生kabukabu用kabukabu男kabukabu皇kabukabu盒kabukabu盖kabukabu看kabukabu真kabukabu眼kabukabu着kabukabu睡kabukabu矿kabukabu磨kabukabu示kabukabu祝kabukabu神kabukabu秀kabukabu秋kabukabu秒kabukabu穆kabukabu穿kabukabu突kabukabu站kabukabu童kabukabu笑kabukabu算kabukabu箭kabukabu篮kabukabu粉kabukabu粑kabukabu红kabukabu级kabukabu纳kabukabu纸kabukabu线kabukabu练kabukabu终kabukabu绑kabukabu绒kabukabu给kabukabu绣kabukabu继kabukabu维kabukabu网kabukabu置kabukabu群kabukabu翻kabukabu育kabukabu肿kabukabu胎kabukabu能kabukabu脚kabukabu脸kabukabu色kabukabu节kabukabu苦kabukabu莲kabukabu菜kabukabu萌kabukabu营kabukabu萨kabukabu蜜kabukabu街kabukabu衫kabukabu袜kabukabu裤kabukabu规kabukabu言kabukabu謝kabukabu访kabukabu译kabukabu询kabukabu语kabukabu请kabukabu诺kabukabu谢kabukabu豆kabukabu贝kabukabu账kabukabu贸kabukabu费kabukabu赛kabukabu跳kabukabu踏kabukabu身kabukabu车kabukabu转kabukabu辛kabukabu辰kabukabu迎kabukabu进kabukabu迷kabukabu选kabukabu通kabukabu速kabukabu逼kabukabu郅kabukabu配kabukabu采kabukabu量kabukabu鉴kabukabu钱kabukabu银kabukabu铺kabukabu链kabukabu锋kabukabu错kabukabu长kabukabu闹kabukabu闻kabukabu防kabukabu雁kabukabu鞋kabukabu顶kabukabu饰kabukabu马kabukabu驭kabukabu验kabukabu高kabukabu鳖kabukabu鸟kabukabu鸭kabukabu麦kabukabu黑kabukabu默kabukabu龙kabukabu﹏kabukabu﹐kabukabu！kabukabu（kabukabu）kabukabu，kabukabu：kabukabu；kabukabu＝kabukabu？kabukabuＣkabukabuＨkabukabu＿kabukabu～kabukabu￣kabukabu￥kabukabu�"""
    # queryGramFreqFeatures(featureName="unigramFreq", goodFeatureNameList=s.split('kabukabu'))
    # testClassifier(featureCollectionName='unigramFreq', scale=True)
    # s = """一双kabukabu一起kabukabu下本kabukabu不错kabukabu专柜kabukabu中体kabukabu中国kabukabu丹1kabukabu为专kabukabu主　kabukabu么么kabukabu了=kabukabu了…kabukabu交易kabukabu亲们kabukabu亲如kabukabu仓处kabukabu仔裤kabukabu代牛kabukabu们请kabukabu价优kabukabu优惠kabukabu位Hkabukabu低帮kabukabu你hkabukabu你通kabukabu佳佳kabukabu佳的kabukabu做专kabukabu兄弟kabukabu光临kabukabu全了kabukabu内线kabukabu出扣kabukabu分享kabukabu利经kabukabu加油kabukabu动作kabukabu北卡kabukabu单飞kabukabu卡路kabukabu原价kabukabu只为kabukabu可爱kabukabu右）kabukabu各大kabukabu君亲kabukabu品均kabukabu品折kabukabu哈~kabukabu哈哈kabukabu哥们kabukabu哦，kabukabu商场kabukabu啊~kabukabu啊…kabukabu啊啊kabukabu啦~kabukabu啦啦kabukabu啪啪kabukabu喜欢kabukabu嘻嘻kabukabu噔噔kabukabu在细kabukabu场库kabukabu均为kabukabu坛 kabukabu外线kabukabu多哦kabukabu大论kabukabu太全kabukabu头哥kabukabu女子kabukabu女生kabukabu好好kabukabu好鞋kabukabu如故kabukabu姚明kabukabu存折kabukabu定假kabukabu宝①kabukabu实战kabukabu对喜kabukabu小女kabukabu帮 kabukabu帮顶kabukabu常年kabukabu年做kabukabu库存kabukabu店 kabukabu店Qkabukabu店主kabukabu店大kabukabu店所kabukabu店：kabukabu度在kabukabu开 kabukabu开心kabukabu快乐kabukabu您好kabukabu您：kabukabu惠 kabukabu我们kabukabu扑ikabukabu扑验kabukabu打球kabukabu扣 kabukabu扣费kabukabu投篮kabukabu折左kabukabu折扣kabukabu拉利kabukabu拜仁kabukabu持各kabukabu支持kabukabu数鞋kabukabu新开kabukabu日 kabukabu旺：kabukabu明Hkabukabu是清kabukabu有商kabukabu服饰kabukabu木有kabukabu本店kabukabu柜 kabukabu柜正kabukabu欢迎kabukabu正品kabukabu油！kabukabu法拉kabukabu清仓kabukabu火箭kabukabu灰北kabukabu牛 kabukabu牛仔kabukabu牛逼kabukabu生日kabukabu白送kabukabu的微kabukabu的白kabukabu看库kabukabu码2kabukabu神马kabukabu秒杀kabukabu突破kabukabu等等kabukabu细微kabukabu绑定kabukabu群：kabukabu老公kabukabu育店kabukabu能力kabukabu营 kabukabu行给kabukabu装单kabukabu裤 kabukabu请放kabukabu账1kabukabu费 kabukabu路里kabukabu身体kabukabu车手kabukabu车队kabukabu转出kabukabu转账kabukabu辛苦kabukabu过银kabukabu迎你kabukabu迎光kabukabu迎关kabukabu迎您kabukabu运球kabukabu这鞋kabukabu进攻kabukabu送本kabukabu通过kabukabu邮 kabukabu配色kabukabu采访kabukabu里 kabukabu鉴定kabukabu银行kabukabu铺新kabukabu长牛kabukabu防守kabukabu零售kabukabu鞋，kabukabu顶 kabukabu顶顶kabukabu顶！kabukabu飞 kabukabu验货kabukabu！！kabukabu（4kabukabu） kabukabu，亲kabukabu，码kabukabu，鉴kabukabu：1kabukabu：hkabukabu＝＝kabukabuＨＣkabukabu～～"""
    # queryGramFreqFeatures(featureName="bigram", goodFeatureNameList=s.split('kabukabu'))
    # testClassifier(featureCollectionName='bigram', scale=True)
    # s = """Mg_wkabukabua_akabukabua_agkabukabua_ankabukabua_ekabukabua_nkabukabua_ngkabukabua_nntkabukabua_nxkabukabua_nzkabukabua_okabukabua_qkabukabua_rrkabukabua_ude2kabukabua_ulekabukabua_vkabukabua_vikabukabua_wkabukabua_ykabukabuad_nxkabukabuad_qkabukabuad_ude1kabukabuad_vshikabukabuag_dkabukabuag_nkabukabuag_nrfkabukabuag_nskabukabuag_nzkabukabuag_qvkabukabuag_rrkabukabual_qtkabukabuan_kkabukabuan_rrkabukabub_Mgkabukabub_dgkabukabub_mkabukabub_nkabukabub_nrkabukabub_nxkabukabub_vkabukabub_wkabukabubl_blkabukabuc_mkabukabuc_nkabukabuc_nxkabukabuc_rrkabukabuc_vnkabukabucc_nkabukabucc_nrkabukabucc_nxkabukabucc_rrkabukabucc_skabukabucc_vnkabukabud_akabukabud_ckabukabud_dkabukabud_mkabukabud_nkabukabud_nzkabukabud_pkabukabud_qkabukabud_rrkabukabud_rzkabukabud_ukabukabud_ude2kabukabud_ulekabukabud_vkabukabud_vikabukabud_vlkabukabud_zkabukabudg_dkabukabudg_vkabukabue_bkabukabue_dkabukabue_ekabukabue_fkabukabue_nxkabukabue_nzkabukabue_okabukabue_vkabukabue_wkabukabue_ykabukabuf_fkabukabuf_mkabukabuf_nkabukabuf_nxkabukabuf_qvkabukabuf_rrkabukabuf_usuokabukabuf_wkabukabugb_zkabukabugi_gikabukabugi_nxkabukabugi_nzkabukabugi_ulekabukabugi_wkabukabuk_dkabukabuk_nndkabukabuk_pkabukabuk_vkabukabum_bkabukabum_ckabukabum_dkabukabum_fkabukabum_mkabukabum_mqkabukabum_nkabukabum_nrkabukabum_nrfkabukabum_nskabukabum_nsfkabukabum_nxkabukabum_pkabukabum_pbeikabukabum_qtkabukabum_rzkabukabum_tkabukabum_tgkabukabum_ude1kabukabum_ude2kabukabum_udhkabukabum_vfkabukabum_vgkabukabum_vikabukabum_vshikabukabum_wkabukabumq_fkabukabumq_mqkabukabun_adkabukabun_bkabukabun_ckabukabun_cckabukabun_ekabukabun_gikabukabun_kkabukabun_nfkabukabun_ngkabukabun_niskabukabun_nntkabukabun_nrfkabukabun_nsfkabukabun_nxkabukabun_okabukabun_rrkabukabun_ude1kabukabun_ude2kabukabun_vgkabukabun_vikabukabun_vshikabukabun_vyoukabukabun_xkabukabun_ykabukabunf_nkabukabunf_nfkabukabunf_nxkabukabunf_ude1kabukabunf_vikabukabung_bkabukabung_nkabukabung_ngkabukabung_nxkabukabung_pkabukabung_pbeikabukabung_rrkabukabung_vikabukabung_vyoukabukabung_wkabukabunis_nkabukabunis_ngkabukabunis_nxkabukabunis_pkabukabunis_uzhikabukabunis_wkabukabunnd_akabukabunnd_qkabukabunnt_kkabukabunnt_mkabukabunnt_ngkabukabunnt_nrfkabukabunnt_nxkabukabunnt_ude1kabukabunr_adkabukabunr_cckabukabunr_ekabukabunr_nxkabukabunr_rrkabukabunr_ude1kabukabunr_vkabukabunr_vgkabukabunr_wkabukabunrf_ntkabukabunrf_nxkabukabunrf_nzkabukabuns_adkabukabuns_dkabukabuns_nkabukabuns_ngkabukabuns_nntkabukabuns_nzkabukabuns_qkabukabuns_qvkabukabuns_ude1kabukabunsf_mkabukabunsf_nkabukabunsf_ngkabukabunsf_niskabukabunsf_ntkabukabunsf_nxkabukabunsf_ude1kabukabunsf_vikabukabunsf_vnkabukabunsf_wkabukabunt_dkabukabunt_mkabukabunt_nkabukabunt_nrfkabukabunt_nxkabukabunt_nzkabukabunt_pkabukabunt_vkabukabunt_vikabukabunt_wkabukabuntc_nkabukabunx_ckabukabunx_ekabukabunx_fkabukabunx_mkabukabunx_nkabukabunx_ngkabukabunx_nrkabukabunx_nrfkabukabunx_nsfkabukabunx_ntckabukabunx_nxkabukabunx_nzkabukabunx_okabukabunx_pkabukabunx_rrkabukabunx_ude1kabukabunx_vkabukabunx_vikabukabunx_wkabukabunz_akabukabunz_ckabukabunz_niskabukabunz_nndkabukabunz_nxkabukabunz_nzkabukabunz_okabukabunz_rrkabukabunz_ude1kabukabunz_ude2kabukabunz_ulekabukabunz_vnkabukabunz_ykabukabuo_akabukabuo_dkabukabuo_ekabukabuo_nkabukabuo_nxkabukabuo_pkabukabuo_rrkabukabuo_vkabukabuo_wkabukabup_akabukabup_nkabukabup_ngkabukabup_niskabukabup_nsfkabukabup_ntkabukabup_qkabukabup_rrkabukabup_rzskabukabup_skabukabup_wkabukabupbei_nrkabukabupbei_vikabukabuq_mkabukabuq_nxkabukabuq_qkabukabuq_vkabukabuq_wkabukabuqt_mkabukabuqt_wkabukabuqv_bkabukabur_dkabukaburr_akabukaburr_agkabukaburr_bkabukaburr_ckabukaburr_cckabukaburr_dkabukaburr_fkabukaburr_kkabukaburr_nkabukaburr_nfkabukaburr_ngkabukaburr_nrkabukaburr_nxkabukaburr_nzkabukaburr_pkabukaburr_qkabukaburr_rzkabukaburr_skabukaburr_tkabukaburr_ukabukaburr_ude1kabukaburr_uguokabukaburr_ulekabukaburr_ulskabukaburr_vkabukaburr_vfkabukaburr_vgkabukaburr_vikabukaburr_vlkabukaburr_vshikabukaburr_wkabukaburr_ykabukabury_ude1kabukaburyv_ulekabukaburz_akabukaburz_mkabukaburz_nkabukaburz_wkabukaburzs_mkabukaburzs_ngkabukaburzs_qkabukaburzv_nkabukaburzv_qkabukaburzv_rykabukabus_fkabukabus_niskabukabus_vkabukabut_dkabukabut_mkabukabut_nsfkabukabut_rrkabukabut_tkabukabut_wkabukabutg_mkabukabutg_nxkabukabutg_vkabukabuu_bkabukabuu_mkabukabuude1_ekabukabuude1_fkabukabuude1_mkabukabuude1_nrkabukabuude1_nskabukabuude1_nsfkabukabuude1_okabukabuude1_rrkabukabuude1_ude1kabukabuude1_vfkabukabuude1_vnkabukabuude1_ykabukabuude2_pkabukabuude2_vkabukabuude3_dkabukabuude3_ude3kabukabuudeng_rrkabukabuudeng_ude2kabukabuudeng_udengkabukabuudeng_uzhekabukabuudh_ckabukabuuguo_nxkabukabuule_dkabukabuule_ekabukabuule_nxkabukabuule_nzkabukabuule_okabukabuule_rrkabukabuule_vfkabukabuule_wkabukabuule_ykabukabuuls_wkabukabuuyy_wkabukabuuzhe_nkabukabuuzhe_rrkabukabuuzhe_wkabukabuuzhi_rzkabukabuv_ekabukabuv_fkabukabuv_kkabukabuv_mkabukabuv_nkabukabuv_nrkabukabuv_nskabukabuv_nsfkabukabuv_ntkabukabuv_nxkabukabuv_okabukabuv_qkabukabuv_qvkabukabuv_rrkabukabuv_rzkabukabuv_rzskabukabuv_rzvkabukabuv_tgkabukabuv_ude3kabukabuv_ulekabukabuv_usuokabukabuv_uzhekabukabuv_vikabukabuv_ykabukabuvf_mkabukabuvf_nsfkabukabuvf_vkabukabuvf_vnkabukabuvf_wkabukabuvf_ykabukabuvg_wkabukabuvi_akabukabuvi_ekabukabuvi_fkabukabuvi_nsfkabukabuvi_ntkabukabuvi_nxkabukabuvi_okabukabuvi_rrkabukabuvi_ulekabukabuvi_vkabukabuvi_vikabukabuvi_vnkabukabuvi_wkabukabuvi_ykabukabuvl_ude1kabukabuvl_ykabukabuvn_adkabukabuvn_bkabukabuvn_cckabukabuvn_dkabukabuvn_ekabukabuvn_kkabukabuvn_mkabukabuvn_nkabukabuvn_ngkabukabuvn_nndkabukabuvn_nntkabukabuvn_qtkabukabuvn_rrkabukabuvn_wkabukabuvn_ykabukabuvshi_mkabukabuvshi_nkabukabuvshi_rrkabukabuvshi_vikabukabuvshi_vnkabukabuvyou_bkabukabuvyou_mkabukabuvyou_nxkabukabuw_akabukabuw_agkabukabuw_ankabukabuw_ckabukabuw_cckabukabuw_dgkabukabuw_ekabukabuw_fkabukabuw_gikabukabuw_mkabukabuw_nkabukabuw_nfkabukabuw_ngkabukabuw_niskabukabuw_nrkabukabuw_nsfkabukabuw_ntkabukabuw_nxkabukabuw_nzkabukabuw_okabukabuw_rrkabukabuw_rzskabukabuw_tgkabukabuw_usuokabukabuw_vfkabukabuw_wkabukabuw_zkabukabux_nrkabukabux_ude1kabukabux_wkabukabux_xkabukabuy_akabukabuy_ckabukabuy_dkabukabuy_ekabukabuy_nkabukabuy_ngkabukabuy_nxkabukabuy_nzkabukabuy_okabukabuy_rrkabukabuy_vikabukabuy_vshikabukabuy_wkabukabuy_ykabukabuz_akabukabuz_nkabukabuz_ude1kabukabuz_ude2"""
    # queryGramFreqFeatures(featureName="postagBigramFreq", goodFeatureNameList=s.split('kabukabu'))
    # testClassifier(featureCollectionName='postagBigramFreq')

    # testClassifierStacking(['wordFreq', 'punctuationMarkFreq'])#,  'wordFreq'])
    # testClassifierComplexFeatures(['wordFreq'])
    # testClassifierLSTM(featureCollectionName='wordFreq', scale=True)#0.77