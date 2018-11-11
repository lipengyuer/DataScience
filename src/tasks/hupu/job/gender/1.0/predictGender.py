#用机器学习算法预测用户的性别
import numpy as np
import pandas as pd
import sklearn as sk
from matplotlib import  pyplot as plt
from pymongo import MongoClient
from hupu.analisys.config import enviroment
from sklearn.linear_model import LogisticRegressionCV,LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BaseNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble  import VotingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.cross_validation import KFold as kf
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import random
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
#获取用户的统计数据

def getConnectionMongo():
    conn = MongoClient(enviroment.MONGO_IP, 27017)
    return conn

class genderDetection():

    def __init__(self):
        self.stasticsData = None
        self.conn = self.getConnectionMongo()
        self.featureWords = []#用于构建特征的词语
        self.featurePos = []
        self.featureChars = []
        self.stopWords = set({})

    def initStopWords(self, path):
        with open(path, 'r', encoding="utf8") as f:
            lines = f.readlines()
            lines = list(map(lambda x: x.replace("\n", ""), lines))
        self.stopWords = set(lines)

    def getConnectionMongo(self):
        conn = MongoClient(enviroment.MONGO_IP, 27017)
        return conn

    def processIt(self, aDict):
        aDict['detail']['uid'] = aDict['_id']
        x = aDict['detail']
        res = {}
        for key in x:
            res[key] = x[key]
        res['replyAve'] = res['replyNumber']/(res['advocatePostNumber'] + 0.0000001)
        res['remainIndex'] = res['footPrintNumDetail'] /(res['footPrintNumMainBlock'] + 0.0000001)
        res['lifeIndex'] = (res['loveIndex'] + 1) * (res['parentIndex'] + 1)
        res['宅男指数'] = (res['宅'] + 1) * (res['学习不好'] + 1) * (res['游戏'] + 1)
        return res

    def getStasticsData(self):
        conn = getConnectionMongo()
        db = conn.hupu  # 连接mydb数据库，没有则自动创建
        # collection = db.userStastics  # 使用test_set集合，没有则自动创建
        collection = db.userFollStastics
        data = collection.find({}, {"detail": 1}).limit(50000)
        data = map(lambda x: {"uid": x["_id"], "feature": x['detail']['follContentFeatrue']}, data)
        # data = map(lambda x: {"uid": x["_id"], "feature": x['detail']['advContentFeatrue']}, data)
        data = filter(lambda x: len(x["feature"])> 100, data)#说话过少的人不处理
        data = list(data)
        data = pd.DataFrame(data)
        pickle.dump(data, open('data.pkl', 'wb'))
        data = pickle.load(open('data.pkl', 'rb'))
        self.stasticsData = data.reset_index(drop=True)

    def addGender(self):
        db = self.conn.hupu  # 连接mydb数据库，没有则自动创建
        collection = db.hupuUserInfo  # 使用test_set集合，没有则自动创建
        uids = list(self.stasticsData["uid"].values)
        # print(uids)
        # 查询用户资料数据
        genders = collection.find({'uid': {'$in': uids}}, {'uid': 1, 'gender': 1})
        genders = list(genders)
        # 提取uid和gender两个字段
        genders = filter(lambda x: 'gender' in x, genders)
        # 滤掉没有性别的用户
        genders = filter(lambda x: x['gender'] != 'NaN', genders)
        genders = list(genders)
        genders = pd.DataFrame(genders)
        genders = genders.reset_index(drop=True)
        self.stasticsData = pd.merge(self.stasticsData, genders, on=['uid'], how="inner")
        maleData = self.stasticsData.loc[self.stasticsData['gender'] == 'm']
        femaleData = self.stasticsData.loc[self.stasticsData['gender'] == 'f']
        maleData = maleData.sample(n=len(femaleData))
        self.stasticsData = femaleData.append(maleData)
        self.stasticsData = self.stasticsData.sample(frac=1)

    def updateWordFreq(self, resultWordFreqMap, aWordFreList):
        # 更新词频map里的词频。由于是浅拷贝，会修改引用对象的取值。这样做的好处是免去了新建一个对象的成本。
        for line in aWordFreList:
            try:
                word, pos = line['word'].split("/")
            except:
                continue
            # print(word, self.stopWords)
            import re
            if word in self.stopWords or len(re.findall("[0-9]", word))>0:
                # print("挺用词", word)
                continue
            num = int(line["freq"])
            if word in resultWordFreqMap:
                resultWordFreqMap[word] += num
            else:
                resultWordFreqMap[word] = num
        return resultWordFreqMap

    def getCharFreq(self, wordFreqList):
        def splitWordIntoChars(aUserWordFreqList):
            # 把词频map里的字符频率统计出来，形成一个新的map
            charFreqList = []
            for wordFreq in aUserWordFreqList:
                try:
                    word, pos = wordFreq['word'].split("/")
                except:
                    continue
                if word in self.stopWords:
                    continue
                freq = int(wordFreq['freq'])
                for char in word:
                    charFreqList.append({ 'word':char + "/" + pos, "freq": freq})
            return charFreqList

        res = {}
        for line in wordFreqList:
            charFreqList = splitWordIntoChars(line)  # 借用统计词频的函数，统计这个字符频率map里的字符频率
            self.updateWordFreq(res, charFreqList)
        freqSorted = sorted(res.items(), key=lambda x: x[1], reverse=True)
        return freqSorted

    def getPosFreq(self, wordFreqList):
        def splitWordIntoChars(aUserWordFreqList):
            # 把词频map里的字符频率统计出来，形成一个新的map
            posFreqList = []
            for wordFreq in aUserWordFreqList:
                try:
                    word, pos = wordFreq['word'].split("/")
                except:
                    continue
                if word in self.stopWords:
                    continue
                freq = int(wordFreq['freq'])
                posFreqList.append({ 'word': pos + "/" + pos, "freq": freq})
            return posFreqList

        res = {}
        for line in wordFreqList:
            posFreqList = splitWordIntoChars(line)  # 借用统计词频的函数，统计这个字符频率map里的字符频率
            self.updateWordFreq(res, posFreqList)
        freqSorted = sorted(res.items(), key=lambda x: x[1], reverse=True)
        return freqSorted

    def showTopWordsInUserName(self):
        N = 10000
        femaleData = self.stasticsData.loc[self.stasticsData['gender'] == 'f']
        maleData = self.stasticsData.loc[self.stasticsData['gender'] == 'm']
        print("男性的数量是", len(maleData), ". 女性的数量是", len(femaleData))
        femaleData, maleData = femaleData['feature'], maleData['feature']
        femaleTopWords, maleTopWords = {}, {}
        for wordFreqList in femaleData:
            femaleTopWords = self.updateWordFreq(femaleTopWords, wordFreqList)
        femaleTopWords = sorted(femaleTopWords.items(), key=lambda x: x[1], reverse=True)
        def get1st(x):
            return map(lambda y: y[0], x)
        print("女性用户的高频字符是", " ".join(get1st(self.getCharFreq(femaleData)[:N])))
        print("女性用户名的高频词是", " ".join(get1st(femaleTopWords[:N])))
        print("女性用户的高频词性是", " ".join(get1st(self.getPosFreq(femaleData)[:N])))
        for wordFreqList in maleData:
            maleTopWords = self.updateWordFreq(maleTopWords, wordFreqList)
        maleTopWords = sorted(maleTopWords.items(), key=lambda x: x[1], reverse=True)
        print("男性用户的高频字符是", " ".join(get1st(self.getCharFreq(maleData)[:N])))
        print("男性用户名的高频词是", " ".join(get1st(maleTopWords[:N])))
        print("男性用户的高频词性是"," ".join(get1st( self.getPosFreq(maleData)[:N])))

    def loadFeatureWords(self, path):
        import re
        N_word, N_pos, N_char = 500, 10, 10
        with open(path, 'r', encoding="utf8") as f:
            lines = f.readlines()
            lines = list(map(lambda x: x.replace("\n", "").split('是')[1][1:], lines))
        words = re.findall("\'.+?\'",lines[0] + lines[3])
        words = list(map(lambda x: x.replace('\'', ''), words))[:N_word]
        posTags = re.findall("\'.+?\'", lines[1] + lines[4])
        posTags = list(map(lambda x: x.replace('\'', ''), posTags))[:N_pos]
        chars = re.findall("\'.+?\'", lines[2] + lines[5])
        chars = list(map(lambda x: x.replace('\'', ''), chars))[:N_char]
        self.featureWords = words#用于构建特征的词语
        self.featurePos = posTags
        self.featureChars = chars

    def featureExtraction(self):
        def initACleanMap(keyList):
            tempMap = {}
            for key in keyList:
                tempMap[key] = 0
            return tempMap
        def normallize(feature):
            feature = list(map(lambda x: x/(np.median(x) + 0.0000001), feature))
            return np.array(feature)

        data = self.stasticsData['feature']
        featureWords, featurePos, featureChars = [], [], []
        for line in data:
            wordFreqMap = initACleanMap(self.featureWords)
            posTagFreqMap = initACleanMap(self.featurePos)
            charFreqMap = initACleanMap(self.featureChars)
            for aWordFreq in line:
                try:
                    word, posTag = aWordFreq['word'].split("/")
                except:
                    continue
                freq = int(aWordFreq['freq'])
                if word in wordFreqMap:
                    wordFreqMap[word] += freq
                if posTag in posTagFreqMap:
                    posTagFreqMap[posTag] += freq
                for char in word:
                    if char in charFreqMap:
                        charFreqMap[char] += freq
            featureWords.append(wordFreqMap)
            featurePos.append(posTagFreqMap)
            featureChars.append(charFreqMap)
        featureWords, featurePos, featureChars = pd.DataFrame(featureWords),\
                                                pd.DataFrame(featurePos),\
                                                pd.DataFrame(featureChars)
        featureWords, featurePos, featureChars = normallize(featureWords.values), \
                                                 normallize(featurePos.values), \
                                                 normallize(featureChars.values)
        self.features = np.concatenate((featureWords, featurePos, \
                                   featureChars), axis=1)
        from numpy.matlib import repmat
        print(np.max(self.features, axis=1))
        self.features = list(map(lambda x: list(map(lambda y: 500 if y>500 else y, x)), self.features))
        self.features = np.array(self.features)
        self.labels = self.stasticsData['gender'].values
        print(self.features)

    def crossValidation(self):
        kf = KFold(n_splits=10)
        inputData = kf.split(self.features)
        count = 1
        totalCM = np.zeros([2,2])
        for train_index, test_index in inputData:
            # clf = DecisionTreeClassifier(max_depth=10)
            # clf = RandomForestClassifier(n_estimators=10, max_depth=5,random_state=666)
            # clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10))
            # clf = MLPClassifier(max_iter=200, hidden_layer_sizes=(200, 20))
            # clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=100)
            # clf = GradientBoostingClassifier(n_estimators=20)
            # clf = SVC(C=0.8)
            clfList = [['desisionTree', DecisionTreeClassifier()],
                       ['mlp', MLPClassifier()],
                       ['KNN', KNeighborsClassifier(n_neighbors=10)]]
            clf = VotingClassifier(clfList, voting='hard')
            X_train, X_test = self.features[train_index], self.features[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]
            # featureProcessor = PCA(n_components=20)
            featureProcessor = LinearDiscriminantAnalysis(n_components=50).fit(X_train, y_train)
            # featureProcessor = SelectKBest(chi2, k=20)#.fit(features, labels)
            featureProcessor.fit(X_train, y_train)
            X_train = featureProcessor.transform(X_train)
            X_test = featureProcessor.transform(X_test)

            clf.fit(X_train, y_train)
            pred_train = clf.predict(X_train)
            y_train = list(y_train)
            cm = confusion_matrix(y_train, pred_train)
            print("第", count, "轮训练集里的表现\n", cm)
            pred = clf.predict(X_test)
            y_test = list(y_test)
            cm = confusion_matrix(y_test, pred, labels=['m', 'f'])
            totalCM = totalCM + np.array(cm)
            print("第", count, "轮测试集里的表现\n", cm)
            res = list(map(lambda x: y_test[x] + '_' + pred[x], range(len(pred))))
            print(res)
            # print(X_test)
            print("####################################")
            count += 1
        print("混淆矩阵的和是\n", totalCM, "准确率是",
              (totalCM[0][0] + totalCM[1][1])/(sum(sum(totalCM))))


def classIt():
    t1 = time.time()
    genderAnnalysis = genderDetection()
    genderAnnalysis.initStopWords(r"C:\Users\Administrator\PycharmProjects\test\hupu\analisys\research\gender\HIT_stop_words.txt")
    genderAnnalysis.getStasticsData()
    genderAnnalysis.addGender()
    genderAnnalysis.showTopWordsInUserName()
    t2 = time.time()
    print("计算的耗时是", int(t2 - t1))
    genderAnnalysis.loadFeatureWords(r"C:\Users\Administrator\PycharmProjects\test\hupu\analisys\research\gender\genderFeature.txt")
    genderAnnalysis.featureExtraction()
    genderAnnalysis.crossValidation()
    # print(genderAnnalysis.stasticsData)
from matplotlib import pyplot as plt
def DE():
    genderAnnalysis = genderDetection()
    genderAnnalysis.initStopWords(
        r"C:\Users\Administrator\PycharmProjects\test\hupu\analisys\research\gender\HIT_stop_words.txt")
    genderAnnalysis.getStasticsData()
    print("开始添加性别。")
    genderAnnalysis.addGender()
    genderAnnalysis.showTopWordsInUserName()
    genderAnnalysis.loadFeatureWords(r"C:\Users\Administrator\PycharmProjects\test\hupu\analisys\research\gender\genderFeature.txt")
    genderAnnalysis.featureExtraction()
    data = genderAnnalysis.features
    plt.boxplot(data)
    plt.show()

import pickle
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
import time
if __name__ == '__main__':
    # DE()
    classIt()



