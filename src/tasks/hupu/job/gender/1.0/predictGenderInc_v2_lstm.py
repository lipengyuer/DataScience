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

from hupu.analisys.algorithm import deepLearning

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
        self.N_word, self.N_pos, self.N_char = 2000, 20, 2000
        self.count = 20000
        self.goodFeatures = None
        with open("优质特征.txt", 'r', encoding="utf8") as f:
            line = f.readlines()
            line = "".join(line)
            self.goodFeatures = line[2:-2].replace("\n", '').split('\' \'')
            self.goodFeatures = set(self.goodFeatures)

    def initStopWords(self, path):
        with open(path, 'r', encoding="utf8") as f:
            lines = f.readlines()
            lines = list(map(lambda x: x.replace("\n", ""), lines))
        self.stopWords = set(lines)

    def getConnectionMongo(self):
        conn = MongoClient(enviroment.MONGO_IP, 27017)
        return conn

    def getStasticsData(self):
        #'''
        conn = getConnectionMongo()
        db = conn.hupu  # 连接mydb数据库，没有则自动创建
        # collection = db.userStastics  # 使用test_set集合，没有则自动创建
        collection = db.userFollStastics
        data = collection.find({}, {"detail": 1}).batch_size(1000)
        count = 0
        self.stasticsData = []
        stasticsUID = []
        for line in data:
            uid = line['_id']
            thisData = line['detail']['follContentFeatrue']
            if len(thisData)>100:
                wcpMap = self.featureExtractionFromWords(thisData)
                wcpMapTemp = {}
                # print(wcpMap)
                for key in wcpMap:
                    if key in self.goodFeatures:
                        wcpMapTemp[key] = wcpMap[key]
                # print(wcpMapTemp)
                wcpMap = {'uid': uid, **wcpMapTemp}
                print(count, len(wcpMap))
                count += 1
                self.stasticsData.append(wcpMap)
            if count == self.count:
                break
        data = pd.DataFrame(self.stasticsData)
        pickle.dump(data, open('data.pkl', 'wb'))
        #'''
        data = pickle.load(open(r'data.pkl', 'rb'))
        data = data.fillna(data.median())
        # print(data.isnull())
        self.stasticsData = data.reset_index(drop=True)

    def featureExtractionFromWords(self, aWordFreqMap):
        def initACleanMap(keyList, addDtr = ""):
            tempMap = {}
            for key in keyList:
                tempMap[addDtr + key] = 0
            return tempMap

        line = aWordFreqMap
        wordFreqMap = initACleanMap(self.featureWords, addDtr='word_')
        posTagFreqMap = initACleanMap(self.featurePos, addDtr='pos_')
        charFreqMap = initACleanMap(self.featureChars, addDtr='char_')
        # print(self.featureWords)
        # print(self.featureChars)

        for aWordFreq in line:
            try:
                word, posTag = aWordFreq['word'].split("/")
            except:
                continue
            freq = int(aWordFreq['freq'])
            #
            if "word_" + word in wordFreqMap:
                wordFreqMap["word_" + word] += freq

            # print(wordFreqMap)
            if "pos_" + posTag in posTagFreqMap:
                posTagFreqMap[ "pos_" + posTag] += freq

            for char in word:
                if "char_" + char in charFreqMap:
                    charFreqMap["char_" +char] += freq
        features = {**wordFreqMap, **charFreqMap, **posTagFreqMap}
        # print(features)
        return features

    def loadFeatureWords(self, path):
        import re
        N_word, N_pos, N_char = self.N_word, self.N_pos,self.N_char
        with open(path, 'r', encoding="utf8") as f:
            lines = f.readlines()
            lines = list(map(lambda x: x.replace("\n", ""), lines))
        words = lines[1].split(" ")[1:int(0.5*N_word)] + \
                lines[4].split(" ")[1:N_word-int(0.5*N_word)]
        print("特征词的数量是", len(words), N_word, len(lines[1].split(" ")), lines[1])
        words = list(map(lambda x: x.replace('\'', ''), words))
        posTags =lines[2].split(" ")[1:int(0.5*N_pos)] + \
                 lines[5].split(" ")[1:N_pos-int(0.5*N_pos)]
        posTags = list(map(lambda x: x.replace('\'', ''), posTags))
        chars = lines[0].split(" ")[1:int(0.5*N_char)] + \
                lines[3].split(" ")[1:N_char-int(0.5*N_char)]
        chars = list(map(lambda x: x.replace('\'', ''), chars))
        self.featureWords = list(sorted(set(words)))#用于构建特征的词语
        self.featurePos = list(sorted(set(posTags) - set(words)))
        self.featureChars = list(sorted(set(chars) - set(words) - set(posTags)))

    def updateWordFreq(self, resultWordFreqMap, aWordFreList):
        # 更新词频map里的词频。由于是浅拷贝，会修改引用对象的取值。这样做的好处是免去了新建一个对象的成本。
        for line in aWordFreList:
            try:
                word, pos = line['word'].split("/")
            except:
                continue
            # print(word, self.stopWords)
            import re
            if word in self.stopWords or len(re.findall("[0-9]", word)) > 0:
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
                    charFreqList.append({'word': char + "/" + pos, "freq": freq})
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
                posFreqList.append({'word': pos + "/" + pos, "freq": freq})
            return posFreqList

        res = {}
        for line in wordFreqList:
            posFreqList = splitWordIntoChars(line)  # 借用统计词频的函数，统计这个字符频率map里的字符频率
            self.updateWordFreq(res, posFreqList)
        freqSorted = sorted(res.items(), key=lambda x: x[1], reverse=True)
        return freqSorted

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
        # print(self.stasticsData)
        self.stasticsData = femaleData.append(maleData)
        self.stasticsData = self.stasticsData.sample(frac=1)
        self.labels = self.stasticsData['gender'].values
        self.features = self.stasticsData.drop(["gender", 'uid', '_id'], axis=1).values

        self.labels = list(map(lambda x: [1] if x == "f" else [0], self.labels))
        self.labels = np.array(self.labels).reshape(len(self.labels), 1)

    def crossValidation(self):
        kf = KFold(n_splits=10)
        inputData = kf.split(self.features)
        count = 1
        totalCM = np.zeros([2,2])
        for train_index, test_index in inputData:
            X_train, X_test = self.features[train_index], self.features[test_index]

            #X_train, X_test = np.abs(X_train), np.abs(X_test)
            # print(X_train)
            y_train, y_test = self.labels[train_index], self.labels[test_index]
            num_feature = len(X_train[0])
            clf = deepLearning.LSTMClassifier(2, num_feature, learning_rate=1e-3,
                                              layer_num=2, hidden_size=50, timestep_size=350)
            clf.initGraph()
            clf.initOneHotEncoder4Y(y_train)
            def aaa(X, L):
                res = []
                for i in range(L):
                    line = X[i]
                    tres =  []
                    for n in line:
                        tres.append(n)
                    res.append(tres)
                return np.array(res)

            def scala(data):
                res = []
                for i in range(len(data)):
                    res.append(data[i]/(0.0000001 + sum(data[i])))
                return np.array(res)

            X_train, X_test = aaa(X_train, len(y_train)), aaa(X_test, len(y_test))
            from sklearn.preprocessing import RobustScaler, StandardScaler
            #norm_x = StandardScaler().fit(X_train)
            #X_train, X_test = norm_x.transform(X_train), norm_x.transform(X_test)
            X_train,X_test = X_train[:,:], X_test[:,:]
            X_train, X_test = scala(X_train), scala(X_test)

            y_train, y_test = np.array(y_train), np.array(y_test)

            for i in range(500):
                print("这是第", count, "折,第", i, "轮训练。", len(y_train) )
                for j in range(0, len(y_train), 200):

                    batch_ys = clf.oneHotEncode(y_train[j: j+ 200,:])
                    batch_xs = np.array(X_train[j: j+ 200,:]).astype(np.float32)
                    clf.fit(batch_xs, batch_ys)
                print("训练集")
                batch_ys = clf.oneHotEncode(y_train)
                batch_xs = np.array(X_train).astype(np.float32)
                clf.test(batch_xs, batch_ys)
                print("测试机")
                batch_ys = clf.oneHotEncode(y_test)
                batch_xs = np.array(X_test).astype(np.float32)
                clf.test(batch_xs, batch_ys)
                print("*************************")
            break

def classIt():
    t1 = time.time()
    genderAnnalysis = genderDetection()
    genderAnnalysis.initStopWords(r"C:\Users\Administrator\PycharmProjects\test\hupu\analisys\research\gender\HIT_stop_words.txt")
    genderAnnalysis.getStasticsData()


from matplotlib import pyplot as plt
def DE():
    genderAnnalysis = genderDetection()
    genderAnnalysis.initStopWords(
        r"C:\Users\Administrator\PycharmProjects\test\hupu\analisys\research\gender\HIT_stop_words.txt")
    genderAnnalysis.loadFeatureWords(r"C:\Users\Administrator\PycharmProjects\test\hupu\analisys\research\gender\genderFeature.txt")
    print("开始添加性别。", genderAnnalysis.featureChars)
    genderAnnalysis.getStasticsData()
    genderAnnalysis.addGender()
    genderAnnalysis.crossValidation()




import pickle
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
import time
if __name__ == '__main__':
    DE()
    # classIt()



