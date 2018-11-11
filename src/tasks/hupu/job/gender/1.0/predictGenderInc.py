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
        self.N_word, self.N_pos, self.N_char = 200, 500, 500
        self.count = 5000
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
        data = collection.find({}, {"detail": 1})
        count = 0
        self.stasticsData = []
        for line in data:
            uid = line['_id']
            thisData = line['detail']['follContentFeatrue']
            if len(thisData)>100:
                f = self.featureExtractionFromWords(thisData)
                print(count, len(f))
                count += 1
                self.stasticsData.append({"uid": uid, "feature": list(f),
                                          'followPostNumber': line['detail']['followPostNumber'],
                                         'lightedNumber': line['detail']['lightedNumber'],
                                          'footBallIndex': line['detail']['footballIndex']})
            if count == self.count:
                break
        data = pd.DataFrame(self.stasticsData)
        pickle.dump(data, open('data.pkl', 'wb'))
        data = pickle.load(open('data.pkl', 'rb'))

        self.stasticsData = data.reset_index(drop=True)

        # print(self.stasticsData)

    def featureExtractionFromWords(self, aWordFreqMap):
        def initACleanMap(keyList):
            tempMap = {}
            for key in keyList:
                tempMap[key] = 0
            return tempMap
        def normallize(feature):
            m = np.median(list(feature))
            m4 = np.percentile(list(feature), 80)
            feature = list(map(lambda x: x if x<1000 else 1000, feature))
            feature = list(map(lambda x: x, feature))
            return feature
        line = aWordFreqMap
        wordFreqMap = initACleanMap(self.featureWords)
        posTagFreqMap = initACleanMap(self.featurePos)
        charFreqMap = initACleanMap(self.featureChars)
        # print(self.featureWords)
        # print(self.featureChars)

        for aWordFreq in line:
            try:
                word, posTag = aWordFreq['word'].split("/")
            except:
                continue
            freq = int(aWordFreq['freq'])
            #
            if word in wordFreqMap:
                # print(word, wordFreqMap[word], freq)
                wordFreqMap[word] += freq
                # print(wordFreqMap[word])
            # print(wordFreqMap)
            if posTag in posTagFreqMap:
                posTagFreqMap[posTag] += freq
            for char in word:
                if char in charFreqMap:
                    charFreqMap[char] += freq
        featureWords, featurePos, featureChars = wordFreqMap.values(),posTagFreqMap.values(), charFreqMap.values()

        featureWords, featurePos, featureChars = normallize(featureWords), \
                                                 normallize(featurePos), \
                                                 normallize(featureChars)
        features = featureWords + featurePos + featureChars
        # print(features)
        return features

    def loadFeatureWords(self, path):
        import re
        N_word, N_pos, N_char = self.N_word, self.N_pos,self.N_char
        with open(path, 'r', encoding="utf8") as f:
            lines = f.readlines()
            lines = list(map(lambda x: x.replace("\n", "").split('是')[1][1:], lines))
        words = re.findall("\'.+?\'",lines[0])[0:int(0.5*N_word)] + \
                                           re.findall("\'.+?\'",lines[1])[0:N_word-int(0.5*N_word)]
        words = list(map(lambda x: x.replace('\'', ''), words))
        posTags = re.findall("\'.+?\'", lines[1])[0:int(0.5*N_pos)] + \
                                           re.findall("\'.+?\'", lines[4])[0:N_pos-int(0.5*N_pos)]
        posTags = list(map(lambda x: x.replace('\'', ''), posTags))
        chars = re.findall("\'.+?\'", lines[2])[0:int(0.5*N_char)] + \
                                           re.findall("\'.+?\'", lines[5])[0:N_char-int(0.5*N_char)]
        chars = list(map(lambda x: x.replace('\'', ''), chars))
        self.featureWords = list(sorted(set(words)))#用于构建特征的词语
        self.featurePos = list(sorted(set(posTags)))
        self.featureChars = list(sorted(set(chars)))

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



    def showTopWordsInUserName(self):
        N = 1000
        femaleData = self.stasticsData.loc[self.stasticsData['gender'] == 'f']
        maleData = self.stasticsData.loc[self.stasticsData['gender'] == 'm']
        print("男性的数量是", len(maleData), ". 女性的数量是", len(femaleData))
        femaleData, maleData = femaleData['feature'], maleData['feature']
        femaleTopWords, maleTopWords = {}, {}
        for wordFreqList in femaleData:
            femaleTopWords = self.updateWordFreq(femaleTopWords, wordFreqList)
        femaleTopWords = sorted(femaleTopWords.items(), key=lambda x: x[1], reverse=True)
        print("女性用户的高频字符是", self.getCharFreq(femaleData)[:N])
        print("女性用户名的高频词是", femaleTopWords[:N])
        print("女性用户的高频词性是", self.getPosFreq(femaleData)[:N])
        for wordFreqList in maleData:
            maleTopWords = self.updateWordFreq(maleTopWords, wordFreqList)
        maleTopWords = sorted(maleTopWords.items(), key=lambda x: x[1], reverse=True)
        print("男性用户的高频字符是", self.getCharFreq(maleData)[:N])
        print("男性用户名的高频词是", maleTopWords[:N])
        print("男性用户的高频词性是", self.getPosFreq(maleData)[:N])

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
        print(self.stasticsData)
        profleData = self.stasticsData[['followPostNumber',
                                        'lightedNumber', 'footBallIndex']].values
        self.stasticsData = femaleData.append(maleData)
        self.stasticsData = self.stasticsData.sample(frac=1)
        self.features = np.concatenate((self.stasticsData['feature'].values, profleData), axis=1)
        self.labels = self.stasticsData['gender'].values
        self.labels = list(map(lambda x: [1] if x == "f" else [0], self.labels))
        # # self.features['gender'] = np.array(self.labels).reshape(len(self.labels), 1)
        self.labels = np.array(self.labels).reshape(len(self.labels), 1)

    def crossValidation(self):
        kf = KFold(n_splits=10)
        inputData = kf.split(self.features)
        count = 1
        totalCM = np.zeros([2,2])
        # self.features = np.array(self.features)
        # self.labels = np.array(self.labels)
        for train_index, test_index in inputData:
            # clf = DecisionTreeClassifier(max_depth=10)
            # clf = RandomForestClassifier(n_estimators=30, max_depth=6,random_state=666)
            # clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), n_estimators=20)
            clf = MLPClassifier(hidden_layer_sizes=(500,))
            # clf = LogisticRegression(max_iter=1000, solver='lbfgs', C=100)
            # clf = GradientBoostingClassifier(n_estimators=20)
            # clf = SVC(C=0.8)
            # clfList = [
            #     ["AD", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), n_estimators=20)],
            #     ["gbdt", GradientBoostingClassifier(n_estimators=20)],
            #     ["LR", LogisticRegression(max_iter=1000, solver='lbfgs', C=100)],
            #     ['desisionTree', DecisionTreeClassifier()],
            #            ['mlp', MLPClassifier(hidden_layer_sizes=(200, 100))],
            #            ['KNN', KNeighborsClassifier(n_neighbors=50)]]
            # clf = VotingClassifier(clfList, voting='hard')
            X_train, X_test = self.features[train_index], self.features[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]
            # featureProcessor = PCA(n_components=20)
            print(len(X_train),'asd', len(y_train))
            # print(X_train,'asd', y_train)
            def aaa(X, L):
                res = []
                for i in range(L):
                    line = X[i]
                    tres =  []
                    for n in line:
                        tres.append(n)
                    res.append(tres)
                return np.array(res)
            X_train, X_test = aaa(X_train, len(y_train)), aaa(X_test, len(y_test))
            X_train,X_test = X_train[:,:], X_test[:,:]
            y_train, y_test = np.array(y_train), np.array(y_test)
            # featureProcessor = PCA(n_components=10)

            featureProcessor = LinearDiscriminantAnalysis(n_components=200).fit(X_train, y_train)
            # # featureProcessor = SelectKBest(chi2, k=20)#.fit(features, labels)
            # featureProcessor.fit(X_train, y_train)
            X_train = featureProcessor.transform(X_train)
            X_test = featureProcessor.transform(X_test)

            clf.fit(X_train, y_train)
            pred_train = clf.predict(X_train)
            y_train = list(y_train)
            cm = confusion_matrix(y_train, pred_train)
            print("第", count, "轮训练集里的表现\n", cm)
            pred = clf.predict(X_test)
            y_test = list(y_test)
            cm = confusion_matrix(y_test, pred, labels=[0, 1])
            totalCM = totalCM + np.array(cm)
            print("第", count, "轮测试集里的表现\n", cm)
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


from matplotlib import pyplot as plt
def DE():
    genderAnnalysis = genderDetection()
    genderAnnalysis.initStopWords(
        r"C:\Users\Administrator\PycharmProjects\test\hupu\analisys\research\gender\HIT_stop_words.txt")
    genderAnnalysis.loadFeatureWords(r"C:\Users\Administrator\PycharmProjects\test\hupu\analisys\research\gender\genderFeature.txt")
    print("开始添加性别。", genderAnnalysis.featureChars)
    genderAnnalysis.getStasticsData()
    genderAnnalysis.addGender()
    genderAnnalysis.showTopWordsInUserName()
    genderAnnalysis.crossValidation()




import pickle
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
import time
if __name__ == '__main__':
    DE()
    # classIt()



