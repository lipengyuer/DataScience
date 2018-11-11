import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
path_src = path.split('src')[0]
sys.path.append(path_src + "src")#项目的根目录
import pymongo
from  pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from analysis.config import environment
userStasticsCollection = "completeUserFeatureSample"
MONGO_IP, MONGO_PORT ,dbname, username, password = environment.MONGO_IP, environment.MONGO_PORT, \
                                                          environment.MONGO_DB_NAME, None, None
conn = MongoClient(MONGO_IP, MONGO_PORT)
db = conn[dbname]
# db.authenticate(username, password)
collection = db[userStasticsCollection]

def compareSimpleFeatures(featureName=""):
    data = collection.find({}, {'uid':1, "gender": 1, featureName: 1})#从mongo中查询这个特征以及对应的性别标签
    dataList = []
    for line in data:
        print("正在读取数据", line['uid'])
        dataList.append({'gender': line['gender'], **line[featureName]})
    #dataList = list(lambda x: x[featureName], dataList)
    print("展示一条数据", line)
    print("完成数据读取，开始分组")
    df = pd.DataFrame(dataList)
    #print(df)
    # df = df.drop(columns=['_id'])
    dataF, dataM = df[df['gender']==1], df[df['gender']==0]#把男性和女性的数据分组
    #删掉两份数据中的性别字段
    dataF = dataF.drop(columns=['gender' ])
    dataM = dataM.drop(columns=['gender'])
    #求两份数据里，各个特征的平均值
    print("男女数量分别是", dataM.shape[0], dataF.shape[0])
    # print(dataF)
    meanF, meanM = dataF.mean(axis = 0), dataM.mean(axis = 0)
    colNames = list(dataF.columns)#字段名列表，用于画x轴刻度
    print(colNames)
    ax = plt.subplot(1,1,1)
    p1, = ax.plot(meanF,marker='*')
    p2, = ax.plot(meanM, marker='+')
    # plt.xticks(colNames)
    plt.xlabel(featureName)
    plt.ylabel("mean frequency")
    plt.legend(handles = [p1, p2], labels = ["female", 'male'])
    plt.show()

def compareLengthFeatures():
    compareSimpleFeatures("sentenceLengthFeatures")

def specialCharFeatures():
    compareSimpleFeatures("specialCharFreq")

def functionWordFeatures():
    compareSimpleFeatures("functoinWordFreq")

def punctuationMarkFeatures():
    compareSimpleFeatures("punctuationMarkFreq")

#################################################
#分析词频，ngram频率,postagNgram频率。这几种特征的特点是，维度很高，需要首先用一定的方法，
#选取较好的特征，然后分析。这样做的目的是发现一些男女差异较大的地方，然后展示出来。
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

#获取词汇表,并统计所有词语的频数,文档频率
import copy
def getWordFreqDocumentFreq(userWordFreqMapList, jobName = 'wordFreqDocumentFreq'):
    userWordFreqMapList = copy.deepcopy(userWordFreqMapList)
    wordFreqMap, ducumentFreqMap = {}, {}
    for userWordFreqMap in userWordFreqMapList:
        if 'gender' in userWordFreqMap:
            del userWordFreqMap['gender']
        for word in userWordFreqMap:
            if word in wordFreqMap:
                wordFreqMap[word] += userWordFreqMap[word]#把该用户的这个词语的频数累加上
                ducumentFreqMap[word] += 1#这个用户使用了这个词语，文档频率加一
            else:
                wordFreqMap[word] = 1
                ducumentFreqMap[word] = 1
    print(jobName, "原始词汇表的大小是", len(wordFreqMap))
    N = 50
    noneWords1 = set(sorted(wordFreqMap.items(), key= lambda x: x[1],reverse=True)[-10000:])
    noneWords2 = set(sorted(ducumentFreqMap.items(), key= lambda x: x[1],reverse=True)[:1000])
    TFIDFMAP = {}
    numDoc = len(userWordFreqMapList)
    for word in wordFreqMap:
        if word in noneWords1 or word in noneWords2:
            continue
        tf = wordFreqMap.get(word, 0)
        idf = numDoc/(ducumentFreqMap.get(word, 1))
        TFIDFMAP[word] = np.log(tf*idf)
    #返回频率和文档频率都比较高的gram，作为较优特征
    res = sorted(TFIDFMAP.items(), key= lambda x: x[1],reverse=True)[1000:50000]
    res = set(map(lambda x: x[0], res))
    return res

# def readlines(path):
#     with open(path, 'r') as f:
import pickle
def ngramFreatures(featureName=""):
    data = collection.find({}, {'_id':1, "gender": 1, featureName: 1})#从mongo中查询这个特征以及对应的性别标签
    dataList = []
    print("正在读取数据")
    count = 0
    for line in data:
        line[featureName].update({'gender': line['gender']})
        dataList.append(line[featureName])
        count += 1
        if count == 5000:
            break
        print("读取数据的进度是", count, "/", 5000)
    # pickle.dump(dataList, open('data.pkl','wb'))
    # dataList = pickle.load(open('data.pkl','rb'))
    print(dataList[0])
    #首先对所有的gram进行一个简单筛选，把普及率低于一定阈值(几乎所有人都不用的),总的使用次数小于一定阈值(大家都用过，然而昙花一现的)
    print("正在初步筛选特征。")
    betterFeatureSet = getWordFreqDocumentFreq(dataList, jobName=featureName)
    maleSpecialWords = [ '武器库',  'UFC','硬邦邦的', '龟头', '前臂', '尼玛比']
    femaleSpecialWords = ['小女子', '小宝贝', "美少年", '萌图', '治愈系', '防晒霜',
                          '萌系']
    betterFeatureSet = betterFeatureSet | set(maleSpecialWords) | set(femaleSpecialWords)
    betterFeatureSet.add("gender")
    #从通过初筛的所有gram中挑选使用率最高的10000个，进入下一步
    import time
    print("删除不优质的特征")
    for sample in dataList:
        for key in list(sample.keys()):
            if key not in betterFeatureSet:
                del sample[key]#删除不是优质特征的条目
    # pickle.dump(dataList, open('data1.pkl','wb'))
    # dataList = pickle.load(open('data1.pkl','rb'))
    df = pd.DataFrame(dataList).fillna(0)
    features = df.drop(columns=['gender'])

    # features = featureProcessor.transform(features)
    featureNames = list(df.columns)
    featureNames.remove('gender')
    featureNameIndex = list(range(len(featureNames)))
    #选取最好的k个特征
    print("挑选好的特征")
    from  sklearn.feature_selection import mutual_info_classif
    featureProcessor = SelectKBest(mutual_info_classif, k=3000)#.fit(features, labels)
    featureProcessor.fit(features, df['gender'])
    featureNameIndex = featureProcessor.transform(np.array([featureNameIndex]))[0]
    featureNames = np.array(featureNames)[featureNameIndex]
    featureNames = set(featureNames) | set(maleSpecialWords) | set(femaleSpecialWords)
    print("提取出来的特征名称是", "kabukabu".join(featureNames))
    with open("goodFeature.txt", 'w') as f:
        f.write("kabukabu".join(featureNames))
    # print("被选中的特征是", featureNameIndex)#与图中的特征名核对一下
    featureNames = list(featureNames)
    dataF, dataM = df[df['gender']==1], df[df['gender']==0]#把男性和女性的数据分组
    dataF, dataM = dataF[featureNames], dataM[featureNames]
    #删掉两份数据中的性别字段
    # dataF = dataF.drop(columns=['gender'])
    # dataM = dataM.drop(columns=['gender'])
    #求两份数据里，各个特征的平均值
    meanF, meanM = dataF.mean(axis=0),dataM.mean(axis=0)
    # colNames = list(meanF.columns)#字段名列表，用于画x轴刻度
    ax = plt.subplot(1,1,1)
    p1, = ax.plot(meanF)
    p2, = ax.plot(meanM)
    # plt.xticks(list(dataF.columns))
    plt.xlabel(featureName)
    plt.ylabel("mean frequency")
    plt.legend(handles = [p1, p2], labels = ["female", 'male'])
    plt.show()

def compareVocabs():
    featureName = 'completeUserFeatureSample'
    collection = db[featureName]
    data = collection.find({}, {'gender': 1, 'wordFreq': 1})
    maleVocab, femaleVocab = set({}), set({})
    data = list(data)
    for line in data:
        if line['gender'] == 0:
            maleVocab = maleVocab | set(line['wordFreq'].keys())
        else:
            femaleVocab = femaleVocab | set(line['wordFreq'].keys())
    print("男性用户的词汇量是", len(maleVocab), ',女性的是', len(femaleVocab), '.')
    only4Male = maleVocab - femaleVocab
    only4Female = femaleVocab - maleVocab
    only4MaleFreq, only4FemaleFreq = {}, {}
    for line in data:
        if line['gender'] == 0:
            for word in line['wordFreq']:
                if word in only4MaleFreq:
                    only4MaleFreq[word] += 1
                else:
                    only4MaleFreq[word] = 1
        else:
            for word in line['wordFreq']:
                if word in only4FemaleFreq:
                    only4FemaleFreq[word] += 1
                else:
                    only4FemaleFreq[word] = 1
    for word in list(only4MaleFreq.keys()):
        if word not in only4Male:
            del only4MaleFreq[word]
    for word in list(only4FemaleFreq.keys()):
        if word not in only4Female:
            del only4FemaleFreq[word]

    only4MaleFreqSorted = sorted(only4MaleFreq.items(), key=lambda x: x[1], reverse=True)
    only4FemaleFreqSorted = sorted(only4FemaleFreq.items(), key=lambda x: x[1], reverse=True)
    with open("maleData.txt", 'w') as f:
        for line in only4MaleFreqSorted:
            f.write(str(line[0]).replace('\n', '') + " " + str(line[1]) + "\n")
    with open("femaleData.txt", 'w') as f:
        for line in only4FemaleFreqSorted:
            f.write(str(line[0]).replace('\n', '') + " " + str(line[1]) + "\n")

if __name__ == '__main__':
    #从抽样表中查询数据，然后查询出这些用户的数据存储到一个新的表中，用来分析
    #查看两种性别的语句长度特征，形成两条取值曲线来对比
    #compareLengthFeatures()

    #特殊符号频率
    #specialCharFeatures()

    #虚词频率
    #functionWordFeatures()
    #标点符号频率
    #punctuationMarkFeatures()
    #基于卡方检验选择用于判断性别的较好特征(词频，ngram等)，画图对比这些较好特征再两性中的分布差异。
    ngramFreatures(featureName='wordFreq')
    # ngramFreatures(featureName='unigramFreq')
    # ngramFreatures(featureName='bigram')
    # ngramFreatures(featureName='postagBigramFreq')
    # ngramFreatures(featureName='postagUnigramFreq')
    # ngramFreatures(featureName='postagTrigramFreq')

    # compareVocabs()


