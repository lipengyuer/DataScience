import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
path_src = path.split('src')[0]
sys.path.append(path_src + "src")#项目的根目录
from pyhanlp import HanLP
from pyhanlp import *
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import time
from pymongo import MongoClient
#from analysis.algorithm import splitSentence, nlp
import splitSentence, nlp
from pyspark import SparkContext, SparkConf
import runTime
import re
#词频
def wordFreq(wordsLists):
    wordFreqMap = {}
    for post in wordsLists:#遍历post
        for sentence in post:#遍历post的句子
            for word in sentence:#遍历句子的词语
                if word in wordFreqMap:
                    wordFreqMap[word] += 1
                else:
                    wordFreqMap[word] = 1
    return wordFreqMap

#字符ngram
def ngramFreq(postList, n=1):
    ngramFreqMap = {}
    for post in postList:
        for i in range(len(post)-n):
            ngram = post[i:i+n]
            if ngram in ngramFreqMap:
                ngramFreqMap[ngram] += 1
            else:
                ngramFreqMap[ngram] = 1
    return ngramFreqMap

#词性标签ngram
def postagNgramFreq(postagsLists, n=1):
    ngramFreqMap = {}
    for post in postagsLists:#遍历post
        for sentence in post:#遍历post的句子
            for i in range(len(sentence)-n):
                ngram = '_'.join(sentence[i:i+n])
                if ngram in ngramFreqMap:
                    ngramFreqMap[ngram] += 1
                else:
                    ngramFreqMap[ngram] = 1
    return ngramFreqMap

#特殊符号频率
def specialCharFreq(postList):
    specialMarks = "@#$%"
    specialMarkSet = set(list(specialMarks))
    markFreqMap = {}
    for post in postList:
        for i in range(len(post)):
            ngram = post[i]
            if ngram not in specialMarkSet:
                continue
            if ngram in markFreqMap:
                markFreqMap[ngram] += 1
            else:
                markFreqMap[ngram] = 1
    return markFreqMap

def nerFreq():
    #命名实体频率暂时不考虑
    pass

#统计虚词的词频
def functionWordFreq(wordsLists):
    functionWords = '自 自从 从 到 向 趁 按 按照 依照 根据 靠 拿 比 因 因为 为 由于 被 给 让 叫 归 由 对 对于 关于 跟 和 同 向 和 同 跟 与 及 或 而 而且 并 并且 或者 不但 不仅 虽然 但是 然而 如果 与其 因为 所以 的 地 得 之 者 着 了 过 看  的 来着 来 把 左右 上下 多 似的 一样 一般 所 给 连 的 了 吧 呢 啊 嘛 呗 罢了 也好 啦 喽 着呢 吗 呢 吧 啊 '
    functionWordSet= set(functionWords.split(' '))
    wordFreqMap = {}
    for post in wordsLists:#遍历post
        for sentence in post:#遍历post的句子
            for word in sentence:#遍历句子的词语
                if word not in functionWordSet:
                    continue
                if word in wordFreqMap:
                    wordFreqMap[word] += 1
                else:
                    wordFreqMap[word] = 1
    return wordFreqMap

#标点符号频率
def punctuationMarkFreq(postList):
    punctuationMarks = """：。«¨〃—～‖‘’“”。，、；：？！―〃～＂＇｀｜﹕﹗/\\"""
    marksSet = set(list(punctuationMarks))
    pmFreqMap = {}
    for post in postList:
        for i in range(len(post)):
            ngram = post[i]
            if ngram not in marksSet:
                continue
            if ngram in pmFreqMap:
                pmFreqMap[ngram] += 1
            else:
                pmFreqMap[ngram] = 1
    return pmFreqMap

    
#基于post列表，以及对应的每一句的分词结果和词性标注结果，来统计相关语言风格特征
def lengthFeature(sentencesList, wordsList, postagsList):
    res = {}
    charNumInSentence, count1 = 0, 0#统计句子的字符个数均值
    sentenceNumInPost, count2 = 0, 0#统计每个评论的句子个数均值
    for sentences in sentencesList:
        sentenceNumInPost += len(sentences)
        count2 += 1
        for sentence in sentences:
            count1 += 1
            charNumInSentence += len(sentence)
    charNumInSentence = charNumInSentence/count1 if count1>0 else 0
    sentenceNumInPost = sentenceNumInPost/count2 if count2>0 else 0
    res['charNumInSentence'] = charNumInSentence
    res['sentenceNumInPost'] = sentenceNumInPost
    wordNumTotal, count1 = 0, 0
    wordLength, count2 = 0, 0#统计每个词语的字符个数均值
    for wordList in wordsList:
        for words in wordList:
            count1 += 1
            wordNumTotal += len(words)
            for word in words:
                count2 += 1
                wordLength += len(word)
    # 统计句子的词语个数均值
    wordNumInSentence = wordNumTotal/count1 if count1>0 else 0
    wordLength = wordLength/count2 if count2>0 else 0
    # 统计每个评论的词语个数均值
    postNum = len(wordsList)
    wordNumInPost = wordNumTotal/postNum if postNum>0 else 0
    res['wordNumInSentence'] = wordNumInSentence
    res['wordLength'] = wordLength
    res['wordNumInPost'] = wordNumInPost
    #统计每个句子的标点个数均值
    postagNumTotal, count1 = 0, 0
    for postagList in postagsList:
        for postags in postagList:
            count1 += 1
            for postag in postags:
                if 'w' in postag:
                    postagNumTotal += 1
    postagNumInSentence = postagNumTotal/count1 if count1>0 else 0
    res['postagNumInSentence'] = postagNumInSentence
    return res

def getGender(uid, userInfoCollectionName):
    MONGO_IP = '192.168.1.198'
    MONGO_PORT = 27017
    MONGO_DB_NAME = "hupu"
    MONGO_IP, MONGO_PORT ,dbname, username, password = MONGO_IP, MONGO_PORT, \
                                                          MONGO_DB_NAME, 'root', '1q2w3e4r'    
    conn = MongoClient(MONGO_IP, MONGO_PORT)
    db = conn[dbname]
    db.authenticate(username, password)
    gender = db[userInfoCollectionName].find({"uid": uid}, {"gender": 1})
    gender = list(gender)
    if len(gender)==0:
        return None
    print(gender)
    gender = gender[0]['gender']
    gender = 1 if gender=='f' else 0
    return gender

def extractTextFeatures(data, userInfoCollectionName=''):
    uid, postList = data[0], data[1]
    gender = -1
    #gender = getGender(uid, userInfoCollectionName)
    #if gender==None:#如果没有性别数据，这条数据无效
    #    return {}
    #获取分句，分词和词性标注结果
    sentencesLists, wordsLists, postagsLists = nlp.sentenceWordPostag(postList)
    #各类特征放在不同的key下
    featureMap = {"uid": uid,
                  'gender': gender,
                  "wordFreq": wordFreq(wordsLists), 
                  "unigramFreq": ngramFreq(postList, n=1),
                   'bigram': ngramFreq(postList, n=2), 
                  'postagUnigramFreq': postagNgramFreq(postagsLists, n=1), 
                  'postagBigramFreq': postagNgramFreq(postagsLists, n=2),
                   'postagTrigramFreq': postagNgramFreq(postagsLists, n=3), 
                   'specialCharFreq': specialCharFreq(postList),
                  'functoinWordFreq': functionWordFreq(wordsLists), 
                  'punctuationMarkFreq': punctuationMarkFreq(postList),
                  'sentenceLengthFeatures': lengthFeature(sentencesLists, wordsLists, postagsLists)}
    return featureMap

def getFirst(x):
    return x[0]

#向mongo插入一条数据
def saveRecord2Mongo(data, collectionName):
    MONGO_IP = '192.168.1.198'
    MONGO_PORT = 27017
    MONGO_DB_NAME = "hupu"
    MONGO_IP, MONGO_PORT ,dbname, username, password = MONGO_IP, MONGO_PORT, \
                                                          MONGO_DB_NAME, 'root', '1q2w3e4r'       
    conn = MongoClient(MONGO_IP, MONGO_PORT)
    db = conn[dbname]
    db.authenticate(username, password)
    collection = db[collectionName]
    collection.insert(data, check_keys=False)
    
if __name__ == '__main__':
    pass
    #获取用户个人资料
    #基于用户个人资料统计基本信息
    #分布式化为pyspark.dataframe
    #与主贴信息连接，然后与这些主贴的回复连接,按照uid对回复分组之后，统计这些回复的情况
    #存储数据
    #没毛病，这次我们需要知道一个用户的主贴的跟帖的情况。而原始回复数据里没有主贴对应用户uid,需要关联才能得到.
    #使用用户个人资料来开始，也是为了缩小数据规模.