#条件随机场
#实现CRF的训练和使用，语料的预处理，标注结果的评估

import copy
import numpy as np
import random
import itertools
from multiprocessing import Pool
import pickle


def dotTwoDict(dict1, dict2):
    res = 0
    for key in dict1:
        if key in dict2:
            res += dict2[key] * dict1[key]
    return res
def addTwoDict(dict1, dict2):#将dict1内的数据加到dict2对应key上
    for key in dict1:
        if key in dict2:
            dict2[key] += dict1[key]

def calValueSumOfMap(aDict):
    res = 0
    for key, value in aDict.items(): res += value
    return res

class LinearChainCRF():
    
    def __init__(self, learningRate = 0.001, epoch = 1):
        self.featureWeightMap = {}#{("stateTransFeature", y_former, y_t): 1, ("stateFeature", y_t, x[t]): 1}
        self.stateList = None
        self.learningRate = 0.0001
        self.epoch = 10
        pass

    #基于超参数和训练数据，初始化CRF的参数
    def initParamWithTraingData(self, traningData):
        self.featureWeightMap = {}
        hiddenStateSet = set({})
        for sentence in traningData:
            charList = '@' + sentence[0] + '@'
            tagList = '*' + sentence[1] + '#'
            for tag in tagList:
                hiddenStateSet.add(tag)
            sentenceLength = len(charList)
            for t in range(1, sentenceLength-1):
                y_t = tagList[t]
                y_former = tagList[t-1]
                #统计特征在语料中出现的次数，如果太低，就删掉
                featuresHere = self.extractFeatures(y_t, y_former, t, charList)
                # print(featuresHere)
                for feature, value in featuresHere.items():
                    self.featureWeightMap[feature] = self.featureWeightMap.get(feature, 0) + value
        # print(self.featureWeightMap)
        self.stateList = list(hiddenStateSet)
        self.hiddenStateNum = len(self.stateList)
        for feature in list(self.featureWeightMap.keys()):
            if self.featureWeightMap[feature]<1:
                del self.featureWeightMap[feature]
            else:#出现次数较高的特征，给一个初始权重
                self.featureWeightMap[feature] = 0*random.uniform(-0.1, 0.1)


    #已知CRF模型参数和一个观测序列x=(x_1, x_2, ..., x_T)，
    # 求x_t处的前向变量取值aplha_t(thisState)
    def forwardAlgrithm(self, state_t, observationList, t):
        alphaList = []#存储前向向量
        alphaList.append([1])#认为添加的start步，也就是t=0的位置，对应的状态只有一个，前向向量
        #的长度就是1,特征函数的个数是0,向量元素的取值就是exp(0)=1
        for i in range(1, t):
            thisAlpha = np.zeros(self.hiddenStateNum)#t>=1时，隐藏状态的个数
            formerAlpha = alphaList[-1]
            stateNumOfFormerStep = len(formerAlpha)
            transProbMatrix = np.zeros((len(formerAlpha), self.hiddenStateNum))#用于存储t-1步的隐藏状态到t步
            #隐藏状态的非规范化转移概率
            for n in range(self.hiddenStateNum):#遍历本步的所有隐藏状态
                thisState = self.stateList[n]
                for j in range(stateNumOfFormerStep):
                    formerState = self.stateList[j]
                    featureFunctionValueMap = self.extractFeatures(thisState, formerState, i, observationList)
                    transProbMatrix[j, n] = dotTwoDict(featureFunctionValueMap, self.featureWeightMap)
            transProbMatrix = np.exp(transProbMatrix)
            thisAlpha = np.dot(formerAlpha, transProbMatrix)
            alphaList.append(thisAlpha)

        thisState = state_t
        thisObservation = observationList[t]
        formerAlpha = alphaList[-1]
        stateNumOfFormerStep = len(formerAlpha)
        alpha_t = 0#x_t处，隐藏状态取值为state_t时的前向变量取值
        for j in range(stateNumOfFormerStep):
            formerState = self.hiddenStatList[j]
            featureFunctionSum = self.getSumOfFeatureFuctions(thisState, formerState, thisObservation)
            alpha_t += np.exp(featureFunctionSum) * formerAlpha[j]
        return  alpha_t

    #已知CRF模型参数和一个观测序列x=(x_1, x_2, ..., x_T)，用后向算法
    # 求x_t处的条件概率p(y_t=s, y_t-1=s_dot|x)
    def backwardAlgrithm(self,  state_t, observationList, t):
        betaList = [None for _ in range(len(observationList))]#存储前向向量
        betaList[-1] = [1]#认为添加的start步，也就是t=0的位置，对应的状态只有一个，前向向量
        #的长度就是1,特征函数的个数是0,向量元素的取值就是exp(0)=1
        for i in range(len(observationList) - 2, t, -1):
            thisObservation = observationList[i]
            thisBeta= np.zeros(self.hiddenStateNum)#t>=1时，隐藏状态的个数
            formerBeta = betaList[i + 1]
            stateNumOfFormerStep = len(formerBeta)
            transProbMatrix = np.zeros((len(formerBeta), self.hiddenStateNum))
            for n in range(self.hiddenStateNum):#遍历本步的所有隐藏状态
                thisState = self.stateList[n]
                for j in range(stateNumOfFormerStep):
                    laterState = self.stateList[j]
                    featureFunctionValueMap = self.extractFeatures(laterState, thisState, i, observationList)
                    transProbMatrix[j, n] = dotTwoDict(featureFunctionValueMap, self.featureWeightMap)
            transProbMatrix = np.exp(transProbMatrix)
            thisBeta = np.dot(formerBeta, transProbMatrix)
            betaList[i] = thisBeta

        thisState = state_t
        laterAlpha = betaList[t+1]
        stateNumOfFormerStep = len(laterAlpha)
        beta_t = 0#x_t处，隐藏状态取值为state_t时的前向变量取值
        for j in range(stateNumOfFormerStep):
            laterState = self.stateList[j]
            featureFunctionValueMap = self.extractFeatures(laterState, thisState, t, observationList)
            beta_t += dotTwoDict(featureFunctionValueMap, self.featureWeightMap)
        return beta_t

    def generatePossibleStateTrans(self, t):#生成第t步，可能的隐藏状态转移
        if t==1:
            featureNames = itertools.product(['*'], self.hiddenStatList)
        else:
            featureNames = itertools.product(self.hiddenStatList, self.hiddenStatList)
        featureNames = list(map(lambda x: ''.join(x), featureNames))
        featureNames = list(filter(lambda x: x in self.featureWeightMap, featureNames))
        return featureNames
    
    def generatePossibleStateFeatueNames(self, t, observation):
        if t==1:
            featureNames = itertools.product(['@'], [observation])
        else:
            featureNames = itertools.product(self.hiddenStatList, [observation])
        featureNames = list(map(lambda x: ''.join(x), featureNames))
        featureNames = list(filter(lambda x: x in self.featureWeightMap, featureNames))
        return featureNames

    #提取观测序列和隐藏状态序列低t步的特征函数取值
    def extractFeatures(self, y_t, y_former, t, x):
        return {("stateTransFeature", y_former, y_t): 1, ("stateFeature", y_t, x[t]): 1}

    #计算模板函数权重对应的梯度
    def calGrad4Weight(self, sentence, corpusSize):
        charList = '@' + sentence[0] + '@'
        tagList = '*' + sentence[1] + '#'
        sentenceLength = len(charList)
        #提取所有的特征
        gradMap = {}
        for t in range(1, sentenceLength-2):
            featureFunctionValueMap = self.extractFeatures(tagList[t], tagList[t-1], t, charList)
            for feature, value in featureFunctionValueMap.items():
                gradMap[feature] = gradMap.get(feature, 0) + value

        z_x = self.backwardAlgrithm('*', charList, 0)#计算配分函数的取值

        for t in range(1, sentenceLength-1):
            thisState, formerState = tagList[t], tagList[t-1]
            thisObservation = charList[t]
            featureName2 = formerState + thisState
            featureName1 = thisState + thisObservation
            grad += self.ifFitFeatureTemplet(featureName1) + self.ifFitFeatureTemplet(featureName2)
            possibleFeatureNames = self.generatePossibleStateFeatueNames(t, thisObservation)
            if featureName in possibleFeatureNames:
                grad -= 1*self.calFeatureFunctionValueAndMargProb(featureName, charList, t)/z_x + \
                             self.featureWeightMap.get(featureName, 0)/(corpusSize * 10)
            possibleFeatureNames = self.generatePossibleStateTrans(t)
            if featureName in possibleFeatureNames:
                grad -= 1*self.calFeatureFunctionValueAndMargProb(featureName, charList, t)/z_x + \
                              self.featureWeightMap.get(featureName, 0)/(corpusSize * 10)

            gradMap[featureName] = grad
        return gradMap
    
    # 基于更新规则更新权重
    def updateWeight(self, gradMap):
        for featureName in gradMap:
            if featureName in self.featureWeightMap:
                self.featureWeightMap[featureName] += self.learningRate * gradMap[featureName]


    #基于训练语料，估计CRF参数
    def fit(self, sentenceList):
        if self.preTrain==False:
            self.initParamWithTraingData(sentenceList)
        corpusSize = len(sentenceList)
        weightList = []
        initLearningRate = float(self.learningRate)
        print(self.featureWeightMap)
        for epoch in range(self.epoch):
            pickle.dump(self, open('md.pkl', 'wb'))
            for n in range(corpusSize):
                sentence = sentenceList[n]#遍历语料中的每一句话，训练模型
                self.learningRate = initLearningRate /(2 * (1 + epoch ))
                gradMap = self.calGrad4Weight(sentence, corpusSize)#计算模板函数权重对应的梯度

                self.updateWeight(gradMap)#基于更新规则更新权重


    #基于观测值序列，也就是语句话的字符串列表，使用模型选出最好的隐藏状态序列，并按照分词标记将字符聚合成分词结果
    def predict(self, text): 
        statPathProbMap = {}#存储以各个初始状态打头的概率最大stat路径
        for stat in self.stateList:#遍历每一个隐藏状态
            statPath = stat#这是目前积累到的stat路径，也就是分词标记序列
            statPathProb = self.getSumOfFeatureFuctions(stat, '*', text[0])
            statPathProbMapOfThis = {}
            statPathProbMapOfThis[statPath] = statPathProb
            for t in range(1, len(text)):
                char  = text[t]
                tempPathProbMap = {}
                for statValue in self.hiddenStatList:
                    thisState = statValue
                    formerState = statPath[-1]
                    tempPath = statPath + thisState
                    tempPathProb = self.getSumOfFeatureFuctions(thisState, formerState, char)
                    tempPathProbMap[tempPath] = tempPathProb
                bestPath = getKeyWithMaxValueInMap(tempPathProbMap)
                statPathProbMapOfThis[bestPath] = tempPathProbMap[bestPath]
                statPath = bestPath
            statPathProbMap[statPath] = statPathProbMapOfThis[statPath]
        bestPath = getKeyWithMaxValueInMap(statPathProbMap)
#         print(len(bestPath) , len(text))
#         for i in range(len(text)):
#             print(bestPath[i], text[i])
        print(text)
        res = mergeCharsInOneWord(text, bestPath)
        return res
        
    def setMode(self, preTrain=True):
        self.preTrain = preTrain

def getKeyWithMaxValueInMap(dataMap):
    dataList = sorted(dataMap.items(), key=lambda x: x[1], reverse=True)
    theKey = dataList[0][0]
    return theKey
    
#基于分词标记把字符聚合起来，形成分词结果
def mergeCharsInOneWord(charList, tagList):
    wordList = []
    word = ''
    for i in range(len(charList)):
        tag, char = tagList[i], charList[i]
        if tag=='E':
            word += char
            wordList.append(word)
            word = ''
        elif tag=="S":
            word += char
            wordList.append(word)
            word = ''
        else:
            word += char
    return wordList

def loadData(fileName, sentenceNum = 100):
    with open(fileName, 'r', encoding='utf8') as f:
        line = f.readline()
        corpus = []
        tempSentence = []
        tempTag = []
        count = 0
        while line!=True:
            line = line.replace('\n', '')
            if line=='':#如果这一行没有字符，说明到了句子的末尾
                tempSentence = ''.join(tempSentence)#把字符都连接起来形成字符串，后面操作的时候会快一些
#                 if "习近平" in tempSentence:
#                     print(tempSentence)
                tempTag = ''.join(tempTag)
                corpus.append([tempSentence,tempTag])
#                 corpus.append([tempSentence[:20],tempTag[:20]])
#                 print("这句话是", [tempSentence,tempTag])
                tempSentence = []
                tempTag = []
                count += 1
                if count==sentenceNum:#如果积累的句子个数达到阈值，返回语料
                    return corpus
            else:
                line= line.split('\t')
#                 print(line)
                [char, tag] = line[0], line[1]#取出语料的文字和分词标记
                tempSentence.append(char)
                tempTag.append(tag)
            line = f.readline()
    return corpus

import time

if __name__ == '__main__':
    fileName = r"trainingCorpus4wordSeg_part.txt"
    sentenceNum = 50
    sentenceList = loadData(fileName, sentenceNum=sentenceNum)#加载语料
#     print(sentenceList)
    preTrain = False#False#,True
    if preTrain:
        model = pickle.load(open('md.pkl', 'rb'))
        model.setMode(preTrain=True)
    else:
        model = LinearChainCRF(epoch=10, learningRate=0.0001)
        model.setMode(preTrain=False)
    
    random.shuffle(sentenceList)
    model.fit(sentenceList)
    pickle.dump(model, open('md.pkl', 'wb'))
    for i in range(sentenceNum):
        res = model.predict(sentenceList[i][0])
        print("分词结果是", res, "真实的分词结果是", mergeCharsInOneWord(sentenceList[i][0], sentenceList[i][1]))
    #
    # s = "我是一个粉刷将，粉刷本领强。我要把我的新房子刷的很漂亮。"
    # res = model.predict(s)
    # testS = ['我是一个粉刷将，粉刷本领强。', '我要把我的新房子刷的很漂亮。',
    #           '我是一个粉刷将，粉刷本领强。我要把我的新房子刷的很漂亮。',
    #           '习近平指出，当前我国社会的主要矛盾仍然是人民日益增长的物质需求与不发达的生产力之间的矛盾。']
    # for s in testS:
    #     t1 = time.time()
    #     res = model.predict(s)
    #     t2 = time.time()
    #     print(t2-t1, res)
   
