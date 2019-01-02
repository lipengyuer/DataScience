#条件随机场
#实现CRF的训练和使用，语料的预处理，标注结果的评估

import copy
import numpy as np
import random
import itertools
from multiprocessing import Pool

class LinearChainCRF():
    
    def __init__(self, learningRate = 0.001, epoch = 1):
        self.epoch = epoch
        #定义CRF的参数
        #特征模板。如果没有设置，就是用使用默认的线性链条件随机场的模板规则，即只考虑当前
        #观测值x_t，当前隐藏状态y_t，以及前一个隐藏状态y_before_t
        self.hiddenStatList = None#隐藏状态的取值空间
        self.minWordNum = None#语料中的词语，出现次数小于这个阈值的，不会用于特征模板
        self.minCharNum = None#语料中的字符，出现次数小于这个阈值的，不会用于模板
        self.gradListOfWeights = None#训练过程中，存储各个模板函数权重的梯度
        self.learningRate = learningRate#模型训练的学习率，这里为了简单，使用一个统一的
        self.hiddenStateNum = None
        self.featureWeightMap =  None#每一个模板函数的权重,为了减小激素啊过程的内存消耗，这里使用map存储
        self.stateTransFeatureSet = None
        self.stateFeatureSet = None
        self.featureFunctionNum = None#特征函数的个数
        #为了简单，这里暂时不加正则化

    #判断当前状态转换是否符合某一个特征模板，返回值为这个特征函数的取值
    def ifFitFeatureTemplet(self, statTransFeature):
        if statTransFeature in self.stateTransFeatureSet or statTransFeature in self.stateFeatureSet:
            return 1.
        else:
            return 0.

    #基于超参数和训练数据，初始化CRF的参数
    def initParamWithTraingData(self, traningData):
        self.featureWeightMap = {}
        stateTransFeatureNumMap, statFeatureNumMap = {}, {}
        self.stateTransFeatureSet, self.stateFeatureSet = None, None
        hiddenStateSet = set({})
        for sentence in traningData:
            charList = '@' + sentence[0] + '@'
            tagList = '*' + sentence[1] + '#'
            for tag in tagList:
                hiddenStateSet.add(tag)
            sentenceLength = len(charList)
            for i in range(1, sentenceLength-1):
                statTransFeature = tagList[i - 1: i+1]  # 线性链CRF的特征函数只有两种
                statFeature = tagList[i] + charList[i]
                #统计特征在语料中出现的次数，如果太低，就删掉
                stateTransFeatureNumMap[statTransFeature] = stateTransFeatureNumMap.get(statTransFeature, 0) + 1
                statFeatureNumMap[statFeature] = statFeatureNumMap.get(statFeature, 0) + 1
        for feature in list(stateTransFeatureNumMap.keys()):
            if stateTransFeatureNumMap[feature]<2:
                del stateTransFeatureNumMap[feature]
            else:#出现次数较高的特征，给一个初始权重
                self.featureWeightMap[feature] = random.uniform(-0.1, 0.1)
        for feature in list(statFeatureNumMap.keys()):
            if statFeatureNumMap[feature]<2:
                del statFeatureNumMap[feature]
            else:#出现次数较高的特征，给一个初始权重
                self.featureWeightMap[feature] = random.uniform(-0.1, 0.1)
        self.stateTransFeatureSet = set(list(stateTransFeatureNumMap.keys()))
        self.stateFeatureSet = set(list(statFeatureNumMap.keys()))
        print("特征函数的初始权重是", self.featureWeightMap)
        featureNameList = list(self.featureWeightMap.keys())
        self.featureFunctionNum = len(featureNameList)
        hiddenStateSet.remove('*')
        hiddenStateSet.remove("#")
        self.hiddenStatList = list(hiddenStateSet)
        self.hiddenStateNum = len(hiddenStateSet)
        print("特征函数的个数是", self.featureFunctionNum)

    #计算观测序列t处，某个特征函数的取值，以及对应的边缘概率，为计算梯度做准备
    def calFeatureFunctionValueAndMargProb(self, featureName, observationList, t):
        margProb = 0
        if featureName in self.stateFeatureSet:#如果是状态特征
            state_t = featureName[0]#状态特征的name的第一位是stat_t, 第二位是x_t
            margProb = self.forwardAlgrithm(state_t, observationList, t) * \
                               self.backwardAlgrithm(state_t, observationList, t)
        elif featureName in self.stateTransFeatureSet:#如果是状态转移特征
            observation_t = observationList[t]
            state_t = featureName[1]
            stateFormer = featureName[0]
            margProb = self.forwardAlgrithm(stateFormer, observationList, t-1) * \
                             self.getSumOfFeatureFuctions(state_t, stateFormer, observation_t) * \
                               self.backwardAlgrithm(state_t, observationList, t)
        return margProb
            
    def calFeatureFunctionValueAndMargProbPar(self, featureNameList, observationList, t):
        pool = Pool(4)
        resList = []
        res = []
        for featureName in featureNameList:
            result = pool.apply_async(self.calFeatureFunctionValueAndMargProb, args = (featureName, \
                                                                                       observationList,\
                                                                                       t))
            resList.append([featureName, result])
        pool.close() # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
        pool.join() # 等待进程池中的所有进程执行完毕
        for result in resList:
            res.append([result[0], result[1].get()])
        return res
    #已知模型参数，求一个观测值处的特征函数加权和
    def getSumOfFeatureFuctions(self, thisState, formerState, thisObservation):
        stateTrans = formerState + thisState
        stateFeature = formerState + thisObservation
        featureValue = self.featureWeightMap.get(stateTrans, 0) * \
                       self.ifFitFeatureTemplet(stateTrans) + \
                       self.featureWeightMap.get(stateFeature, 0) * \
                       self.ifFitFeatureTemplet(stateFeature)
        return featureValue

    #已知CRF模型参数和一个观测序列x=(x_1, x_2, ..., x_T)，
    # 求x_t处的前向变量取值aplha_t(thisState)
    def forwardAlgrithm(self, state_t, observationList, t):
        alphaList = []#存储前向向量
        alphaList.append([1])#认为添加的start步，也就是t=0的位置，对应的状态只有一个，前向向量
        #的长度就是1,特征函数的个数是0,向量元素的取值就是exp(0)=1
        for i in range(1, t):
            thisObservation = observationList[i]
            thisAlpha = np.zeros(self.hiddenStateNum)#t>=1时，隐藏状态的个数
            formerAlpha = alphaList[-1]
            stateNumOfFormerStep = len(formerAlpha)
            for n in range(len(self.hiddenStatList)):#遍历本步的所有隐藏状态
                thisState = self.hiddenStatList[n]
                for j in range(stateNumOfFormerStep):
                    formerState = self.hiddenStatList[j]
                    featureFunctionSum = self.getSumOfFeatureFuctions(thisState, formerState, thisObservation)
                    thisAlpha[n] += np.exp(featureFunctionSum) * formerAlpha[j]
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
            for n in range(len(self.hiddenStatList)):#遍历本步的所有隐藏状态
                thisState = self.hiddenStatList[n]
                for j in range(stateNumOfFormerStep):
                    formerState = self.hiddenStatList[j]
                    featureFunctionSum = self.getSumOfFeatureFuctions(thisState, formerState, thisObservation)
                    thisBeta[n] += np.exp(featureFunctionSum) * formerBeta[j]
            betaList[i] = thisBeta

        thisState = state_t
        thisObservation = observationList[t]
#         print(t, len(betaList), betaList)
        laterAlpha = betaList[t+1]
        stateNumOfFormerStep = len(laterAlpha)
        beta_t = 0#x_t处，隐藏状态取值为state_t时的前向变量取值
        for j in range(stateNumOfFormerStep):
            formerState = self.hiddenStatList[j]
            featureFunctionSum = self.getSumOfFeatureFuctions(thisState, formerState, thisObservation)
            beta_t += np.exp(featureFunctionSum) * laterAlpha[j]
        return beta_t

    def generatePossibleStateTrans(self, t):#生成第t步，可能的隐藏状态转移
        featureNames = []
        if t==1:
            featureNames = itertools.product(['*'], self.hiddenStatList)
        else:
            featureNames = itertools.product(self.hiddenStatList, self.hiddenStatList)
        featureNames = list(map(lambda x: ''.join(x), featureNames))
        featureNames = list(filter(lambda x: x in self.featureWeightMap, featureNames))
        return featureNames
    
    def generatePossibleStateFeatueNames(self, t, observation):
        featureNames = []
        if t==1:
            featureNames = itertools.product(['@'], [observation])
        else:
            featureNames = itertools.product(self.hiddenStatList, [observation])
        featureNames = list(map(lambda x: ''.join(x), featureNames))
        featureNames = list(filter(lambda x: x in self.featureWeightMap, featureNames))
        return featureNames
        
    #计算模板函数权重对应的梯度
    def calGrad4Weight(self, sentence):
        charList = '@' + sentence[0] + '@'
        tagList = '*' + sentence[1] + '#'
        sentenceLength = len(charList)
        tt1, tt2, tt3 = 0, 0, 0
        t1 = time.time()
        z_x = self.backwardAlgrithm('*', charList, 0)#计算配分函数的取值
        t2 = time.time()
        tt1 = t2-t1
        
        gradMap = {}
        for t in range(1, sentenceLength-1):
            thisState, formerState = tagList[t], tagList[t-1]
            thisObservation = charList[t]
            featureName2 = formerState + thisState
            featureName1 = thisState + thisObservation
            possibleFeatureNames = self.generatePossibleStateFeatueNames(t, thisObservation)
            if featureName1 in self.featureWeightMap:
#                 print("状态特征是", featureName1)
                gradMap[featureName1]  = gradMap.get(featureName1, 0) + self.ifFitFeatureTemplet(featureName1)
                
#                 print(possibleFeatureNames)
#                 t1 = time.time()
#                 FeatureFunctionValueAndMargProbList = self.calFeatureFunctionValueAndMargProbPar(possibleFeatureNames, charList, t)
#                 for line in FeatureFunctionValueAndMargProbList:
#                     featureName, prob = line[0], line[1]
#                     gradMap[featureName] = gradMap.get(featureName, 0) - prob/z_x
#                 t2 = time.time()
#                 tt2 += t2-t1
            t1 = time.time()
            for featureName in possibleFeatureNames:  
#                 print(gradMap.get(featureName, 0)) 
                gradMap[featureName] = gradMap.get(featureName, 0) - self.calFeatureFunctionValueAndMargProb(featureName, charList, t)/z_x
                t2 = time.time()
                tt2 += t2-t1
            possibleFeatureNames = self.generatePossibleStateTrans(t)               
            if featureName2 in self.featureWeightMap:
                gradMap[featureName2]  = gradMap.get(featureName2, 0) + self.ifFitFeatureTemplet(featureName2)
                
                t1 = time.time()
            for featureName in possibleFeatureNames:                
                gradMap[featureName] = gradMap.get(featureName, 0) - self.calFeatureFunctionValueAndMargProb(featureName, charList, t)/z_x                      
                t2 = time.time()
                tt3 += t2-t1
#         print("计算配分函数的耗时是", tt1)
#         print("计算状态特征边缘概率的耗时是", tt2)
#         print("计算状态转移特征边缘概率的耗时是", tt3)
        return gradMap
       
    # 基于更新规则更新权重
    def updateWeight(self, gradMap):
        for featureName in gradMap:
#             print(self.featureWeightMap.keys())
            if featureName in self.featureWeightMap:
                self.featureWeightMap[featureName] -= self.learningRate * gradMap[featureName]
#             else:
#                 print(featureName)

    #基于训练语料，估计CRF参数
    def fit(self, sentenceList):
        self.initParamWithTraingData(sentenceList)
        for epoch in range(self.epoch):
            for sentence in sentenceList:#遍历语料中的每一句话，训练模型
                t1 = time.time()
                gradMap = self.calGrad4Weight(sentence)#计算模板函数权重对应的梯度
                t2 = time.time()
#                 print("梯度是", list(gradMap.items()))
#                 if 'ES' in gradMap:
#                     print("grad is", gradMap['ES'])
                self.updateWeight(gradMap)#基于更新规则更新权重
                
                t3 = time.time()
    #             print("更新后的权重是", list(self.featureWeightMap.items())[:10])
                print(self.learningRate, "epoch:", epoch, "weight of 'ES':", self.featureWeightMap['ES'], 'time cost:',t2-t1, t3-t2)
            
    #基于观测值序列，也就是语句话的字符串列表，使用模型选出最好的隐藏状态序列，并按照分词标记将字符聚合成分词结果
    def predict(self, text): 
        statPathProbMap = {}#存储以各个初始状态打头的概率最大stat路径
        for stat in self.initStatProbDist:#遍历每一个初始状态
            statPath = stat#这是目前积累到的stat路径，也就是分词标记序列
            firstChar = text[0]
            conditionProbOfThisChar = self.charProbDistOfEachStat[stat].get(firstChar, 0.000001)
            statPathProb = self.initStatProbDist[stat] * conditionProbOfThisChar
            statPathProbMapOfThis = {}
            statPathProbMapOfThis[statPath] = statPathProb
            for i in range(1, len(text)):
                char  = text[i]
                tempPathProbMap = {}
                for statValue in self.statValueSet:
                    tempStatPath = statPath + statValue
                    statTrans = statPath[-1] + statValue
                    tempPathProb = statPathProbMapOfThis[statPath]* \
                           self.statTransProbMap.get(statTrans, 0.01)*\
                           self.charProbDistOfEachStat[statValue].get(char, 0.000001)
                    tempPathProbMap[tempStatPath] = tempPathProb
                bestPath = getKeyWithMaxValueInMap(tempPathProbMap)
                statPathProbMapOfThis[bestPath] = tempPathProbMap[bestPath]
                statPath = bestPath
            statPathProbMap[statPath] = statPathProbMapOfThis[statPath]
        bestPath = getKeyWithMaxValueInMap(statPathProbMap)
        res = mergeCharsInOneWord(text, bestPath)
        return res
                    

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
                
    
dataStr = """
我 S
喜 B
欢 E
吃 S
好 B
吃 M
的 E
， W
因 B
为 E
这 B
些 E
东 B
西 E
好 B
吃 E
。
"""

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
#                 corpus.append([tempSentence,tempTag])
                corpus.append([tempSentence[:10],tempTag[:10]])
#                 print("这句话是", [tempSentence,tempTag])
                tempSentence = []
                tempTag = []
                count += 1
                if count==sentenceNum:#如果积累的句子个数达到阈值，返回语料
                    return corpus
            else:
                line= line.split('\t')
#                 print(line)
                [char, tag] = line[0], line[2]#取出语料的文字和分词标记
                tempSentence.append(char)
                tempTag.append(tag)
            line = f.readline()
    return corpus
                      
import time
if __name__ == '__main__':
    fileName = r"msra_training.txt"
    sentenceNum = 50
    sentenceList = loadData(fileName, sentenceNum=sentenceNum)#加载语料
#     print(sentenceList)
    model = LinearChainCRF(epoch=10, learningRate=0.1)
    model.fit(sentenceList)
#     print("一个隐藏状态序列的概率是",
#           sentenceList[0],
#           model.calConditionalProbOfStates(sentenceList[0][1],
#                                                            sentenceList[0][0]))
    # res = model.predict(sentenceList[10][0])
    # print("分词结果是", res, "真实的分词结果是", )
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
   
