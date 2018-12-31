#条件随机场
#实现CRF的训练和使用，语料的预处理，标注结果的评估

import copy
import numpy as np
import random

class LinearChainCRF():
    
    def __init__(self):
        #定义CRF的参数
        #特征模板。如果没有设置，就是用使用默认的线性链条件随机场的模板规则，即只考虑当前
        #观测值x_t，当前隐藏状态y_t，以及前一个隐藏状态y_before_t
        self.hiddenStatList = None#隐藏状态的取值空间
        self.minWordNum = None#语料中的词语，出现次数小于这个阈值的，不会用于特征模板
        self.minCharNum = None#语料中的字符，出现次数小于这个阈值的，不会用于模板
        self.gradListOfWeights = None#训练过程中，存储各个模板函数权重的梯度
        self.learningRate = None#模型训练的学习率，这里为了简单，使用一个统一的
        self.statTransMap = None#存储隐藏状态之间的转换
        self.hiddenStateNum = None
        self.statObserveMap = None#存储各个隐藏状态与观测值
        self.weightOfFeatures =  None#每一个模板函数的权重
        self.featureFunctionNum = None#特征函数的个数
        self.featureFunctionNameIndex = None#存储每一个特征函数在权重向量中的位置
        #为了简单，这里暂时不加正则化

    #基于模板规则，计算各个特征函数，在各个观测值处的取值
    def calFeatureFunctionValueArray4LinearChainCRF(self, stateList, observationList):
        featureFunctionValue = np.zeros(self.featureFunctionNum)#每一个特征函数的取值
        charList = observationList
        tagList = stateList
        lengthOfSentence = len(charList)
        # print(stateList)
        # print(observationList)
        for i in range(1, lengthOfSentence-1):#遍历观测序列上的每一个观测值
            statTransFeature = tagList[i-1:i+1]#线性链CRF的特征函数只有两种
            statFeature = tagList[i] + charList[i]
            # print('asd',statTransFeature)
            indexOfStatTransFeature = self.featureFunctionNameIndex[statTransFeature]
            # print(featureFunctionValue)
            featureFunctionValue[indexOfStatTransFeature] += self.ifFitStatTransFeatureTemplet(statTransFeature)
            indexOfStatFeature = self.featureFunctionNameIndex[statFeature]
            featureFunctionValue[indexOfStatFeature] += self.ifFitStatTransFeatureTemplet(statTransFeature)
        return featureFunctionValue

    #判断当前状态转换是否符合某一个特征模板，返回值为这个特征函数的取值
    def ifFitStatTransFeatureTemplet(self, statTransFeature):
        if statTransFeature in self.statTransMap:
            return 1.
        else:
            return 0.
    def ifFitStatFeatureTemplet(self, statFeature):
        if statFeature in self.statObserveMap:
            return 1.
        else:
            return 0.

    #基于超参数和训练数据，初始化CRF的参数
    def initParamWithTraingData(self, traningData):
        self.statTransMap, self.statObserveMap = {}, {}
        self.featureFunctionNameIndex = {}
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
                self.statTransMap[statTransFeature] = self.statTransMap.get(statTransFeature, 0) + 1
                self.statObserveMap[statFeature] = self.statObserveMap.get(statFeature, 0) + 1
        print("可用的状态转移有", self.statTransMap)
        print("隐藏状态-观测值:",self.statObserveMap)
        featureNameList = list(self.statTransMap.keys()) + list(self.statObserveMap.keys())
        self.featureFunctionNum = len(featureNameList)
        self.hiddenStatList = list(hiddenStateSet)
        self.hiddenStateNum = len(hiddenStateSet)
        self.weightOfFeatures = np.array([random.uniform(-0.1, 0.1) for _ in range(self.featureFunctionNum)])
        for i in range(self.featureFunctionNum):
            # if featureNameList[i]=='@第':
            #     print(featureNameList[i])
            self.featureFunctionNameIndex[featureNameList[i]] = i
        print("特征函数的个数是", self.featureFunctionNum)

    #已知CRF参数，以及一个观测序列，计算一个隐藏状态序列的概率
    def calConditionalProbOfStates(self, stateList, observationList):
        observationList = '@' + observationList + '@'#为首位和末尾观测值的模板添加占位符号
        stateList = '*' + stateList+ '#'
        lengthOfSentence = len(stateList)
        featureValueArray = self.calFeatureFunctionValueArray4LinearChainCRF(stateList, observationList)
        numerator = np.exp(sum(self.weightOfFeatures*featureValueArray))

        #用前向算法，计算分布Z(x)
        scoreOfEachStateChain = np.ones(self.hiddenStateNum)#以某个隐藏状态结尾的
        tempResList = [copy.deepcopy(scoreOfEachStateChain) for _ in range(lengthOfSentence)]
        #state序列，对应的特征函数取值和
        for i in range(1, len(stateList)-1):#遍历每一个观测值和隐藏状态
            scoreOfLastStep = tempResList[i-1]#上一步一算得到的，各组路径的得分和
            print(scoreOfLastStep)
            scoreOfThisStep = tempResList[i]#这一步，各组路径的得分和
            for j in range(self.hiddenStateNum):#遍历每一个隐藏状态取值
                thisState = self.hiddenStatList[j]#这个状态的取值
                for n in range(self.hiddenStateNum):#遍历上一步的所有路径的得分
                    lastState = self.hiddenStatList[n]#上一步的一个状态的取值
                    stateTrans = lastState + thisState
                    stateTransFeatureValue = self.ifFitStatTransFeatureTemplet(stateTrans)
                    scoreOfThisStep[j] += scoreOfLastStep[n] * np.exp(stateTransFeatureValue)
                thisObservation = observationList[i]
                stateFeature = thisState + thisObservation
                scoreOfThisStep[j] *= np.exp(self.ifFitStatFeatureTemplet(stateFeature))
        denominator = np.sum(tempResList[-2])
        prob = numerator/denominator
        print("条件概率的分子是", numerator, '分母是',denominator, prob)
        return prob

    #已知CRF参数，以及一个观测序列，计算一个隐藏状态序列的概率
    def calSecondPartOfGrads(self, stateList, observationList):
        observationList = '@' + observationList + '@'#为首位和末尾观测值的模板添加占位符号
        stateList = '*' + stateList+ '#'
        lengthOfSentence = len(stateList)
        featureValueArray = self.calFeatureFunctionValueArray4LinearChainCRF(stateList, observationList)
        numerator = np.exp(sum(self.weightOfFeatures*featureValueArray))

        #用前向算法，计算分布Z(x)
        secondPartOfGrads = np.ones(self.hiddenStateNum)#梯度公式中，第二项的取值
        tempResList = [copy.deepcopy(secondPartOfGrads) for _ in range(lengthOfSentence)]
        #state序列，对应的特征函数取值和
        for i in range(1, len(stateList)-1):#遍历每一个观测值和隐藏状态
            scoreOfLastStep = tempResList[i-1]#上一步一算得到的，各组路径的得分和
            scoreOfThisStep = tempResList[i]#这一步，各组路径的得分和
            for j in range(self.hiddenStateNum):#遍历每一个隐藏状态取值
                thisState = self.hiddenStatList[j]#这个状态的取值
                for n in range(self.hiddenStateNum):#遍历上一步的所有路径的得分
                    lastState = self.hiddenStatList[n]#上一步的一个状态的取值
                    stateTrans = lastState + thisState
                    stateTransFeatureValue = self.ifFitStatTransFeatureTemplet(stateTrans)
                    scoreOfThisStep[j] += scoreOfLastStep[n] * np.exp(stateTransFeatureValue)
                thisObservation = observationList[i]
                stateFeature = thisState + thisObservation
                scoreOfThisStep[j] *= np.exp(self.ifFitStatFeatureTemplet(stateFeature))
        denominator = np.sum(tempResList[-2])
        prob = numerator/denominator
        print("条件概率的分子是", numerator, '分母是',denominator, prob)
        return prob

    #计算模板函数权重对应的梯度
    def calGrad4Weight(self, sentence):
        pass

    # 基于更新规则更新权重
    def updateWeight(self):
        pass

    #基于训练语料，估计CRF参数
    def fit(self, sentenceList):
        self.initParamWithTraingData(sentenceList)
        # for sentence in sentenceList:#遍历语料中的每一句话，训练模型
        #     self.calGrad4Weight(sentence)#计算模板函数权重对应的梯度
        #     self.updateWeight()#基于更新规则更新权重
            
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
                corpus.append([tempSentence,tempTag])
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
    print(sentenceList)
    model = LinearChainCRF()
    model.fit(sentenceList)
    print("一个隐藏状态序列的概率是",
          sentenceList[0],
          model.calConditionalProbOfStates(sentenceList[0][1],
                                                           sentenceList[0][0]))
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
   
