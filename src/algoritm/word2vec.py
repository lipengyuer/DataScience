# -*- coding: utf-8 -*-
#实现word2evc,用来训练一定维数的词向量
import random
import numpy as np
import copy
from multiprocessing import Pool
#直接使用神经网络来训练词向量
#考虑到语料可能非常大，无法一次性加载，这里使用的过程是:(1)逐行读取语料，统计词频数据，并删除生僻词语
#;（2）以行为单位，生成训练数据;(3)训练神经网络

#使用softmax的word2vec训练太慢了，这里使用负采样的方式来降低计算量。
#负采样是提升word2vec训练速度的3种经典策略中比较符合直觉的一种,不论是理论上还是实践中。

class BPANN():
    
    def __init__(self,learningRate=.1, stepNum=1000, hiddenLayerStruct = [5, 5], workerNum=3):
        self.workerNum = workerNum
        self.weights = None#一个三维矩阵，第一维对应神经网络的层数，第二维对应一层神经网络的神经元的序号，第三维对应
        #一个神经元接收的输入的序号。
        #这里为了方便，截距被当作一个取值固定的变量来处理，系数是1.模型输入后，会初始化这个向量
        self.learningRate = learningRate#学习率。这里每个参数的学习率是一样的；我们也可以为各个参数设置不同的学习率。
        #hiddenLayerStruct,隐藏层的结构，默认是两层，分别有5个神经元
        self.hiddenLayerStruct = hiddenLayerStruct
        self.layerStruct = [0] + hiddenLayerStruct + [0]#一个实用的ANN,需要一个假想的输入层，
        #和一个真的会用到的输出层。“真”和"假"会在后面的部分体现出来
        self.classNum = None
        self.featureNum = None
        self.weightMatrixList = None#存储各层神经元的权重矩阵
        self.stepNum = stepNum
        
    #初始化权重
    def initANNWeight(self, trainInput, trainOutput):
        self.classNum = len(trainOutput[0])#类别的个数
        self.featureNum = len(trainInput[0])#输入特征的个数
        self.weightMatrixList = []#各层的权重矩阵
        #输入层的神经元个数与特征数相同，这样第一层隐藏层就可以接收输入特征了；
        #输出层的神经元个数与类别个数相同，这样我们就可以把最后一层隐藏层的输出变换为我们需要的类别概率了。
        self.layerStruct[0], self.layerStruct[-1] = self.featureNum, self.classNum
        for i in range(1, len(self.layerStruct)):
            nodeNum_i = self.layerStruct[i]#本层神经元的个数
            inputNum = self.layerStruct[i-1]#上一层神经元的输出个数，也就是本层神经元的输入和数
            weighMatrix_i = []#本层神经元的权重向量组成的矩阵
            for j in range(nodeNum_i):#本层的每一个神经元都需要一个权重向量
                weights4Node_i_j = [random.uniform(-0.1,0.1) for _ in range(inputNum)] + [random.uniform(-0.1,0.1)]#这是一个神经元接收到输入信号后，对输入进行线性组合
                #使用的权重向量，每一个输入对应一个权重。最后加上的那个权重，是多项式的截距
                weighMatrix_i.append(weights4Node_i_j)
            weighMatrix_i = np.array(weighMatrix_i)#处理成numpy.array,后面我们会直接使用矩阵运算，这样可以减少代码量
            #同时利用numpy加快运算速度
            self.weightMatrixList.append(weighMatrix_i)#把这层的权重矩阵存储器起来
        #打印网络结构
        for i in range(len(self.layerStruct)):
            print("网络的第", i + 1 , '层节点数是', self.layerStruct[i])
            
    def predict4Train(self, inputData):#训练里使用的一个predict函数
        res = inputData
        outputOfEachLayer = [inputData]
        for i in range(1, len(self.layerStruct)):
            weightMatrix = self.weightMatrixList[i-1]
            res = np.concatenate((res,np.array([1])))#为截距增加一列取值为1的变量
            res = np.dot(weightMatrix, res)#计算线性组合的结果
            res = self.sigmod(res)#使用sigmod函数进行映射
            outputOfEachLayer.append(res)
        return res, outputOfEachLayer
    
    #基于本次计算的输出，以及当前网络参数，逐层计算各个参数对应的梯度
    def calGrad4Weights(self, predOutput, realOutput, outputOfEachLayer ):
        gradsList = [None for _ in range(len(self.layerStruct))]#存储每一层的神经元与
        #前一层神经元的连接权重对应的梯度
        self.errorFromLaterLayerNodeList = [None for _ in range(len(self.layerStruct))]#存储反向传播过程中
        #一个神经元接受的来自后面一层神经元的误差
        error = predOutput - realOutput#这是最后一层神经元的输出，与真实值的误差。是一个向量
        self.errorFromLaterLayerNodeList[-1] = error
        #计算一层节点与之前一层节点连接权重的梯度.并计算这一层节点反向传播给前一层每一个神经元的误差
        for i in range(len(self.layerStruct)-1, 0, -1):#从后向前遍历每一层神经元
            nodeNumOfThisLayer = self.layerStruct[i]#这一层节点的个数
            nodeNumOfFormerLayer = self.layerStruct[i-1]#前一层节点的个数
            error4EveryNode = self.errorFromLaterLayerNodeList[i]#这是后一层节点传播过来的误差数据
            weightMatrix4ThisLayer = self.weightMatrixList[i-1]#这一层神经元与前一层神经元的连接权重矩阵
            #现在计算连接权重对应的梯度
            inputOfThisLayer = outputOfEachLayer[i-1]#这一层神经元接收到的前一层神经元的输出，也就是这一层的输入
            outputOfThisLayer = outputOfEachLayer[i]#这一层神经元的输出
            tempGradMatrix = np.zeros(weightMatrix4ThisLayer.shape)
#             print("现在计算第", i, "层")
#             print(error4EveryNode)
            for j in range(nodeNumOfFormerLayer):#遍历每一个输入
                for n in range(nodeNumOfThisLayer):#遍历这一层的每一个节点
                    sumErrorFromFormerLayer = error4EveryNode[n]#前一层神经元传播给这个神经元的误差
                    tempGradMatrix[n, j] = inputOfThisLayer[j] * outputOfThisLayer[n] *\
                     (1 - outputOfThisLayer[n]) * sumErrorFromFormerLayer
            gradsList[i] = tempGradMatrix#收集这一层神经元与前一层神经元连接权重的梯度
            #开始计算这一层神经元传播给前一层神经元的误差
            tempErrorMatrix = copy.deepcopy(weightMatrix4ThisLayer)#存储每个神经元向前传播的误差
            errorArray4FormerLayer = []
            for j in range(nodeNumOfThisLayer):#遍历这一层的每一个神经元
                error4ThisNode = error4EveryNode[j]#取出这个神经元接收得到的来自后一层所有神经元的误差
                for n in range(nodeNumOfFormerLayer):#遍历前一层的每一个神经元
                    tempErrorMatrix[j,n] *= error4ThisNode#权重乘以对应的误差
            for n in range(nodeNumOfFormerLayer):#遍历前一层的每一个神经元
                error4FormerLayerNode = sum(tempErrorMatrix[:, n])#神经元接收的来自后一层神经元传播的误差
                errorArray4FormerLayer.append(error4FormerLayerNode)
            self.errorFromLaterLayerNodeList[i-1] = errorArray4FormerLayer
        return gradsList
            
    def updateWeightsWithGrad(self, gradsList, batchSize):
        for i in range(1, len(gradsList)):
            self.weightMatrixList[i-1] -= self.learningRate/batchSize * gradsList[i]
    
    def calCost(self, trainInput, trainOutput):
        cost = 0.
        for n in range(trainInput.shape[0]):
            thisInput, thisOutput = trainInput[n, :], trainOutput[n, :]
            predOutput, _ = self.predict4Train(thisInput)
            for i in range(len(predOutput)):
                if predOutput[i]>0:
                    cost += -thisOutput[i]*np.log2(predOutput[i])
        return cost
    
    def addTwoList(self, list1, list2):
        res = []
        for i in range(len(list1)): 
#             print(list1[i], list2[i])
            if type(list1[i])!=type(None): res.append(list1[i]+list2[i])
            else:res.append(None)
        return res
    
    def calGrad4Batch(self, inputBatch, outputBatch):
        predOutput, outputOfEachLayer = self.predict4Train(inputBatch[0])#基于当前网络参数计算输出
        gradList = self.calGrad4Weights(predOutput, outputBatch[0], outputOfEachLayer)
        for i in range(1, inputBatch.shape[0]):
            predOutput, outputOfEachLayer = self.predict4Train(inputBatch[0])#基于当前网络参数计算输出
            gradTemp = self.calGrad4Weights(predOutput, outputBatch[0], outputOfEachLayer)  
            gradList = self.addTwoList(gradList, gradTemp)
        return gradList
        
    def fit(self, trainInput, trainOutput):
        if self.classNum==None:
            self.initANNWeight(trainInput, trainOutput)
        for i in range(self.stepNum):#数据需要学习多次
            totalCost = 0.
            pool = Pool(self.workerNum)
            batchSize = int(trainInput.shape[0]/self.workerNum)
            if batchSize==0: batchSize=1
            resList = []
            for n in range(0, len(trainInput), batchSize):#遍历样本
#                 print("神经网络正在学习第", n, "个训练数据。")
                inputBatch, outputBatch = trainInput[n:n+batchSize, :], trainOutput[n:n+batchSize, :]
                res = pool.apply_async(self.calGrad4Batch, args=(inputBatch, outputBatch))
                resList.append(res)
            pool.close()
            pool.join()
            for res in resList:
                gradList = res.get()
                self.updateWeightsWithGrad(gradList, batchSize)
#                 print("权重是", self.weightMatrixList[-1][0])
            totalCost = self.calCost(trainInput, trainOutput)
            print('step', i, "cost is", totalCost)

    #计算一个观测值的输出
    def predictOne(self, inputData):
        res = inputData
        for i in range(1, len(self.layerStruct)):
            weightMatrix = self.weightMatrixList[i-1]
            res = np.concatenate((res,np.array([1])))#为截距增加一列取值为1的变量
            res = np.dot(weightMatrix, res)#计算线性组合的结果
            res = self.sigmod(res)#使用sigmod函数进行映射
        res = list(res)
        maxV = np.max(res)
        label = [0 for _ in range(self.classNum)]
        label[res.index(maxV)] = 1 
        return label
    
    #计算得到网络的最后一层隐藏层的输出，也就是词向量
    def getWordVector(self, inputData):
        res = inputData
        # print("计算词向量。")
        for i in range(1, len(self.layerStruct)-1):
            weightMatrix = self.weightMatrixList[i-1]
            res = np.concatenate((res,np.array([1])))#为截距增加一列取值为1的变量
            res = np.dot(weightMatrix, res)#计算线性组合的结果
            res = self.sigmod(res)#使用sigmod函数进行映射
        res = np.array(res)
        return res
    
    #sigmod函数
    def sigmod(self, x):
        res = 1/(1 + np.exp(-x))
        return res
    
import time 
class SimpleWord2Vec():
    """"""
    def __init__(self, learningRate=0.001, window=3, min_count=1, del_top_N=100, \
                 modelFile='model.pkl', workerNum=3, corpusSize=5, maxVocabSize=7000,
    negativeNum = 10, stepNum=1):
        self.stepNum = stepNum
        self.negativeNum=negativeNum
        self.maxVocabSize = maxVocabSize
        self.corpusSize = corpusSize
        self.workerNum = workerNum
        self.learningRate = learningRate
        self.window = window#需要关注的上下文词语的个数
        self.min_count = min_count#语料中，出现个数小于这个值的词语不进入词汇表
        self.del_top_N = del_top_N
        self.word2VectorMap = {}
        self.wordOneHotVectorMap = {}#用来存储词语-独热编码。因为词表可能会比较大，如果一股脑把整个训练语料中的词语全都转换为独热编码
        #内存就完蛋了;训练的时候，把分好组的词语列表转换并拼接成需要的向量即可
        self.vocabSet = None
        self.vocabSize = None
        self.inputSizeOfNetwork = None#神经网络的输入的维度，由于需要大量使用，这里直接记录下来
        self.ann = None
        self.modelFile = modelFile

    def fit(self, corpusFileName):
        self.initVocab(corpusFileName)
        self.ann = BPANN(learningRate=self.learningRate, stepNum=self.stepNum, hiddenLayerStruct=[20],\
                         workerNum=self.workerNum)
        #开始逐行读取数据并训练神经网络
        epochNum = 10
        for epoch in range(epochNum):
            with open(corpusFileName, 'r') as f:
                lines = f.readlines()[:self.corpusSize]
                count = 0
                # while line!="":
                for line in lines:
                    t1 = time.time()
                    print("数据预处理。")
                    wordsInThisLine = line.replace('\n', '').split(' ')[1:]
                    wordsInThisLine = list(filter(lambda x: len(x)>2, wordsInThisLine))
                    # print(wordsInThisLine[:100])
                    trainingDataInput, trainingDataOutput, _ = self.orgniseTraningData(wordsInThisLine)
                    # print("标签数据是", len(trainingDataOutput[0]), np.sum(trainingDataOutput, axis=1))
                    # print("标签数据是", len(trainingDataInput[0]), np.sum(trainingDataInput, axis=1))
    
                    count += 1
                    # print(_[:100])
                    print(epoch, "轮。正在学习第", count, '句。这个句子有', len(wordsInThisLine), '个词语,训练数据数量是', trainingDataInput.shape[0])
                    self.ann.fit(trainingDataInput, trainingDataOutput)

                    t2 = time.time()
                    print("耗时是",int(t2-t1))#, "更新后的结果是", list(self.word2VectorMap.items())[:10])
                    self.save()
                    if count%self.corpusSize==0:
                        break
                    if count%20==0:
                        data = list(self.word2VectorMap.keys())[:10]
                        print("正在计算每一个词语的词向量")
                        t1 = time.time()
                        self.generateVector4EachWord(corpusFileName)
                        t2 = time.time()
                        print("开始展示部分词语的关联词", int(t2-t1))
                        for word in data:
                            nearWord = self.getNearestWords(word)
                            print(word, '的关联词是', list(map(lambda x: x[0], nearWord)))

    def generateVector4EachWord(self, corpusFileName):
        #获取每一个词语的词向量
        count = 0
        inputList = []
        wordList = []
        wordFilter = set({})
        with open(corpusFileName, 'r') as f:
            lines = f.readlines()[:self.corpusSize]
            for line in lines:
            # while line!="":
                count += 1
                wordsInThisLine = line.replace('\n', '').split(' ')[1:]
                wordsInThisLine = list(filter(lambda x: len(x) > 2, wordsInThisLine))
                if len(wordsInThisLine)<500: continue
                print("正在生成词向量，读取的是第", count, "行语料。")
                trainingDataInput, _, fineWords = self.orgniseTraningDataSimple(wordsInThisLine, filter=wordFilter)
                wordFilter  = wordFilter | set(fineWords)
                # print("词语", fineWords)
                for i in range(len(fineWords)):
                    word = fineWords[i]
                    input4ANN  = trainingDataInput[i]
                    inputList.append(input4ANN)
                    wordList.append(word)

                if count%self.corpusSize==0:
                    break
        pool = Pool(self.workerNum)
        resList = []
        batchSize = int(len(inputList)/self.workerNum)
        for i in range(0,len(wordList), batchSize):
            res = pool.apply_async(self.getWordVector, args=(wordList[i: i+batchSize],
                                                             inputList[i: i+batchSize], i))
            resList.append(res)
        pool.close()
        pool.join()
        for res in resList:
            res = res.get()
            for line in res:
                [word, vector] = line
                self.word2VectorMap[word] = vector

    def getWordVector(self, wordList, inputList, no):
        print("正在计算第", no, '个词向量')
        resList = []
        for i in range(len(wordList)):
            vector = self.ann.getWordVector(inputList[i])
            resList.append([wordList[i], vector])
        return resList
        
    def getNearestWords(self, word, topN=10):
        distMap = {}
        if word not in self.vocabSet: return []
#         print(self.word2VectorMap.keys())
        vector = self.word2VectorMap[word]
        for aword in self.word2VectorMap:
            distMap[aword] = np.sqrt(np.sum((vector-self.word2VectorMap[aword])**2))
        words = sorted(distMap.items(), key=lambda x: x[1])[:topN]
        return words
    
    def negativeSampling(self, word):#从词汇表中抽样，除了这个词语
        resList = []
        rList = [random.uniform(0, 1) for _ in range(self.negativeNum)]
        rList  =sorted(rList)
        index = 0
        for [word, startEnd] in self.rSpan4EachWord:
            if index==len(rList)-1: break
            r = rList[index]
            if startEnd[0] <= r <= startEnd[1]:
                resList.append(word)
                index += 1
        if word in resList: resList.remove(word)
        return resList

    def orgniseTraningDataSimple(self, wordList, filter = None):
        trainingDataInputList, trainingDataOutputList = [], []
        fineWords = []
        for i in range(self.window, len(wordList)-self.window):
            targetWord = wordList[i]#需要预测的词语
            if filter!=None and targetWord in filter: continue
            #构造正例
            if targetWord not in self.wordOneHotVectorMap: continue#如果这个词语是生僻词语，跳过
            targetWordOneHot = self.wordOneHotVectorMap[targetWord]
            contextWords = wordList[i-self.window:i] + wordList[i+1: i+1+ self.window]#上下文词语
            contextWordsOneHot = [self.wordOneHotVectorMap.get(word, np.zeros(self.vocabSize)) for word in contextWords]
#             print("上下文词语个数是", len(contextWords), len(contextWordsOneHot[0]))
            if np.sum(contextWordsOneHot)<=2: continue
            # print("输入的情况", np.sum(contextWordsOneHot))
            positiveSample = [targetWordOneHot] + contextWordsOneHot
            positiveSample = np.array(positiveSample).reshape((self.inputSizeOfNetwork))
            trainingDataInputList.append(positiveSample)
            trainingDataOutputList.append([1, 0])#正例
            fineWords.append(targetWord)
            
        trainingDataInputList = np.array(trainingDataInputList)
        return trainingDataInputList, trainingDataOutputList, fineWords

    def worker2OrgniseTraningData(self, start, end,  wordList, filter):
        trainingDataInputList, trainingDataOutputList = [], []
        fineWords = []
        end = min(end, len(wordList))
        for i in range(start, end):
            targetWord = wordList[i]
            if filter != None and targetWord in filter: continue
            # 构造正例
            if targetWord not in self.wordOneHotVectorMap: continue  # 如果这个词语是生僻词语，跳过
            targetWordOneHot = self.wordOneHotVectorMap[targetWord]
            contextWords = wordList[i - self.window:i] + wordList[i + 1: i + 1 + self.window]  # 上下文词语
            contextWordsOneHot = [self.wordOneHotVectorMap.get(word, np.zeros(self.vocabSize)) for word in contextWords]
            #             print("上下文词语个数是", len(contextWords), len(contextWordsOneHot[0]))
            if np.sum(contextWordsOneHot) <= 2: continue
            # print("输入的情况", np.sum(contextWordsOneHot))
            positiveSample = [targetWordOneHot] + contextWordsOneHot
            positiveSample = np.array(positiveSample).reshape((self.inputSizeOfNetwork))
            trainingDataInputList.append(positiveSample)
            trainingDataOutputList.append([1, 0])  # 正例
            fineWords.append(targetWord)

            # 接下来构造负例
            negativeWords = self.negativeSampling(targetWord)
            for word in negativeWords:
                targetWordOneHot = self.wordOneHotVectorMap[word]
                negativeSample = [targetWordOneHot] + contextWordsOneHot
                trainingDataInputList.append(negativeSample)
                trainingDataOutputList.append([0, 1])  # 负例
        return trainingDataInputList, trainingDataOutputList, fineWords

    def orgniseTraningData(self, wordList, filter = None):
        trainingDataInputList, trainingDataOutputList = [], []
        fineWords = []
        pool = Pool(self.workerNum)
        resList = []
        step =  int(len(wordList)/self.workerNum)
        for i in range(self.window, len(wordList)-self.window, step):
            res = pool.apply_async(self.worker2OrgniseTraningData, args=(i, i+step,wordList, filter ))
            resList.append(res)
        pool.close()
        pool.join()
        for res in resList:
            InputList, OutputList, fineWordsTemp = res.get()
            trainingDataInputList += InputList
            trainingDataOutputList += OutputList
            fineWords += fineWordsTemp

        trainingDataInputList, trainingDataOutputList = np.array(trainingDataInputList), np.array(trainingDataOutputList)
        return trainingDataInputList, trainingDataOutputList, fineWords
            
            
    #把分词后的语料组织成训练数据，这里针对CBOW
    def initVocab(self,corpusFileName):
        if self.vocabSet==None:#如果词汇表还没有初始化
            self.vocabSet = set({})
            wordFreqMap = {}
            count = 0
            with open(corpusFileName, 'r') as f:
                lines = f.readlines()[:self.corpusSize]
                for line in lines:
                # while line!="":
                    wordsInThisLine = line.replace('\n', '').split(' ')[1:]
                    for word in wordsInThisLine: wordFreqMap[word] = wordFreqMap.get(word, 0) + 1
                    # line = f.readline()
                    count += 1
                    if count%self.corpusSize==0: break
                #删除生僻词语
            wordFreqList = sorted(wordFreqMap.items(), key=lambda x: x[1])[:-self.del_top_N]
            wordFreqList = wordFreqList[-self.maxVocabSize:]
            finalWordFreqMap = {}
            for [word, freq] in wordFreqList:
                if freq >= self.min_count: 
                    self.vocabSet.add(word) 
                    finalWordFreqMap[word] = freq

        self.vocabSize = len(self.vocabSet)
        self.vocabList = list(self.vocabSet)
        self.inputSizeOfNetwork = 2*self.vocabSize * self.window + self.vocabSize#带有负采样策略的情况下，
        #目标自于和上下文词语都会作为神经网络的输入
        print("aa词汇表的大小是", self.vocabSize, self.inputSizeOfNetwork)
        for i in range(self.vocabSize):
            oneHotVector = np.zeros(self.vocabSize)
            oneHotVector[i] = 1#这个词语的位置置为1,形成独热编码
            thisWord = self.vocabList[i]
            self.wordOneHotVectorMap[thisWord] = oneHotVector
            
        self.rSpan4EachWord = []#存储每一个词语对应的r值区间。0<=r<=1，每个词语分得与词频成正比长度的一段。
        #负采样的时候，均匀随机数如果落在一个词语对应的线段内，就挑选这个词语
        sumOfFreq = 0
        for word in finalWordFreqMap: sumOfFreq += (finalWordFreqMap[word])**0.75
        tempSpan = np.array([0, 0])
        for word in finalWordFreqMap:
            tempSpan[0] = tempSpan[1]
            tempSpan[1] = tempSpan[1] + finalWordFreqMap[word]**0.75/sumOfFreq
            self.rSpan4EachWord.append([word, tempSpan])

    def save(self):
        import pickle
        with open(self.modelFile, 'wb') as f:
            pickle.dump(self, f)

def addLine(line, fileName):
    with open(fileName, 'a+') as f:
        f.write(line)

allpath=[]
allname=[]

import os
def getallfile(path):
    allfilelist=os.listdir(path)
    # 遍历该文件夹下的所有目录或者文件
    for file in allfilelist:
        filepath=os.path.join(path,file)
        # 如果是文件夹，递归调用函数
        if os.path.isdir(filepath):
            getallfile(filepath)
        # 如果不是文件夹，保存文件路径及文件名
        elif os.path.isfile(filepath):
            allpath.append(filepath)
            allname.append(file)
    return allpath, allname

marks = {'。', '.', '‘', '\'', '"', '“', ',', '，', "：", ':', '<', '>'
                   , '《', '》', '！', '!', '?', '？'}
def  filterMarks(words):
    res = ''
    for word in words:
        if word not in marks:
            res += " " + word
    return res

def preprocessData():
    fileName = r'../../data/msr_training.txt'
    targetFile = r'../../data/msr_training_temp.txt'
    with open(fileName, "r") as f:
        lines = f.readlines()
        lines = map(lambda x: filterMarks(x.split(' ')), lines)
        tempLines = []
        for line in lines:
            if len(tempLines)==100:
                addLine(''.join(tempLines).replace('\n', ''), targetFile)
                tempLines = []
            tempLines.append(line)


import pickle
if __name__ == '__main__':
    corpusFileNew = r'../../data/msr_training_temp.txt'
    # preprocessData()
    modelFile = 'model.pkl'
    model = SimpleWord2Vec(learningRate=0.1, window=2, min_count=4, corpusSize=1000,
                           del_top_N=1000, modelFile=modelFile, workerNum=8, maxVocabSize=5000,
                           negativeNum=10, stepNum=3)
    model.fit(corpusFileNew)

    # model = pickle.load(open(modelFile, 'rb'))
    # data = list(model.word2VectorMap.keys())[:10]
    # print("正在计算每一个词语的词向量")
    # model.generateVector4EachWord(corpusFileName)
    # print("开始展示部分词语的关联词")
    # for word in data:
    #     nearWord = self.getNearestWords(word)
    #     print(word, '的关联词是', list(map(lambda x: x[0], nearWord)))
    
    
    
    