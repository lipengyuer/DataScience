# -*- coding: utf-8 -*-
#实现word2evc,用来训练一定维数的词向量
import random
import numpy as np
import copy
#直接使用神经网络来训练词向量
#考虑到语料可能非常大，无法一次性加载，这里使用的过程是:(1)逐行读取语料，统计词频数据，并删除生僻词语
#;（2）以行为单位，生成训练数据;(3)训练神经网络


class BPANN():
    
    def __init__(self,learningRate=.1, stepNum=1000, hiddenLayerStruct = [5, 5]):
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
        self.gradsList = []#在训练中记录各个神经元的参数对应的梯度，训练完成后，要清空这个变量,控制内存消耗
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
        self.gradsList = [None for _ in range(len(self.layerStruct))]#存储每一层的神经元与
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
            self.gradsList[i] = tempGradMatrix#收集这一层神经元与前一层神经元连接权重的梯度
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
            
    def updateWeightsWithGrad(self):
        for i in range(1, len(self.gradsList)):
            self.weightMatrixList[i-1] -= self.learningRate * self.gradsList[i]
    
    def calCost(self, predOutput, trainOutput):
        cost = 0.
        for i in range(len(trainOutput)):
            if predOutput[i]>0:
                cost += -trainOutput[i]*np.log2(predOutput[i])
        return cost
            
    def fit(self, trainInput, trainOutput):
        if self.classNum==None:
            self.initANNWeight(trainInput, trainOutput)
        for i in range(self.stepNum):#数据需要学习多次
            totalCost = 0.
            for n in range(0, len(trainInput)):#遍历样本
#                 print("神经网络正在学习第", n, "个训练数据。")
                thisInput = trainInput[n, :]
                thisOutPut = trainOutput[n, :]
                predOutput, outputOfEachLayer = self.predict4Train(thisInput)#基于当前网络参数计算输出
                totalCost += self.calCost(predOutput, thisOutPut)
                self.calGrad4Weights(predOutput, thisOutPut, outputOfEachLayer)
                self.updateWeightsWithGrad()
                # print("权重是", self.weightMatrixList[-1][0])
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
    
    
class SimpleWord2Vec():
    """"""
    def __init__(self, learningRate=0.001, window=3, min_count=1, del_top_N=100, modelFile='model.pkl'):
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
        self.ann = BPANN(learningRate=self.learningRate, stepNum=5, hiddenLayerStruct=[5])
        #开始逐行读取数据并训练神经网络
        with open(corpusFileName, 'r') as f:
            line = f.readline()
            count = 0
            while line!="":
                wordsInThisLine = line.replace('\n', '').split(' ')
                wordsInThisLine = list(filter(lambda x: len(x)>0, wordsInThisLine))
                trainingDataInput, trainingDataOutput, _ = self.orgniseTraningData(wordsInThisLine)
                # print("标签数据是", len(trainingDataOutput[0]), np.sum(trainingDataOutput, axis=1))
                # print("标签数据是", len(trainingDataInput[0]), np.sum(trainingDataInput, axis=1))

                count += 1
                print(_)
                print("正在学习第", count, '句。这个句子有', len(wordsInThisLine), '个词语,训练数据数量是', trainingDataInput.shape[0])
                self.ann.fit(trainingDataInput, trainingDataOutput)

                line = f.readline()
                self.generateVector4EachWord(corpusFileName)
                # print("更新后的结果是", self.word2VectorMap)
                self.save()
                if count%50==0:
                    break


    def generateVector4EachWord(self, corpusFileName):
        #获取每一个词语的词向量
        with open(corpusFileName, 'r') as f:
            line = f.readline()
            while line!="":
                wordsInThisLine = line.replace('\n', '').split(' ')
                wordsInThisLine = list(filter(lambda x: len(x) > 0, wordsInThisLine))
                trainingDataInput, _, fineWords = self.orgniseTraningData(wordsInThisLine)
                # print("词语", fineWords)
                for i in range(len(fineWords)):
                    word = fineWords[i]
                    input4ANN  = trainingDataInput[i]
                    self.word2VectorMap[word] = self.ann.getWordVector(input4ANN)
                line = f.readline()

    def getNearestWords(self, word, topN=10):
        distMap = {}
        if word not in self.vocabSet: return []
        print(self.word2VectorMap.keys())
        vector = self.word2VectorMap[word]
        for aword in self.word2VectorMap:
            print(aword, vector)
            distMap[aword] = np.sqrt(np.sum((vector-self.word2VectorMap[aword])**2))
        words = sorted(distMap.items(), key=lambda x: x[1], reverse=True)[:topN]
        return words

    def orgniseTraningData(self, wordList):
        trainingDataInputList, trainingDataOutputList = [], []
        fineWords = []
        for i in range(self.window, len(wordList)-self.window):
            targetWord = wordList[i]#需要预测的词语
            if targetWord not in self.wordOneHotVectorMap: continue#如果这个词语是生僻词语，跳过
            targetWordOneHot = self.wordOneHotVectorMap[targetWord]
            trainingDataOutputList.append(targetWordOneHot)
            contextWords = wordList[i-self.window:i] + wordList[i+1: i+1+ self.window]#上下文词语
            contextWordsOneHot = [self.wordOneHotVectorMap.get(word, np.zeros(self.vocabSize)) for word in contextWords]
#             print("上下文词语个数是", len(contextWords), len(contextWordsOneHot[0]))
            if np.sum(contextWordsOneHot)<=1: continue
            # print("输入的情况", np.sum(contextWordsOneHot))
            contextWordsOneHot = np.array(contextWordsOneHot).reshape((self.inputSizeOfNetwork))
            trainingDataInputList.append(contextWordsOneHot)
            fineWords.append(targetWord)
        trainingDataInputList, trainingDataOutputList = np.array(trainingDataInputList), np.array(trainingDataOutputList)
        return trainingDataInputList, trainingDataOutputList, fineWords
            
            
    #把分词后的语料组织成训练数据，这里针对CBOW
    def initVocab(self,corpusFileName):
        if self.vocabSet==None:#如果词汇表还没有初始化
            self.vocabSet = set({})
            wordFreqMap = {}
            with open(corpusFileName, 'r') as f:
                line = f.readline()
                while line!="":
                    wordsInThisLine = line.replace('\n', '').split(' ')
                    for word in wordsInThisLine: wordFreqMap[word] = wordFreqMap.get(word, 0) + 1
                    line = f.readline()
                #删除生僻词语
            wordFreqList = sorted(wordFreqMap.items(), key=lambda x: x[1])[:-self.del_top_N]
            for [word, freq] in wordFreqList:
                if freq >= self.min_count: self.vocabSet.add(word) 
        
        self.vocabSize = len(self.vocabSet)
        self.vocabList = list(self.vocabSet)
        self.inputSizeOfNetwork = 2*self.vocabSize * self.window
        print("aa词汇表的大小是", self.vocabSize, self.inputSizeOfNetwork)
        for i in range(self.vocabSize):
            oneHotVector = np.zeros(self.vocabSize)
            oneHotVector[i] = 1#这个词语的位置置为1,形成独热编码
            thisWord = self.vocabList[i]
            self.wordOneHotVectorMap[thisWord] = oneHotVector
    def save(self):
        import pickle
        with open(self.modelFile, 'wb') as f:
            pickle.dump(self, f)
            
import pickle
if __name__ == '__main__':
    corpusFile = '../../data/pku_training.txt'
    corpusFileNew = '../../data/pku_training_temp.txt'
    # with open(corpusFile, 'r') as f:
    #     lines = f.readlines()
    #
    # for i in range(0, len(lines), 500):
    #     with open(corpusFileNew, 'a+') as f:
    #         line = ' '.join(lines[i: i + 500]).replace("\n", '') + '\n'
    #         f.write(line)
    modelFile = 'model.pkl'
    model = SimpleWord2Vec(learningRate=0.1, window=2, min_count=10, del_top_N=1000, modelFile=modelFile)
    model.fit(corpusFileNew)

    model = pickle.load(open(modelFile, 'rb'))
    data = ['食品', '飞机', '‘', '倍', '出版社', '态度', '廉政', '西方']
    for word in data:
        nearWord = model.getNearestWords(word)
        print(word, '的关联词是', nearWord)
    
    
    
    