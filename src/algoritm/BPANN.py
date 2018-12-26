'''
Created on 2018年11月20日

@author: pyli
'''
#一个BP神经网络
import random
import numpy as np
import copy

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
#             print(i, res)
            outputOfEachLayer.append(res)
#         print(outputOfEachLayer)
        return res, outputOfEachLayer
    
    #基于本次计算的输出，以及当前网络参数，逐层计算各个参数对应的梯度
    def calGrad4Weights(self, predOutput, realOutput, outputOfEachLayer ):
        self.gradsList = [None for _ in range(len(self.layerStruct))]#清空梯度列表
        self.gradsListPlas = [None for _ in range(len(self.layerStruct))]#清空梯度列表
        error = predOutput - realOutput#这是最后一层神经元的输出，与真实值的误差。是一个向量
        
        #计算最后一层节点与导数第二层节点连接权重的梯度.单独计算的原因是，这部分权重的梯度只考虑一个输出节点，而不是像其他
        #层一样，考虑后一层的所有梯度
        weightMatrix = self.weightMatrixList[-1]#取出连接权重矩阵.这个矩阵的一列，是最后一层的一个神经元，与前面一层的
        #所有神经元的连接权重
        tempGradMaxtrix = copy.deepcopy(weightMatrix)*0#初始化一个梯度矩阵
        tempGradMaxtrixPlas = np.ones(tempGradMaxtrix.shape)
        for j in range(self.layerStruct[-1]):#遍历最后一层神经元的每一个节点
            tempGradMaxtrix[j, :] = weightMatrix[j, :] * error[j]#每一个节点对应的连接权重，乘以这个节点的输出误差
            tempGradMaxtrixPlas[j, :] *= error[j]
            
        self.gradsListPlas[-1] = tempGradMaxtrixPlas
        self.gradsListPlas[-2] = tempGradMaxtrix
        self.gradsList[-1] = tempGradMaxtrix#最后一层神经元的输出对应的误差
        for i in range(len(self.layerStruct)-2, 0, -1):#从倒数第二层开始，向前遍历每一层神经元
            gradArrayFromLaterLayer = self.gradsList[i+1]#后面一层的神经元对应的误差;权重列表比神经元层数少1.
            #这是本层节点与后面一层的所有节点的连接权重矩阵;矩阵的一列，是本层的一个节点与后面一层的所有节点的连接权重
            weightMatrix = self.weightMatrixList[i-1]#这个是本层神经元与前面一层神经元的连接权重矩阵
            tempGradMaxtrix = copy.deepcopy(weightMatrix)*0#初始化一个梯度矩阵
            tempGradMaxtrixPlas = np.ones(tempGradMaxtrix.shape)
            for n in range(self.layerStruct[i]):#遍历这一层的每一个节点
                totalErrorFromLaterLayerNode = np.sum(gradArrayFromLaterLayer[:, n])#来自于这个节点相连接的
                #后一层所有节点传播过来的误差
                tempGradMaxtrix[n, :] = weightMatrix[n, :]*totalErrorFromLaterLayerNode 
                tempGradMaxtrixPlas[n, :] *= totalErrorFromLaterLayerNode
            self.gradsList[i] = tempGradMaxtrix#保留这一层节点与前一层节点的连接权重对应的梯度 
            self.gradsListPlas[i] = tempGradMaxtrixPlas
#         print("aaa", self.gradsListPlas)
#         print('grad1', self.gradsList)
        self.gradsList = self.gradsListPlas
        for i in range(1, len(self.gradsListPlas)):#遍历每一层神经元
            gradMatrix = self.gradsListPlas[i]
            inputOfThisLayer = outputOfEachLayer[i-1]#这一层神经元的输入
            outputOfThisNode = outputOfEachLayer[i]
            inputOfThisLayer = np.concatenate((inputOfThisLayer,np.array([1])))
            outputOfThisNode = np.concatenate((outputOfThisNode,np.array([1])))
#             print(gragMatrix)
#             print(i, inputOfThisLayer)
            for j in range(len(gradMatrix[0,:])):#遍历上一层的神经元，把每个神经元的输出乘到梯度上
                gradMatrix[:, j] *= inputOfThisLayer[j]
            for j in range(len(gradMatrix[:,0])):#遍历这一层的每一个神经元，把每个神经元乘到梯度上
                gradMatrix[j, :] *= outputOfThisNode[j]
            
    def updateWeightsWithGrad(self):

#         print('grad2', self.gradsList)
        for i in range(1, len(self.gradsListPlas)):
#             print(self.gradsListPlas[i])
            self.weightMatrixList[i-1] -= self.learningRate * self.gradsListPlas[i]
    
    
    def calCost(self, predOutput, trainOutput):
        cost = 0.
        for i in range(len(trainOutput)):
            if predOutput[i]>0:
                cost += -trainOutput[i]*np.log2(predOutput[i])
        return cost
            
    def fit(self, trainInput, trainOutput):
        self.initANNWeight(trainInput, trainOutput)
        totalCostList = []
        for _ in range(self.stepNum):#数据需要学习多次
#             print(_, "w", self.weightMatrixList[0][0,:])
            totalCost = 0.
            for n in range(0, len(trainInput)):#遍历样本
                thisInput = trainInput[n, :]
                thisOutPut = trainOutput[n, :]

                predOutput, outputOfEachLayer = self.predict4Train(thisInput)#基于当前网络参数计算输出
#                 print('thisInput', thisInput)
#                 print('thisOutPut', thisOutPut)
#                 print('outputOfEachLayer', outputOfEachLayer)
#                 print('oldWeights', self.weightMatrixList)
                totalCost += self.calCost(predOutput, thisOutPut)
                self.calGrad4Weights(predOutput, thisOutPut, outputOfEachLayer)
                self.updateWeightsWithGrad()
#                 print('newWeights', self.weightMatrixList)
                

#                 print(self.weightMatrixList)
#                 break
            print("cost is", totalCost)
            totalCostList.append(totalCost)
        import matplotlib.pyplot as plt
        plt.plot(totalCostList)
        plt.show()
                              
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
    
    def predict(self, inputList):
        resList = []
        for line in inputList:
            res = self.predictOne(line)
            resList.append(res)
        return resList
    
    #sigmod函数
    def sigmod(self, x):
        res = 1/(1 + np.exp(-x))
        return res
    
    #统计并打印混淆矩阵
    def showConfusionMaxtrix(self, predOutput, realOutput):
        confusionMatrix = np.zeros((self.classNum, self.classNum))
        for i in range(len(predOutput)):
            output = list(realOutput[i])
            classIndex = output.index(1)
            confusionMatrix[classIndex, :] += predOutput[i]
        print("混淆矩阵:")
        print("列表示预测类别;行表示真实类别")
        print(confusionMatrix)

dataStr = """5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
4.6,3.1,1.5,0.2,Iris-setosa
5.0,3.6,1.4,0.2,Iris-setosa
5.4,3.9,1.7,0.4,Iris-setosa
4.6,3.4,1.4,0.3,Iris-setosa
5.0,3.4,1.5,0.2,Iris-setosa
4.4,2.9,1.4,0.2,Iris-setosa
4.9,3.1,1.5,0.1,Iris-setosa
5.4,3.7,1.5,0.2,Iris-setosa
5.2,2.7,3.9,1.4,Iris-versicolor
5.0,2.0,3.5,1.0,Iris-versicolor
5.9,3.0,4.2,1.5,Iris-versicolor
6.0,2.2,4.0,1.0,Iris-versicolor
6.1,2.9,4.7,1.4,Iris-versicolor
5.6,2.9,3.6,1.3,Iris-versicolor
6.7,3.1,4.4,1.4,Iris-versicolor
5.6,3.0,4.5,1.5,Iris-versicolor
5.8,2.7,4.1,1.0,Iris-versicolor
6.2,2.2,4.5,1.5,Iris-versicolor
5.6,2.8,4.9,2.0,Iris-virginica
7.7,2.8,6.7,2.0,Iris-virginica
6.3,2.7,4.9,1.8,Iris-virginica
6.7,3.3,5.7,2.1,Iris-virginica
7.2,3.2,6.0,1.8,Iris-virginica
6.2,2.8,4.8,1.8,Iris-virginica
6.1,3.0,4.9,1.8,Iris-virginica
6.4,2.8,5.6,2.1,Iris-virginica"""
from sklearn.cross_validation import train_test_split
if __name__ == '__main__':
    fileName = 'iris.data'
    lines = dataStr.split('\n')
    with open(fileName, 'r') as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.replace('\n', '').split(','), lines))
    outputList = []
    inputList = []
    for line in lines[:-1]:
        label = line[-1]
        line = list(map(lambda x: float(x)/5, line[:-1]))
        if label=='Iris-virginica':
            outputList.append([1., 0., 0.])
        elif label=='Iris-setosa':
            outputList.append([0., 1., 0.])
        else:
            outputList.append([0., 0., 1.])
        inputList.append(line)
        
    inputList, testInput, outputList, testOutput= train_test_split(inputList, outputList, test_size=0.2)
    inputList = np.array(inputList)
    outputList = np.array(outputList)
    model = BPANN(stepNum = 4000, learningRate=0.1, hiddenLayerStruct=[15])#初始化
    model.fit(inputList, outputList)#训练
    myX = inputList[5]
    #这里直接使用训练样本来测试模型，实际上是违规的
    predList = model.predict(testInput)#预测
#     for i in range(len(predList)):
#         print('myX对应的输出是', predList[i], '实际类别是', outputList[i])
    model.showConfusionMaxtrix(predList, testOutput)
    
    
    
    

