'''
Created on 2018年11月20日

@author: pyli
'''
#softmax回归
import random
import numpy as np
import copy

class Softmax():
    
    def __init__(self,learningRate=.01, stepNum=10, batchSize=1):
        self.pars = None#参数矩阵，每一行是一个类别对应的自变量系数
        self.parNum = 0#模型里自变量的个数，后面需要初始化
        #这里为了方便，截距被当作一个取值固定的变量来处理，系数是1.模型输入后，会初始化这个向量
        self.diffFuctions = []#存储每个变量对应方向的偏导数
        self.learningRate = learningRate#学习率。这里每个参数的学习率是一样的；我们也可以为各个参数设置不同的学习率。
        self.stepNum = stepNum#每一批数据学习的步数
        
        
    def fit(self, trainInput, trainOutput):
        self.x = copy.deepcopy(trainInput)
        self.N = len(trainOutput)
        self.y = trainOutput
        self.classNum = len(trainOutput[0])
        def predict4Train(inputData):#训练里使用的一个predict函数，
            # 针对训练数据已经为截距增加了变量的情况
            probList = np.dot(self.pars, np.transpose(inputData))
            probList = list(probList[:, 0])#矩阵的第一行是概率分布列表
            probList = self.softmax(probList)
            probList = list(probList)
            return probList
        trainOutput = np.array(trainOutput)
        self.parNum = len(trainInput[0,:])#初始化模型里自变量的个数
        trainInput = np.insert(trainInput, self.parNum, 1, axis=1)#为截距增加一列取值为1的变量
        self.parNum += 1
        self.pars = [[0*random.uniform(-0.2, 0.2) for i in range(self.parNum)]
                      for j in range(self.classNum)]#初始化模型参数矩阵(self.classNum行self.parNum列)，这里使用0。
        self.pars = np.array(self.pars)#处理成numpy的数组，便于进行乘法等运算
        self.k = self.parNum
        for _ in range(self.stepNum):#数据需要学习多次
            for i in range(0, len(trainInput)):#遍历样本
                thisInput = trainInput[i, :]
                thisInput = np.array([thisInput])
                thisOutPut = trainOutput[i, :]
                delta = np.zeros((self.classNum, self.parNum))#用来存储基于当前参数和这个样本计算出来的参数修正量
                costValue = 0
                predProbList = predict4Train(thisInput)
                for m in range(self.classNum):#遍历每一个softmax函数
                    realProbThisClass = thisOutPut[m]
                    for n in range(self.parNum):#遍历这个softmax函数的每一个参数
                        predProbThisClass = predProbList[m]
                        if predProbThisClass>0:
                            costValue += realProbThisClass*np.log10(predProbThisClass)
                        gradOnThisDim = thisInput[0,n] * (predProbThisClass - realProbThisClass)
                        #python3X里，除以int时，如果不能整除，就会得到一个float;python2X里则会得到一个想下取整的结果，要注意。
                        delta[m, n] = -self.learningRate*gradOnThisDim
                self.pars += delta #更新参数
                print("损失值为", costValue)
                # print(self.pars)
                              
    #计算一个观测值的输出
    def predict(self, inputData):
        flag = False#是单个样本
        try: 
            inputData[0][0]
            flag = True
            print("是多个样本", flag, type(inputData[0][0]),type(inputData[0][0])==int, type(inputData[0][0])==float)
        except:
            pass
        if flag==False:
            inputData = np.array(inputData.tolist()+ [1])#为截距增加一列取值为1的变量
            probList = np.dot(self.pars, np.transpose(inputData))
            probList = list(probList)#从矩阵的第一行才是概率分布列表
            probList = self.softmax(probList)
            maxProb = np.max(probList)
            probList = list(probList)
            maxProbIndex = probList.index(maxProb)
            #用来做预测的时候，需要将概率值二值化，也就是输出类别标签
            predLabel = [1 if i==maxProbIndex else 0 for i in range(len(probList))]
            return predLabel
        else:
            res = []
            for line in inputData:
                line = np.array(line.tolist()+ [1])#为截距增加一列取值为1的变量
                probList = np.dot(self.pars, np.transpose(line))
                probList = list(probList)#从矩阵的第一行才是概率分布列表
                probList = self.softmax(probList)
                maxProb = np.max(probList)
                probList = list(probList)
                maxProbIndex = probList.index(maxProb)
                predLabel = [1 if i==maxProbIndex else 0 for i in range(len(probList))]
                res.append(predLabel)
        return res

    def softmax(self, xList):
        xArray = np.array(xList)
        xArray = np.exp(xArray)
        sumV = sum(xArray)
        if sumV==0:#如果各家概率都是零
            result = xArray*0
        else:
            result = xArray/sumV
        return result

    #统计并打印混淆矩阵
    def showConfusionMaxtrix(self, predOutput, realOutput):
        confusionMatrix = np.zeros((self.classNum, self.classNum))
        for i in range(len(predOutput)):
            classIndex = realOutput[i].index(1)
            confusionMatrix[classIndex, :] += predOutput[i]
        print("混淆矩阵:")
        print("列表示预测类别;行表示真实类别")
        print(confusionMatrix)

class Softmax4CNN():#需要为cnn的输出做一些改动，比如需要将cnn传过来的抽象图像拉直，拼接成一个向量；
    # 特征的个数就是这个向量的长度。初始化softmax分类器参数的时候，需要接收来自前面的结果。

    def __init__(self, numOfNode, classNum, learningRate=.01, stepNum=10):
        self.weights = None  # 参数矩阵，每一行是一个类别对应的自变量系数
        self.parNum = numOfNode  # 模型里自变量的个数，后面需要初始化
        self.kernelNum = numOfNode#卷积层在训练的时候，需要获得后面一层的卷积核个数，这里假装softmax的节点就是卷积核
        # 这里为了方便，截距被当作一个取值固定的变量来处理，系数是1.模型输入后，会初始化这个向量
        self.diffFuctions = []  # 存储每个变量对应方向的偏导数
        self.learningRate = learningRate  # 学习率。这里每个参数的学习率是一样的；我们也可以为各个参数设置不同的学习率。
        self.stepNum = stepNum  # 每一批数据学习的步数
        self.classNum = classNum
        self.init()

    def init(self):
        self.weights = [[random.uniform(-0.2, 0.2) for i in range(self.parNum)]
                     for j in range(self.classNum)]  # 初始化模型参数矩阵(self.classNum行self.parNum列)，这里使用0。
        self.weights = np.array(self.weights)  # 处理成numpy的数组，便于进行乘法等运算
        self.grad = np.array(self.weights)#梯度矩阵和权重矩阵shape相同
        self.bias = [random.uniform(-0.2, 0.2) for j in range(self.classNum)]
        self.bias = np.array(self.bias)

    def calGrad(self, outputVector):
        #print("softmax的输入是", self.trainingInput)
        inputVector = np.array(self.trainingInput).reshape((self.parNum))
        predOutputVector =self.traningOutput
        for classNO in range(self.classNum):#遍历每一个类别的位置
            for i in range(self.parNum):#遍历与当前节点相连的所有特征
                #print(inputVector[i])
                self.grad[classNO, i] = inputVector[i] * \
                                        (predOutputVector[classNO] - outputVector[classNO])
                                        
        self.error2FormerLayer = np.zeros(self.trainingInput.shape)#softmax反向传播到前一层的误差
        indexList = np.array(range(len(inputVector)))
        indexCube = indexList.reshape(self.trainingInput.shape)#存储原始数据每一个点，在拉直后形成的一维向量中的位置
        kernelNumOfFormerLayer, picNumOfFormerLayerKernel, height, width = self.trainingInput.shape
        for i in range(kernelNumOfFormerLayer):
            for j in range(picNumOfFormerLayerKernel):
                for m in range(height):
                    for n in range(width):
                        softmaxFeatureNO = indexCube[i,j,m,n]#当前像素点对应的softmax特征序号
                        #把与这个像素连接的所有来自后一层的误差加权球和
                        self.error2FormerLayer[i,j,m,n] = np.sum(self.weights[:, softmaxFeatureNO] * (self.traningOutput-outputVector))          
        return self.grad
    
    def updateWeights(self):
        self.weights -= self.grad * self.learningRate
        
    def updateWeights4Multi(self, grad):
#         print("softmax更新参数", self)
        self.weights -= grad * self.learningRate  
        
    # 计算一个观测值的输出
    def predict(self, inputImageList):
        inputData = np.array(inputImageList).reshape((1, self.parNum))
        probList = np.dot(self.weights, np.transpose(inputData))
        probList = np.transpose(probList)
        probList += self.bias
        probList = probList[0]
        probList = list(probList)  # 从矩阵的第一行才是概率分布列表
        probList = self.softmax(probList)
        maxProb = np.max(probList)
        probList = list(probList)
        maxProbIndex = probList.index(maxProb)
        # 用来做预测的时候，需要将概率值二值化，也就是输出类别标签
        predLabel = [1 if i == maxProbIndex else 0 for i in range(len(probList))]
        return predLabel
    
    # 计算一个观测值的输出
    def predict4Train(self, inputImageList):
        self.trainingInput = inputImageList#训练过程中需要用的变量，训练完成后，需要清空
        inputData = np.array(inputImageList).reshape((1, self.parNum))
        probList = np.dot(self.weights, np.transpose(inputData))
        probList = np.transpose(probList) + self.bias
        probList = list(probList[0])  # 从矩阵的第一行才是概率分布列表
        probList = self.softmax(probList)
        self.traningOutput = probList
        return probList

    def softmax(self, xList):
        xArray = np.array(xList)
        # print(xArray)
        xArray = np.exp(xArray)
        sumV = sum(xArray)
        if sumV == 0:  # 如果各家概率都是零
            result = xArray * 0
        else:
            result = xArray / sumV
        return result

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
from sklearn.model_selection import train_test_split
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
        line = list(map(lambda x: float(x), line[:-1]))
        if label=='Iris-virginica':
            outputList.append([1., 0., 0.])
        elif label=='Iris-setosa':
            outputList.append([0., 1., 0.])
        else:
            outputList.append([0., 0., 1.])
        inputList.append(line)
        
    inputList, _, outputList, _ = train_test_split(inputList, outputList, test_size=0.0)
    inputList = np.array(inputList)
    model = Softmax(stepNum = 100, learningRate=0.002)#初始化
    model.fit(inputList, outputList)#训练
    myX = inputList[5]
    #这里直接使用训练样本来测试模型，实际上是违规的
    predList = model.predict(inputList)#预测
    for i in range(len(predList)):
        print('myX对应的输出是', predList[i], '实际类别是', outputList[i])
    model.showConfusionMaxtrix(predList, outputList)
    
    
    
    

