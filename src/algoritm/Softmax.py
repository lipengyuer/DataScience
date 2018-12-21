'''
Created on 2018年11月20日

@author: pyli
'''
#逻辑回归
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
#             print(self.pars)
#             print(inputData)
            probList = np.dot(self.pars, np.transpose(inputData))
            probList = list(probList[:, 0])#从矩阵的第一行才是概率分布列表
#             print(probList)

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
        for _ in range(self.stepNum):#当前批次的数据需要学习多次
            for i in range(0, len(trainInput)):#遍历样本
                thisInput = trainInput[i, :]
                thisInput = np.array([thisInput])
                thisOutPut = trainOutput[i, :]
                ####################开始计算各个参数的梯度
                delta = np.zeros((self.classNum, self.parNum))#用来存储基于当前参数和这批数据计算出来的参数修正量
                costValue = 0
                predProbList = predict4Train(thisInput)
#                 print(thisOutPut, predProbList)

                for m in range(self.classNum):#遍历每一个softmax函数
#                     print(thisInput)
                    realProbThisClass = thisOutPut[m]
                    for n in range(self.parNum):#遍历这个softmax函数的每一个参数
                        predProbThisClass = predProbList[m]
                        if predProbThisClass>0:
                            costValue += realProbThisClass*np.log10(predProbThisClass)
                        gradOnThisDim = thisInput[0,n] * (predProbThisClass - realProbThisClass)
#                         print(gradOnThisDim)
                        #python3X里，除以int时，如果不能整除，就会得到一个float;python2X里则会得到一个想下取整的结果，要注意。
                        delta[m, n] = -self.learningRate*gradOnThisDim
#                 print("修正量是", delta)
                self.pars += delta #更新参数
                print("损失值为", costValue)
#                 print(self.pars)
                              
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
            predLabel = [1 if i==maxProbIndex else 0 for i in range(len(probList))]
            return predLabel
        else:
            res = []
#             print(inputData)
            for line in inputData:
#                 print(line)
                line = np.array(line.tolist()+ [1])#为截距增加一列取值为1的变量
                probList = np.dot(self.pars, np.transpose(line))
#                 print(probList)
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
        sumV = sum(xArray)
        if sumV==0:
            result = xArray*0
        else:
            result = xArray/sumV
        return result
        
    #评估模型
    def evaluateModel(self, testInput, testOutput):
        predOutput = self.predict(testInput)
    
    def showConfusionMaxtrix(self, ):
        pass


from sklearn.cross_validation import train_test_split
if __name__ == '__main__':
    fileName = 'irisData.txt'
    with open(fileName, 'r') as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.replace('\n', '').split(','), lines))
    outputList = []
    inputList = []
    for line in lines[:-1]:
#         print(line)
        label = line[-1]
        line = list(map(lambda x: float(x)/10, line[:-1]))
        if label=='Iris-virginica':
            outputList.append([1., 0., 0.])
        elif label=='Iris-setosa':
            outputList.append([0., 1., 0.])
        else:
            outputList.append([0., 0., 1.])
        inputList.append(line)
        
    inputList, _, outputList, _ = train_test_split(inputList, outputList, test_size=0.0)
    inputList = np.array(inputList)
    model = Softmax(stepNum = 100, learningRate=0.001)#初始化
    model.fit(inputList, outputList)#训练
    myX = inputList[5]
    predList = model.predict(inputList)#预测
    for i in range(len(predList)):
        print('myX对应的输出是', predList[i], '实际类别是', outputList[i])

