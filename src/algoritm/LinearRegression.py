'''
Created on 2018年11月20日

@author: pyli
'''
#一个线性回归工具，支持一元线性回归和多元线性回归，损失函数用最小二乘，优化算法用梯度下降
import random
import numpy as np

class LinearRegressionModel():
    
    def __init__(self,learningRate=.003, stepNum=30000, batchSize=100):
        self.pars = None#回归模型的参数，也就是各个变量的系数。
        self.parNum = 0
        #这里为了方便，截距被当作一个取值固定的变量来处理，系数是1.模型输入后，会初始化这个向量
        self.diffFuctions = []#存储每个变量对应方向的偏导数
        self.learningRate = learningRate
        self.stepNum = stepNum#每一批数据学习的步数
        self.batchSize = batchSize#我们把数据分成小块喂给模型学习，通常来说，这样做比让模型一下子学习所有的数据效果要好一点。
        
    def fit(self, trainInput, trainOutput):
#         if 
        trainInput = np.insert(trainInput, len(trainInput[0,:]), 1, axis=1)
        self.pars = [random.uniform(0,1) for i in range(len(trainInput[0, :]))]#初始化模型参数，这里使用随机数。random.uniform(0,1)
        self.pars = np.array(self.pars)
        self.parNum = len(self.pars)
#         print(trainInput)
        for j in range(self.stepNum):#当前批次的数据需要学习多次
            for i in range(0, len(trainInput), self.batchSize):
                trainInputBatch = trainInput[i: i + self.batchSize, :]#取出这一批数据
    #             print(trainInputBatch)
                trainOutputBatch = trainOutput[i: i + self.batchSize]
                delta = []
                for n in range(self.parNum):#更新每一个变量的系数
                    diffOnThisDim = [trainInputBatch[m,n]*(self.predict(trainInputBatch[m,:]) - trainOutputBatch[m])
                                     for m in range(len(trainInputBatch))]#当前变量对应的导数，在每个观测值的情况下的取值
#                     print("每个样本对应的偏导数是", (self.predict(trainInputBatch[0,:]) - trainOutputBatch[0]))
                    diffOnThisDim = np.sum(diffOnThisDim)/(1.0*len(trainInputBatch))#梯度在这个维度上分量的大小
                    delta.append(-self.learningRate*diffOnThisDim)
                print("修正量是", delta)
                self.pars = [self.pars[m] + delta[m] 
                                     for m in range(len(self.pars))] #更新这个维度上的参数，也就是第n个变量的系数
                print(self.pars)
#                 break
                    
                
    #计算一个观测值的输出
    def predict(self, inputData):
        res = np.sum(self.pars*inputData)
        print("计算", inputData, res, self.pars)

        return res

from sklearn.cross_validation import train_test_split
if __name__ == '__main__':
    inputList = [[i] for i in range(1, 10)]
    outputList = [i for i in range(21, 30)]
    inputList, _, outputList, _ = train_test_split(inputList, outputList, test_size=0.0)
    print(inputList)
    print(outputList)
    inputList = np.array(inputList)

    model = LinearRegressionModel()
    model.fit(inputList, outputList)
    



