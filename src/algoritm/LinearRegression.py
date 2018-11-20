'''
Created on 2018年11月20日

@author: pyli
'''
#一个线性回归工具，支持一元线性回归和多元线性回归，损失函数用最小二乘，优化算法用梯度下降
import random
import numpy as np

class LinearRegressionModel():
    
    def __init__(self,learningRate=.03, stepNum=1000, batchSize=100):
        self.pars = None#回归模型的参数，也就是各个变量的系数。
        self.parNum = 0#模型里自变量的个数，后面需要初始化
        #这里为了方便，截距被当作一个取值固定的变量来处理，系数是1.模型输入后，会初始化这个向量
        self.diffFuctions = []#存储每个变量对应方向的偏导数
        self.learningRate = learningRate#学习率。这里每个参数的学习率是一样的；我们也可以为各个参数设置不同的学习率。
        self.stepNum = stepNum#每一批数据学习的步数
        self.batchSize = batchSize#我们把数据分成小块喂给模型学习，通常来说，这样做比让模型一下子学习所有的数据效果要好一点。
        
    def fit(self, trainInput, trainOutput):
        def predict4Train(inputData):#训练里使用的一个predict函数，
            # 针对训练数据已经为截距增加了变量的情况
            res = np.sum(self.pars * inputData)
            return res
        trainInput = np.insert(trainInput, len(trainInput[0,:]), 1, axis=1)#为截距增加一列取值为1的变量
        self.pars = [random.uniform(0,1) for i in range(len(trainInput[0, :]))]#初始化模型参数，这里使用随机数。
        self.pars = np.array(self.pars)#处理成numpy的数组，便于进行乘法等运算
        self.parNum = len(self.pars)#初始化模型里自变量的个数
        for j in range(self.stepNum):#当前批次的数据需要学习多次
            for i in range(0, len(trainInput), self.batchSize):#分批来遍历样本
                trainInputBatch = trainInput[i: i + self.batchSize, :]#取出这一批数据的输入
                trainOutputBatch = trainOutput[i: i + self.batchSize]#去除这一批数据的输出
                delta = []#用来存储基于当前参数和这批数据计算出来的参数修正量
                for n in range(self.parNum):#更新每一个变量的系数
                    diffOnThisDim = [trainInputBatch[m,n]*(predict4Train(trainInputBatch[m,:]) - trainOutputBatch[m])
                                     for m in range(len(trainInputBatch))]#当前变量对应的导数，在每个观测值的情况下的取值
                    diffOnThisDim = np.sum(diffOnThisDim)/(len(trainInputBatch))#梯度在这个维度上分量的大小。
                    #python3X里，除以int时，如果不能整除，就会得到一个float;python2X里则会得到一个想下取整的结果，要注意。
                    delta.append(-self.learningRate*diffOnThisDim)
                print("修正量是", delta)
                self.pars = [self.pars[m] + delta[m] 
                                     for m in range(len(self.pars))] #更新这个维度上的参数，也就是第n个变量的系数
                print(self.pars)
                              
    #计算一个观测值的输出
    def predict(self, inputData):
        inputData = np.array(inputData.tolist()+ [1])#为截距增加一列取值为1的变量
        res = np.sum(self.pars*inputData)
        return res
    
    #评估模型
    def evaluateModel(self):
        #对模型参数进行t检验
        #对模型进行f检验
        #如果是多元线性回归，用方差膨胀因子检查自变量是否存在多重共线性
        #计算拟合优度和调整拟合优度
        pass

from sklearn.cross_validation import train_test_split
if __name__ == '__main__':
    inputList = [[i] for i in range(1, 10)]
    outputList = [i for i in range(21, 30)]#y = x+20
    inputList, _, outputList, _ = train_test_split(inputList, outputList, test_size=0.0)
    inputList = np.array(inputList)
    model = LinearRegressionModel()#初始化
    model.fit(inputList, outputList)#训练
    myX = [22]
    res = model.predict(np.array(myX))#预测
    print('myX对应的输出是', res)
    



