'''
Created on 2018年11月20日

@author: pyli
'''
#逻辑回归
import random
import numpy as np
import copy

class LogisticRegressionModel():
    
    def __init__(self,learningRate=.01, stepNum=10, batchSize=1):
        self.pars = None#回归模型的参数，也就是各个变量的系数。
        self.parNum = 0#模型里自变量的个数，后面需要初始化
        #这里为了方便，截距被当作一个取值固定的变量来处理，系数是1.模型输入后，会初始化这个向量
        self.diffFuctions = []#存储每个变量对应方向的偏导数
        self.learningRate = learningRate#学习率。这里每个参数的学习率是一样的；我们也可以为各个参数设置不同的学习率。
        self.stepNum = stepNum#每一批数据学习的步数
        self.batchSize = batchSize#我们把数据分成小块喂给模型学习，通常来说，这样做比让模型一下子学习所有的数据效果要好一点。
        
        
    def fit(self, trainInput, trainOutput):
        self.x = copy.deepcopy(trainInput)
        self.N = len(trainOutput)
        self.y = trainOutput
        def predict4Train(inputData):#训练里使用的一个predict函数，
            # 针对训练数据已经为截距增加了变量的情况
            res = np.sum(self.pars * inputData)
            res = 1/(1 + np.exp(-res))
            return res
        trainInput = np.insert(trainInput, len(trainInput[0,:]), 1, axis=1)#为截距增加一列取值为1的变量
        self.pars = [random.uniform(0,1) for i in range(len(trainInput[0, :]))]#初始化模型参数，这里使用随机数。
        self.pars = np.array(self.pars)#处理成numpy的数组，便于进行乘法等运算
        self.parNum = len(self.pars)#初始化模型里自变量的个数
        self.k = self.parNum
        for j in range(self.stepNum):#当前批次的数据需要学习多次
            for i in range(0, len(trainInput), self.batchSize):#分批来遍历样本
                trainInputBatch = trainInput[i: i + self.batchSize, :]#取出这一批数据的输入
                trainOutputBatch = trainOutput[i: i + self.batchSize]#去除这一批数据的输出
                delta = []#用来存储基于当前参数和这批数据计算出来的参数修正量
                for n in range(self.parNum):#更新每一个变量的系数
                    diffOnThisDim = [trainInputBatch[m,n]*(predict4Train(trainInputBatch[m,:]) - trainOutputBatch[m])
                                     for m in range(len(trainInputBatch))]#当前变量对应的导数，在每个观测值的情况下的取值
                    diffOnThisDim = np.sum(diffOnThisDim)/(len(trainInputBatch))#梯度在这个维度上分量的大小。
                    diffOnThisDim = diffOnThisDim/np.sqrt(np.abs(diffOnThisDim))

                    #python3X里，除以int时，如果不能整除，就会得到一个float;python2X里则会得到一个想下取整的结果，要注意。
                    delta.append(-self.learningRate*diffOnThisDim)
#                 print("修正量是", delta)
                self.pars = [self.pars[m] + delta[m] 
                                     for m in range(len(self.pars))] #更新这个维度上的参数，也就是第n个变量的系数
                # print(self.pars)
                              
    #计算一个观测值的输出
    def predict(self, inputData):
        flag = False#是单个样本
        try: 
            inputData[0][0]#如果报错，说明是一个一维数组，输入数据是一个样本；反之是多个样本
            flag = True
            print("是多个样本", flag, type(inputData[0][0]),type(inputData[0][0])==int, type(inputData[0][0])==float)
        except:
            pass
        if flag==False:
            inputData = np.array(inputData.tolist()+ [1])#为截距增加一列取值为1的变量
            res = np.sum(self.pars*inputData)
            res = 1/(1 + np.exp(-res))
            if res>0.5:#如果概率值大于0.5,我们认为是正例，类别标签赋值为1。
                #阈值不局限于0.5,可以根据需求调高或者降低。
                res = 1
            else:#如果概率值小于0.5,我们认为是负例，类别标签赋值为0
                res = 0
        else:
            res = []
            for line in inputData:
                line = np.array(line.tolist()+ [1])#为截距增加一列取值为1的变量
                tempRes = np.sum(self.pars*line)
                tempRes = 1/(1 + np.exp(-tempRes))
                if tempRes > 0.5:
                    tempRes = 1
                else:
                    tempRes = 0
                res.append(tempRes)
        return res
    
    #评估模型
    #统计分类器的分类准确率
    def calAccuracy(self, predOutput, realOutput):
        rightDecisionNum = 0#正确分类的样本数
        for i in range(len(predOutput)):
            if predOutput[i] == realOutput[i]:#如果预测的类别和真实类别相同
                rightDecisionNum += 1.
        print("分类的准确率是", rightDecisionNum/len(predOutput))

    #分类器对正例的召回率
    def calRecall(self, predOutput, realOutput):
        foundRealPositiveNum = 0#找到的正例个数
        for _ in range(len(predOutput)):
            if predOutput[i] == realOutput[i]==1:#如果预测类别和真实类别都是正例
                foundRealPositiveNum += 1.
        recall = foundRealPositiveNum / len(predOutput)
        print("对正例的召回率是", recall)
        return recall

    #分类器对正例的分类精度
    def calPrecision(self, predOutput, realOutput):
        foundRealPositiveNum = 0
        realPositiveNum = 0#真实类别为正例的样本个数
        for i in range(len(predOutput)):
            if realOutput[i]==1:
                realPositiveNum += 1.
                if predOutput[i] == realOutput[i]:
                    foundRealPositiveNum += 1.
        precision = foundRealPositiveNum /realPositiveNum
        print("对正例的分类精度是", precision)
        return precision

    #分类器对正例的F1-score
    def calF1Score(self, predOutput, realOutput):
        recall = self.calRecall(predOutput, realOutput)
        precision = self.calPrecision(predOutput, realOutput)
        if recall+precision==0:#如果两个指标都为0
            f1score = 0
        else:
            f1score = 2*recall*precision/(recall + precision)#f1-score的计算公式
        print("对正例的f1-score是", f1score)

    #统计分类器的混淆矩阵
    def calConfusionMatrix(self, predOutput, realOutput):
        confusionMatrix = np.zeros((2, 2))
        for i in range(len(predOutput)):
            print(predOutput[i], realOutput[i])
            confusionMatrix[predOutput[i], realOutput[i]] += 1#分类器预测的类别为predOutput[i]
            #真实类别为realOutput[i]的样本个数加一
        print("混淆矩阵:")
        print("列表示预测类别;行表示真实类别")
        print(confusionMatrix)


dataStr = """5.1	3.5	1.4	0.2	Iris-setosa
4.9	3	1.4	0.2	Iris-setosa
4.7	3.2	1.3	0.2	Iris-setosa
4.6	3.1	1.5	0.2	Iris-setosa
7.7	3	6.1	2.3	Iris-virginica
6.3	3.4	5.6	2.4	Iris-virginica
6.4	3.1	5.5	1.8	Iris-virginica
6	3	4.8	1.8	Iris-virginica
6.9	3.1	5.4	2.1	Iris-virginica
6.7	3.1	5.6	2.4	Iris-virginica"""

from sklearn.cross_validation import train_test_split
if __name__ == '__main__':
    lines = dataStr.split('\n')
    # lines = open('iris.data', 'r').readlines()
    lines = list(map(lambda x: x.replace('\n', '').split('\t'), lines))
    print(lines)
    outputList = []
    inputList = []
    for line in lines[:-1]:
        label = line[-1]
        line = list(map(lambda x: float(x), line[:-1]))
        if label=='Iris-virginica':
            outputList.append(1)
        else:
            outputList.append(0)
        inputList.append(line)
        
    inputList, testInputList, outputList, testOutputList = \
        train_test_split(inputList, outputList, test_size=0.2)
    inputList, testInputList = np.array(inputList),  np.array(testInputList)
    model = LogisticRegressionModel(stepNum = 20)#初始化
    model.fit(inputList, outputList)#训练
    myX = inputList[5]
    predList = model.predict(testInputList)#预测
    for i in range(len(predList)):
        print('myX对应的输出是', predList[i], '实际类别是', outputList[i])
    model.calF1Score(predList, testOutputList)
    model.calConfusionMatrix(predList, testOutputList)

