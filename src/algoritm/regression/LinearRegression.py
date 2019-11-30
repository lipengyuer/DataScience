'''
Created on 2018年11月20日

@author: pyli
'''
#一个线性回归工具，支持一元线性回归和多元线性回归，损失函数用最小二乘，优化算法用梯度下降
import random
import numpy as np
from scipy.stats import chi2
from scipy.stats import t
from scipy.stats import f as ff
from scipy.stats import norm
import copy

class LinearRegressionModel():
    
    def __init__(self,learningRate=.0001, stepNum=10, batchSize=1, if_print=True):
        self.pars = None#回归模型的参数，也就是各个变量的系数。
        self.parNum = 0#模型里自变量的个数，后面需要初始化
        #这里为了方便，截距被当作一个取值固定的变量来处理，系数是1.模型输入后，会初始化这个向量
        self.diffFuctions = []#存储每个变量对应方向的偏导数
        self.learningRate = learningRate#学习率。这里每个参数的学习率是一样的；我们也可以为各个参数设置不同的学习率。
        self.stepNum = stepNum#每一批数据学习的步数
        self.batchSize = batchSize#我们把数据分成小块喂给模型学习，通常来说，这样做比让模型一下子学习所有的数据效果要好一点。
        
        #用于统计检验的变量
        self.ESS = 0
        self.RSS = 0
        self.N = 0
        self.k =  0
        self.x = None
        self.y = None#存储因变量的真实值
        self.y_hat = None#存储因变量的平均值
        self.y_bar = None#使用回归模型计算的到的因变量估计值
        
        self.if_print = if_print#是否打印计算过程和检验结果。某些场景下不需要打印，比如stepwise回归
        
    def fit(self, trainInput, trainOutput):
        self.x = copy.deepcopy(trainInput)
        self.N = len(trainOutput)
        self.y = trainOutput
        def predict4Train(inputData):#训练里使用的一个predict函数，
            # 针对训练数据已经为截距增加了变量的情况
            res = np.sum(self.pars * inputData)
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
                print(self.pars)
                              
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
            res = np.sum(self.pars*inputData)
        else:
            res = []
#             print(inputData)
            for line in inputData:
#                 print(line)
                line = np.array(line.tolist()+ [1])#为截距增加一列取值为1的变量
                res.append(np.sum(self.pars*line))
        return res
    
    #评估模型
    def evaluateModel(self):
        #对模型参数进行t检验
        #对模型进行f检验
        #如果是多元线性回归，用方差膨胀因子检查自变量是否存在多重共线性
        #计算拟合优度和调整拟合优度
        pass

    #对训练得到的模型进行F检验，打印输出检验结果
    def Ftest(self, alpha=0.05):
        """
        f统计量计算公式:f = [ESS/(k-1)]/[RSS/(N-k)]
        ESS(exlplained sum of suqres),可解释平方和，回归平方和
        RSS(residual sum of squares),残差平方和
        k, 回归系数的个数(实际上就是自变量的个数+1);N,训练样本的个数
        """
        self.y_hat = np.mean(self.y)
        print("输入是", self.x)
        self.y_bar = self.predict(self.x)
        ESS = np.sum([(self.y_bar[i]-self.y_hat)**2 for i in range(self.N)])
        RSS = np.sum([(self.y_bar[i]-self.y[i])**2 for i in range(self.N)])
        print("f分布的自由度是",  self.k-1,  self.N-self.k)
        fValue = (ESS/(self.k-1))/(RSS/(self.N-self.k))#要求是多元线性回归模型,样本数量大于参数个数
        f_alpha = ff.isf(alpha, self.k-1, self.N-self.k)
        print(fValue, f_alpha)
        if fValue>f_alpha:
            print("全部参数全为0的情况下，出现f统计量取值为", fValue, '的概率小于等于', alpha, "说明全部参数不为0")
            print("模型通过了f检验")
            return True
        else:
            return False
            
    #对模型的各个参数进行T检验，并打印输出检验结果
    def Ttest(self, alpha = 0.05):
        self.y_hat = np.mean(self.y)
        self.y_bar = self.predict(self.x)
        ESS = np.sum([(self.y_bar[i]-self.y_hat)**2 for i in range(self.N)])
        for i in range(self.k - 1):#对所有自变量的系数进行t检验
            x_i_list = list(map(lambda x: x[i], self.x))#提取出所有样本的第i个特征的取值
            var_par_i = ESS/np.var(x_i_list)#模型的第i个系数的方差
            tValue = ESS/np.sqrt(var_par_i)
            t_alpha = np.abs(t.ppf(alpha, self.k-1))#求要求显著水平下的t值
#             print(t_alpha, tValue)
            print("对参数", i, "的t检验结果")
            if tValue>t_alpha:
                print("这个参数的t统计量取值落在显著性水平对应的区间外，出现的概率小于", alpha)
                print("这说明这个参数等于0的概率很小，是显著的")
            else:
                print("这个参数是不显著的")
            print("^^^^^^^^^^^^^^")
            

    #计算模型的调整判定系数
    def goodnessOfFit(self):
        self.y_hat = np.mean(self.y)#输出的均值
        print("输出的均值是", self.y_hat)
        self.y_bar = self.predict(self.x)#因变量的预测值
        from matplotlib import pyplot as plt
        plt.plot(self.y, self.y_bar, '.')
        plt.show()
        TSS = np.sum([(self.y[i] - self.y_hat)**2 for i in range(self.N)])#因变量的总平方和
        ESS = np.sum([(self.y_bar[i]-self.y_hat)**2 for i in range(self.N)])#回归平方和
        RSS = np.sum([(self.y_bar[i]-self.y[i])**2 for i in range(self.N)])#残差平方和
#         TSS = ESS + RSS
        ESS = TSS - RSS
        print(ESS, RSS, TSS)
        r2 = (ESS/(self.N-self.k-1))/(TSS/(self.N-1))
        print("模型的调整判定系数是", r2)

    def VIFTest(self):
        pass

from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    inputList = [[i, i**0.5] for i in range(1, 10)]
    outputList = [i for i in range(21, 30)]#y = x+20

#     fileName = '/home/pyli/tasks/小任务/data.txt'
#     with open(fileName, 'r') as f:
#         lines = f.readlines()
#         lines = list(map(lambda x: x.replace('\n', '').split('\t'), lines))
#     outputList = []
#     inputList = []
#     for line in lines[:-1]:
# #         print(line)
#         line = list(map(lambda x: float(x), line))
#         outputList.append(line[-1])
#         inputList.append(line[:-1])
        
    inputList, _, outputList, _ = train_test_split(inputList, outputList, test_size=0.0)
    inputList = np.array(inputList)
    model = LinearRegressionModel()#初始化
    model.fit(inputList, outputList)#训练
    myX = inputList[5]
    res = model.predict(np.array(myX))#预测
    print('myX对应的输出是', res)
    model.Ftest()
    model.Ttest()
    





