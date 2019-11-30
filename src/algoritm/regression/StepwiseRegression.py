'''
Created on 2019年11月17日

@author: Administrator
'''
'''
Created on 2018年11月20日

@author: pyli
'''
import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)

import random
import numpy as np
from scipy.stats import chi2
from scipy.stats import t
from scipy.stats import f as ff
from scipy.stats import norm
import copy
from LinearRegression import LinearRegressionModel

class StepwiseRegression():
    
    def __init__(self,learningRate=.0001, stepNum=10, batchSize=1):
        self.feature_id_list = []
        self.all_feature_num = None
        self.best_model = None#最模型
        self.model_p = 0.05
        
    def fit(self, trainInput, trainOutput): 
        self.all_feature_num = len(trainInput[0])#所有候选特征的个数
        left_feature_id_ist = list(range(self.all_feature_num))#剩余的特征列表
        
        print("最初的候选特征是", left_feature_id_ist[::])
        end = False#是否停止计算
        while end!=True:
            end = True
            for feature_id in left_feature_id_ist[::]:
                if self.if_fine_feature(feature_id, trainInput, trainOutput):
                    left_feature_id_ist.remove(feature_id)
                    self.feature_id_list.append(feature_id)
                    end = False
                    break
        print("有用特征的id是", self.feature_id_list)
        
        #用最好的特征拟合数据
        self.best_model = LinearRegressionModel(if_print=False,stepNum=2000)
        trainInput = trainInput[:, self.feature_id_list]
        self.best_model.fit(trainInput, trainOutput)
    
    def predict(self, inputData):
        inputData = np.array(inputData)
        inputData = inputData[:, self.feature_id_list]
        return self.best_model.predict(inputData)
        
                    
    def if_fine_feature(self, feature_id, trainInput, trainOutput):
        temp_feature_ids = self.feature_id_list + [feature_id]
        trainInput = trainInput[:, temp_feature_ids]
        model = LinearRegressionModel(if_print=False,stepNum=2000)
        model.fit(trainInput, trainOutput)
        if_sig = model.Ftest()
        return if_sig            
    
    #评估模型
    def evaluateModel(self):
        #对模型参数进行t检验
        #对模型进行f检验
        #如果是多元线性回归，用方差膨胀因子检查自变量是否存在多重共线性
        #计算拟合优度和调整拟合优度
        pass

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    inputList = [[i, i**0.5] for i in range(1, 100)]
    outputList = [i for i in range(21, 120)]#y = x+20
      
    inputList, _, outputList, _ = train_test_split(inputList, outputList, test_size=0.0)
    inputList = np.array(inputList)
    model = StepwiseRegression()#初始化
    model.fit(inputList, outputList)#训练
    res = model.predict(inputList)
    plt.plot(outputList)
    plt.plot(res)
    plt.show()

    





