#一个用来对简单情况下的样本，即取值空间有限的离散特征，进行分类的决策树。
#输入可以是字符串或者整数，保证是类别变量就可以。
import numpy as np
import copy

class ID3DecisionTree():

    def __init__(self):
        aTree = {}

    def fit(self, inputData, outputData):
        pass
    
    #基于特征的信息熵，从未使用的特征中挑选最好的分组特征
    def chooseBestFeatureWithEntropy(self, inputData, leftFeatresIndexList, outputData):
        totalNumOfSamples = len(inputData)#样本的总数，用来计算某个特征取值出现的概率
        ##############开始统计各个特征的取值在样本中出现的次数#############
        #这种统计在朴素贝叶斯等算法中是常用的，通常用来计算需要的概率
        valueSampleNumMap = {}#存储各个特征的各个取值出现的次数
        for line in inputData:#遍历剩下的每一个特征，计算各自对应的信息熵
            for i in leftFeatresIndexList:#遍历每个剩余特征的编号(就是索引值)
                featureValue = line[i]#当前特征的取值
                if i in valueSampleNumMap:#如果这个特征编号已经收录
                    valueSampleNumMap[i][featureValue] = valueSampleNumMap[i].get(featureValue, 0.) + 1.
                    """
                    valueSampleNumMap[i].get(featureValue, 0.)+ 1.
                    if featureValue in valueSampleNumMap[i]:
                        valueSampleNumMap[i][featureValue] += 1
                    else:
                        valueSampleNumMap[i][featureValue] = 1
                    """
                else:
                    valueSampleNumMap[i] = {}
                    valueSampleNumMap[i][featureValue] = 1.
        ##############完成统计各个特征的取值在样本中出现的次数#############
        #基于各个取值的出现次数，计算每一个特征的信息熵
        entropyMap = {}
        for featureNO in leftFeatresIndexList:
            valueFreqMap = valueSampleNumMap[featureNO]
            featureEntropy = 0.
            for featureValue in valueFreqMap:
                num = valueFreqMap[featureValue]
                featureEntropy -= (num/totalNumOfSamples)*np.log2(num/totalNumOfSamples)
                #信息熵公式entropy= - p_1*log(p_1) - p_2*log(p_2) - ... - p_k*log(p_k)
            entropyMap[featureNO] = featureEntropy
        print("各个特征的信息熵情况是", entropyMap)
        entropyList = sorted(entropyMap.items(), key=lambda x: x[1], reverse=True)#按照熵的大小倒序排列
        bestFeatureNO = entropyList[0][0]#取出熵最大的特这个编号并返回
        return bestFeatureNO
    
    #基于信息增益(information gain, IG)来挑选剩余特征中最好的分组特征
    def chooseBestFeatureWithIG(self, inputData, leftFeatresIndexList, outputData):
        classLabelSet = set(outputData)
        emptyClassNumMap = {}
        for classLabel in classLabelSet:
            emptyClassNumMap[classLabel] = 0.
            
        totalNumOfSamples = len(inputData)#样本的总数，用来计算某个特征取值出现的概率
        ##############开始统计各个特征的取值在样本中出现的次数#############
        #这种统计在朴素贝叶斯等算法中是常用的，通常用来计算需要的概率
        valueSampleNumMap = {}#存储各个特征的各个取值下，各类样本的数量分布，用于计算按照特征取值分组后的信息熵
        classSampleNumMap = {}#存储各类样本的数量部分，用于计算分组之前的信息熵
        for j in range(totalNumOfSamples):#遍历剩下的每一个特征，计算各自对应的信息熵
            line = inputData[j]
#             print(line)
            classLabel = outputData[j]
#             print(classLabel)
            classSampleNumMap[classLabel] = classSampleNumMap.get(classLabel, 0.) + 1.
            for i in leftFeatresIndexList:#遍历每个剩余特征的编号(就是索引值)
                featureValue = line[i]#当前特征的取值
                if i in valueSampleNumMap:#如果这个特征编号已经收录
                    
                    if featureValue in valueSampleNumMap[i]:
                        valueSampleNumMap[i][featureValue][classLabel] += 1.
                    else:
                        valueSampleNumMap[i][featureValue] = copy.deepcopy(emptyClassNumMap)
                        valueSampleNumMap[i][featureValue][classLabel] += 1.
#                     if featureValue=="晴":
#                         print(line)
#                         print(featureValue, valueSampleNumMap[i][featureValue])
                else:
                    valueSampleNumMap[i] = {}
                    if i in valueSampleNumMap:#如果这个特征编号已经收录
                        if featureValue in valueSampleNumMap[i]:
                            valueSampleNumMap[i][featureValue][classLabel] += 1.
                        else:
                            valueSampleNumMap[i][featureValue] = copy.deepcopy(emptyClassNumMap)
                            valueSampleNumMap[i][featureValue][classLabel] += 1.
        print("初步统计的结果是", valueSampleNumMap) 
        print("各个分类的数量分布是", classSampleNumMap)         
        ##############完成统计各个特征的取值在样本中出现的次数#############
        #####开始计算分组之前的信息熵######
        entropy_before_group = 0.
        for classLabel in classSampleNumMap:
            p_this_class = classSampleNumMap[classLabel]/totalNumOfSamples
            entropy_before_group -= p_this_class*np.log2(p_this_class)
        #####完成计算分组之前的信息熵######
        #######开始计算按照各个特征分组之后的信息熵(条件熵)###########
        entropyMap = {}
        for featureNO in leftFeatresIndexList:
            valueFreqMap = valueSampleNumMap[featureNO]#取出这个特征的各个取值水平下，各个类别的数量分布
            featureEntropy = 0.#这个按照这个特征分组后的信息熵
            for featureValue in valueFreqMap:#计算各个取值水平对应的样本的信息熵
                numOfEachClassMap = valueFreqMap[featureValue]#取出这个取值水平对应样本的列别数量分布
                numOfSamplesWithFeatureValue = np.sum(list(numOfEachClassMap.values()))#计算这个取值水平对应的样本总数
                featureValueEntropy = 0.
                for classLabel in numOfEachClassMap:#遍历每一个类别
                    #这个取值水平对应的样本中，类别为当前类别的概率
                    p_featureValue_class = numOfEachClassMap[classLabel]/numOfSamplesWithFeatureValue
#                     print(p_featureValue_class)
                    if p_featureValue_class==0:
                        pass
                    else:
                        featureValueEntropy -= p_featureValue_class*np.log2(p_featureValue_class)
#                     print(featureNO, featureValue, classLabel, featureValueEntropy, -p_featureValue_class*np.log2(p_featureValue_class))
                p_feature_value = numOfSamplesWithFeatureValue/totalNumOfSamples#这个特征取值出现的概率
                featureEntropy += p_feature_value*featureValueEntropy
                #信息熵公式entropy= - p_1*log(p_1) - p_2*log(p_2) - ... - p_k*log(p_k)
            entropyMap[featureNO] = featureEntropy
        print("按照各个特征分组后的信息熵是", entropyMap)
        #######完成计算按照各个特征分组之后的信息熵###########
        
        #计算信息增益
        IGMap = {}
        for featureNO in entropyMap:
            IGMap[featureNO] = entropy_before_group - entropyMap[featureNO]
        print("各个特征对应的信息增益是", IGMap)
        IGList = sorted(IGMap.items(), key=lambda x: x[1],reverse=True)
        bestFeatureNO = IGList[0][0]
        return bestFeatureNO

    #基于信息增益比率(information gain ratio, IGR)来挑选剩余特征中最好的分组特征
    def chooseBestFeatureWithIGR(self, inputData, leftFeatresIndexList, outputData):
        classLabelSet = set(outputData)
        emptyClassNumMap = {}
        for classLabel in classLabelSet:
            emptyClassNumMap[classLabel] = 0.
            
        totalNumOfSamples = len(inputData)#样本的总数，用来计算某个特征取值出现的概率
        ##############开始统计各个特征的取值在样本中出现的次数#############
        #这种统计在朴素贝叶斯等算法中是常用的，通常用来计算需要的概率
        valueSampleNumMap = {}#存储各个特征的各个取值下，各类样本的数量分布，用于计算按照特征取值分组后的信息熵
        classSampleNumMap = {}#存储各类样本的数量部分，用于计算分组之前的信息熵
        for j in range(totalNumOfSamples):#遍历剩下的每一个特征，计算各自对应的信息熵
            line = inputData[j]
            classLabel = outputData[j]
            classSampleNumMap[classLabel] = classSampleNumMap.get(classLabel, 0.) + 1.
            for i in leftFeatresIndexList:#遍历每个剩余特征的编号(就是索引值)
                featureValue = line[i]#当前特征的取值
                if i in valueSampleNumMap:#如果这个特征编号已经收录
                    if featureValue in valueSampleNumMap[i]:
                        valueSampleNumMap[i][featureValue][classLabel] += 1.
                    else:
                        valueSampleNumMap[i][featureValue] = copy.deepcopy(emptyClassNumMap)
                        valueSampleNumMap[i][featureValue][classLabel] += 1.
                else:
                    valueSampleNumMap[i] = {}
                    if i in valueSampleNumMap:#如果这个特征编号已经收录
                        if featureValue in valueSampleNumMap[i]:
                            valueSampleNumMap[i][featureValue][classLabel] += 1.
                        else:
                            valueSampleNumMap[i][featureValue] = copy.deepcopy(emptyClassNumMap)
                            valueSampleNumMap[i][featureValue][classLabel] += 1.
        ##############完成统计各个特征的取值在样本中出现的次数#############
        #####开始计算分组之前的信息熵######
        entropy_before_group = 0.
        for classLabel in classSampleNumMap:
            p_this_class = classSampleNumMap[classLabel]/totalNumOfSamples
            entropy_before_group -= p_this_class*np.log2(p_this_class)
        #####完成计算分组之前的信息熵######
        #######开始计算按照各个特征分组之后的信息熵###########
        entropyMap = {}
        entropyGroupByFeatreOnlyMap = {}
        for featureNO in leftFeatresIndexList:
            valueFreqMap = valueSampleNumMap[featureNO]#取出这个特征的各个取值水平下，各个类别的数量分布
            featureEntropy = 0.#这个按照这个特征分组后的信息熵
            entropyGroupByFeatreOnlyMap[featureNO] = 0.
            for featureValue in valueFreqMap:#计算各个取值水平对应的样本的信息熵
                numOfEachClassMap = valueFreqMap[featureValue]#取出这个取值水平对应样本的列别数量分布
                numOfSamplesWithFeatureValue = np.sum(list(numOfEachClassMap.values()))#计算这个取值水平对应的样本总数
                featureValueEntropy = 0.
                for classLabel in numOfEachClassMap:#遍历每一个类别
                    #这个取值水平对应的样本中，类别为当前类别的概率
                    p_featureValue_class = numOfEachClassMap[classLabel]/numOfSamplesWithFeatureValue
                    if p_featureValue_class==0:
                        pass
                    else:
                        featureValueEntropy -= p_featureValue_class*np.log2(p_featureValue_class)
                p_feature_value = numOfSamplesWithFeatureValue/totalNumOfSamples#这个特征取值出现的概率
                featureEntropy += p_feature_value*featureValueEntropy
                #信息熵公式entropy= - p_1*log(p_1) - p_2*log(p_2) - ... - p_k*log(p_k)
                entropyGroupByFeatreOnlyMap[featureNO] -= p_feature_value*np.log2(p_feature_value)
            entropyMap[featureNO] = featureEntropy
        #######完成计算按照各个特征分组之后的信息熵###########
        
        #计算信息增益
        IGMap = {}
        for featureNO in entropyMap:
            IGMap[featureNO] = entropy_before_group - entropyMap[featureNO]
        print("各个特征对应的信息增益是", IGMap)
        #计算信息增益比率
        IGRMap = {}
        for featureNO in entropyMap:
            IGRMap[featureNO] = entropyMap[featureNO]/entropyGroupByFeatreOnlyMap[featureNO]
            
        IGRList = sorted(IGMap.items(), key=lambda x: x[1],reverse=True)
        print("信息增益比是", IGRList)
        bestFeatureNO = IGRList[0][0]
        return bestFeatureNO

#基于测试数据检查算法正确性
def check():
    #获取测试数据https://www.cnblogs.com/kanjian2016/p/7746005.html
    data = [['晴', '炎热', '高', '弱', '取消'], 
            ['晴', '炎热', '高', '强', '取消'], 
           ['晴', '适中', '正常', '强', '进行'],
              ['晴', '适中', '高', '弱', '取消'], 
              ['晴', '寒冷', '正常', '弱', '进行'], 
            ['阴', '炎热', '高', '弱', '进行'], 
            ['雨', '适中', '高', '弱', '进行'], 
            ['雨', '寒冷', '正常', '弱', '进行'],
             ['雨', '寒冷', '正常', '强', '取消'],
              ['阴', '寒冷', '正常', '强', '进行'], 

              ['雨', '适中', '正常', '弱', '进行'],
                ['阴', '适中', '高', '强', '进行'], 
                ['阴', '炎热', '正常', '弱', '进行'], 
                ['雨', '适中', '高', '强', '取消']]
    inputData = []
    outputData = []
    for line in data:
        inputData.append(line[:4])
        outputData.append(line[4])
    
    #初始化决策树对象
    clf = ID3DecisionTree()
    leftFeatresIndexList = list(range(len(inputData[0])))#当前剩余的特征数是全部
    clf.chooseBestFeatureWithEntropy(inputData, leftFeatresIndexList, outputData)
    clf.chooseBestFeatureWithIG(inputData, leftFeatresIndexList, outputData)
    clf.chooseBestFeatureWithIGR(inputData, leftFeatresIndexList, outputData)

if __name__ == '__main__':
    #特征:体重{1:轻, 2:中等， 3: 重}，身高{1：矮， 2：中等， 3：高}，性别{1: 男， 2：女}
    #类别{1:成年,2:未成年}
    
#     inputData = [[1, 1, 2], [2, 3, 2], [3, 3, 1], [2, 1, 1]]
#     outputData = [2, 1, 1, 2]
#     clf = ID3DecisionTree()
#     clf.fit(inputData, outputData)
#     preds = clf.predict(inputData)
#     print(outputData)
#     print(preds)
    check()

