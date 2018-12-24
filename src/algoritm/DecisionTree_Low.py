#一个用来对简单情况下的样本，即取值空间有限的离散特征，进行分类的决策树。
#输入可以是字符串或者整数，保证是类别变量就可以。
import numpy as np
import copy

class Tree4Decision():
    def __init__(self):
        self.featureNO = -1#当前节点对应的特征编号
        self.featureValue = None#当前节点对应的特征取值
        self.children = {}#当前节点的子节点
        self.classLabel = None#类别标签

class LowDecisionTree():
    #一个比较low的决策树，使用信息熵来选择分组特征
    #数据科学是一个交叉领域，使用了很多学科的思想和技术。在表述同一个概念的时候，不同知识背景的人会使用不同的理论或者语言。
    #比如这里的"分组特征"，实际上是本人的口语化表达；教材里通常称为"划分属性"(split criteria)，大家把split criteria翻译成
    #各式各样的词语，这样会给我们带来很大的干扰。为了减小这种干扰，大部分教材会给一些重要的词语后面加上(英文及其简称)。记住这些英
    #文名称并在文章中使用，可以阅读者消歧的工作量
    def __init__(self):
        self.decisionTree = None#用于存储决策树
        pass

    #“训练决策树”这个提法有误导性，在"训练"开始之前，决策树还没有存在，没啥可以训练的。我们需要做的实际上是"构建"一个决策树。
    #fit 这个词语换成generate之类的更合适。
    #决策树的生成过程非常适合用递归的方式来实现。
    def fit(self, inputData, outputData):
        rootNode = Tree4Decision()#初始化根节点
        leftFeatresIndexList = list(range(len(inputData[0])))#根节点对应的未使用特征列表就是所有特征
        self.generateDesisionTree(inputData, outputData, rootNode,  leftFeatresIndexList)#生成决策树
        self.decisionTree = rootNode
#         print(rootNode.__dict__)
        self.showTree()
    
    #预测一批样本的类别
    def predictOne(self, inputData):
        ifHasChildren = True
        childrenTree = self.decisionTree
        classLabel = None
        while ifHasChildren==True:
            featureNO = childrenTree.featureNO
            featureValue = inputData[featureNO]
#             print(featureNO, featureValue, childrenTree.__dict__)
            if childrenTree.children=={}:
                ifHasChildren = False
                classLabel = childrenTree.classLabel
            else:
                childrenTree = childrenTree.children[featureValue]
        return classLabel  
    
    #把决策树的结构打印到控制台
    def showTree(self):
        print("开始展示决策树", self.decisionTree.__dict__)
        self.showTreeRecusively(self.decisionTree)
#         print(self.decisionTree.children[3].__dict__)
        
    def showTreeRecusively(self, currentNode):
#         print(currentNode)
        if currentNode.children=={}:
            return
        else:
#             print(currentNode.children)
            for featureValue in currentNode.children:
                node = currentNode.children[featureValue]
                self.showTreeRecusively(node)
                print(node.__dict__)
    
    def calPurity(self, outputData):
        labels = set(outputData)
        labelNumMap = {}
        for label in labels:
            labelNumMap[label] = labelNumMap.get(label, 0) + 1
        labelNumList = sorted(labelNumMap.items(), key=lambda x: x[1], reverse=True)
        mostLabel = labelNumList[0][0]
        purity = labelNumMap[mostLabel]/len(outputData)
        return purity
    
    def generateDesisionTree(self, inputData, outputData, currentNode, leftFeatresIndexList):
        purity = self.calPurity(outputData)
#         if len(set(outputData))==len(outputData):#决策树生长到这样一个节点时需要停止，停止的基本策略有两种:
            #(1)在决策树的生长过程中，基于特定的规则停止生长，从而获得精简的决策树——这样的策略叫做预剪枝(Pre-Pruning).
            #(2)对应的，还有一种剪枝策略，称为后剪枝(post-pruning):在决策树完全生长后，再基于特定规则删除不需要的节点。
            #在基本策略的基础上，我们可以组合或者改造出适合场景的各种生长策略
            #这里使用的是预剪枝策略，停止生长的条件非常粗暴，就是“这个节点的特征取值对应的所有样本属于同一个类别”。实际上
            #我们可以构造"纯度"之类的指标，比如某个类别的占比达到60%,这个节点对应的样本就足够纯，停止生长;或者对各个类别加权再计算纯度;
            #或者我们可以计算信息熵;等等。
        if purity>0.8 or leftFeatresIndexList==[]:
            currentNode.classLabel = outputData[0]
            return currentNode
        else:
            #选择最好的划分属性
            bestSplitFeatureNO = self.chooseBestFeatureWithEntropy(inputData, leftFeatresIndexList, outputData)
            currentNode.featureNO = bestSplitFeatureNO#当前节点的最优划分属性
            sampleGroupMap = {}#按照划分属性的取值水平来对样本进行分组
            for i in range(len(outputData)):#遍历每一个样本，按照最优划分属性对样本进行分组
                sampleInputData = inputData[i]
                sampleOutputData = outputData[i]
                bestFeatureValue = sampleInputData[bestSplitFeatureNO]
                if bestFeatureValue in sampleGroupMap:
                    sampleGroupMap[bestFeatureValue]['inputData'].append(sampleInputData)
                    sampleGroupMap[bestFeatureValue]['outputData'].append(sampleOutputData)
                else:
                    sampleGroupMap[bestFeatureValue] = {'inputData': [sampleInputData], 
                                                        'outputData': [sampleOutputData]}
            leftFeatresIndexList.remove(bestSplitFeatureNO)#更新剩余特征编号列表
            for featureValue in sampleGroupMap:#遍历按照最优化分属性取值水平分组的样本，再对每一组样本进行最优划分特征选择等操作
                thisNode = Tree4Decision()#初始化一个子节点
                thisNode.featureNO, thisNode.featureValue, thisNode.children= \
                            bestSplitFeatureNO, featureValue, {}#初始化这个取值对应的子节点
#                 print(currentNode.__dict__)
                currentNode.children[featureValue] = thisNode#把子节点放到当前节点里
                self.generateDesisionTree(sampleGroupMap[featureValue]['inputData'],
                                          sampleGroupMap[featureValue]['outputData'],
                                          thisNode,
                                          leftFeatresIndexList)
                
        
    #基于特征的信息熵，从未使用的特征中挑选最好的分组特征
    def chooseBestFeatureWithEntropy(self, inputData, leftFeatresIndexList, outputData):
        totalNumOfSamples = len(inputData)#样本的总数，用来计算某个特征取值出现的概率
        print(leftFeatresIndexList)
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
        print(valueSampleNumMap)
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
        print(entropyList)
        bestFeatureNO = entropyList[0][0]#取出熵最大的特这个编号并返回
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
    clf = LowDecisionTree()
    leftFeatresIndexList = list(range(len(inputData[0])))#当前剩余的特征数是全部
    clf.fit(inputData, outputData)
    print(outputData)
    preds = clf.predictOne(inputData[6])
    print(preds)   
if __name__ == '__main__':
    check()

    #特征:体重{1:轻, 2:中等， 3: 重}，身高{1：矮， 2：中等， 3：高}，性别{1: 男， 2：女}
    #类别{1:成年,2:未成年}
    
#     inputData = [[1, 1, 2], [2, 3, 2], [3, 3, 1], [2, 1, 1],[2, 2, 1]]
#     outputData = [2, 1, 1, 1]
#     clf = LowDecisionTree()
#     clf.fit(inputData, outputData)
#     print(outputData)
#     preds = clf.predictOne(inputData[0])
#     print(preds)

