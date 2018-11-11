import numpy as np
import sys
import os
path = os.getcwd()
path = os.path.dirname(path)
sys.path.append(path)
from dataProcessing import dataSets
import modelEvaluate

class NaiveBayesForDicreteFeature():
    def __init__(self):
        self.labelSetList = []#用来存储类标签的集合，
        # 求出各类概率后，就按照索引从这里查询到类标签返回
        self.numClass = 0#类别个数，用来创建存储各类概率的向量等
        self.classProbLList = None#存储各个类别的先验概率
        self.wordCountMap = {}#用来存储每一个词语在训练集中出现的次数
        self.wordFreqMap = {}#用来存储每一个词语在各个分类中出现的概率
        #以上就是朴素贝叶斯模型的实体部分，我们需要从训练数据中统计出这几
        #个数据，然后就可以依靠它们去计算一个待预测样本的类别了。

    def train(self, dataList, labelList):
        self.labelSetList = list(set(labelList))#训练样本的类别标签是齐全的(必须的)
        #这里把所有的类别标签收集起来
        self.numClass = len(self.labelSetList)#获取类别的个数
        sampleNumEachClassMap = {}#用来统计各个类别的样本的数量
        for label in labelList:
            sampleNumEachClassMap[label] = sampleNumEachClassMap.get(label, 0) + 1.0
        totalNumSample = float(len(labelList))#用来统计
        classProbMap = {}  # 用来存储各个类别的先验概率
        for key in sampleNumEachClassMap:
            classProbMap[key] = np.log(sampleNumEachClassMap[key] / totalNumSample)
        # for key in sampleNumEachClassMap:
        #     totalNumSample += sampleNumEachClassMap[key]
        self.classProbLList = np.zeros(self.numClass)
        for label in self.labelSetList:
            labelIndex = self.labelSetList.index(label)
            # 把每个类别的先验概率放到规定位置里，便于后面计算
            self.classProbLList[labelIndex] = classProbMap[label]

        for i in range(len(dataList)):
            words = dataList[i]
            label = labelList[i]
            labelIndex = self.labelSetList.index(label)#获取当前样本的类别标签在
            #self.labelSetList中的位置
            for word in words:
                #统计word这个词语出现的总数
                self.wordCountMap[word] = self.wordCountMap.get(word, 0) + 1
                if word not in self.wordFreqMap:
                    tempCount = np.zeros(self.numClass)  # 一个长度为self.numClass的一维向量
                    # 用来存储word这个词语在各个类别的样本中出现的次数
                    tempCount += 0.0000001  # 如果这个词语没有在某些类别中出现过，比如政治类新闻里不会出现
                    # "大力灌筐"这个词语(目测是肯定不会),就会导致tempCount里有一个元素取值为0,我们后面要用
                    # 它做乘法以及取对数操作，结果会不好看。最最最简单粗暴的做法，就是给一个极小值；
                    # 效果不好的话，咱们可以考虑一下高级手段，平滑啥的。
                    tempCount[labelIndex] += 1.0
                    self.wordFreqMap[word] = tempCount  # 把前面统计得到的次数给放到dict里，便于
                    # 以后计算时快速查询
                else:
                    self.wordFreqMap[word][labelIndex] += 1.0

        wordsInEachClass = np.zeros(self.numClass)
        for word in self.wordFreqMap:
            wordsInEachClass += self.wordFreqMap[word]#统计各个类别样本的总词数

        for word in self.wordFreqMap:
            self.wordFreqMap[word] = self.wordFreqMap[word] / wordsInEachClass
            # 这里的操作对应的就是贝叶斯
            # 公式里的分子里的条件概率(后验概率，就是观察到事件A的情况下，比如知道新闻类别是政治类，
            # 发生事件B,比如看到这篇新闻里有"大力灌筐"这个词语的概率)
            self.wordFreqMap[word] = np.log(self.wordFreqMap[word] / self.wordCountMap[word])
            #这里求一个对数，有两方面的原因:(1)防止下溢出，一篇文章里有
            # 250个词语，而其中的每一个词语在文本中出现的概率是0.1的话，乘积就小于我们的小破电脑的
            # 计数范围了；(2)提升计算效率，贝叶斯公式的右边取对数后，就成了加法，小破电脑算
            # 起加法来比乘法快多了(电脑算乘法n*m可以粗暴理解为把n累加m次，累;如果m比较大，
            # log(n) + log(m)是相对较快的，取对数虽然慢，但是只取两次啊)
            #由于使用贝叶斯分类器的时候，各个类别的贝叶斯公式表达式的分母是一样的，不影响大小顺序，
            # 大家可以不关心分母

    def predict(self, dataList):
        res = []
        for words in dataList:
            probs = np.zeros(self.numClass)#存储各类的概率
            for word in words:
                if word not in self.wordFreqMap:
                    #如果word这个词语不在模型里(我们的词频map里没有收录这个词语)
                    #就跳过(这里加0)
                    probs += 0  # np.ones(numClass)
                else:#如果词语被收录了，把条件概率"加上"
                    probs += self.wordFreqMap[word]
            probs += self.classProbLList#把条件概率个先验概率"加"起来，
            # 就是我们对各个类别概率的估计了
            indexClass = list(probs).index(max(probs))#获取概率最大的那个类别标签的序号
            pred = self.labelSetList[indexClass]#得到这个样本的类别标签
            res.append(pred)
        return res

if __name__ == '__main__':
    dataList, labelList, X_test, y_test = dataSets.processWeibo()
    dataList = [['大力灌筐', '加油', '好球', '三分'], ['三国', '三分', '二哥', '麦城'],
    ['加油','好球','进了'], ['三分', '麦城', '败走', '华容道']]
    labelList = [1,0,1, 0]
    X_test, y_test =dataList, labelList#作弊一把，用训练集来做测试集
    clf = NaiveBayesForDicreteFeature()
    clf.train(dataList, labelList)
    res = clf.predict(X_test)
    for i in range(len(res)):
        print("真实标签是：", y_test[i], "。预测标签是：", res[i])
    modelEvaluate.showConfusionMatrix(res, y_test, len(set(labelList)))

