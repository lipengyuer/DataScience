import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
#展示不同分类的样本某个指标的分布，输入是两个列表
def showDistrbutionOfEachClass(dataList, labelList):
    dataDict = {}
    for i in range(len(labelList)):
        label = labelList[i]
        data = dataList[i]
        if label in dataDict:
            dataDict[label].append(data)
        else:
            dataDict[label] = [data]
    plt.figure()
    dataList = dataDict.values()
    labelList = dataDict.keys()
    plt.hist(list(dataList),  bins = int(180/15), label=list(labelList))
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    datas = [1,1,3,4,45,6,1,23,1]
    labels = [1,1,1,2,2,2,3,3,3]
    print(len(datas), len(labels))
    showDistrbutionOfEachClass(datas, labels)

    
    
    
    
    
    
    
    