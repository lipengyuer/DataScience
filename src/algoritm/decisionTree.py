#一个用来对简单情况下的样本，即取值空间有限的离散特征，进行分类的决策树。
#输入可以是字符串或者整数，保证是类别变量就可以。

class ID3DecisionTree():

    def __init__(self):
        aTree = {}

    def fit(self, inputData, outputData):
        pass

    def choseBestFeature(self, inputData, leftFeatresIndexList, outputData):
        entropyMap = {}
        valueSampleNumMap = {}#存储各个特征的各个取值出现的次数

        for line in inputData:#遍历剩下的每一个特征，计算各自对应的信息熵
            tempLine = []
            for i in leftFeatresIndexList:
                tempLine.append(line[i])
            leftInputData.append(tempLine)

        for i in range(len(leftInputData[0])):




if __name__ == '__main__':
    #特征:体重{1:轻, 2:中等， 3: 重}，身高{1：矮， 2：中等， 3：高}，性别{1: 男， 2：女}
    #类别{1:成年,2:未成年}
    inputData = [[1, 1, 2], [2, 3, 2], [3, 3, 1], [2, 1, 1]]
    outputData = [2, 1, 1, 2]
    clf = ID3DecisionTree()
    clf.fit(inputData, outputData)
    preds = clf.predict(inputData)
    print(outputData)
    print(preds)