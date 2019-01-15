from Softmax import Softmax4CNN
import pickle
import numpy as np
import random, copy
from CNN import CNN
# 加载手写体数字识别数据集
def loadData():
    import tensorflow as tf
    data = tf.contrib.learn.python.learn.datasets.mnist.read_data_sets('../../data/mnist/', one_hot=True)
    testData = data.test
    trainData = data.train
    testInput, testOutput = testData.images, testData.labels
    trainInput, trainOutPut = trainData.images, trainData.labels
    simpleDataSet = [testInput[0:10], testOutput[0:10], trainInput[0:10], trainOutPut[0:10]]
    pickle.dump(simpleDataSet, open('simpleMnist.pkl', 'wb'))


def loadSimpleData():
    [testInput, testOutput, trainInput, trainOutPut] = pickle.load(open('simpleMnist.pkl', 'rb'))
    return testInput, testOutput, trainInput, trainOutPut


def loadImageOfSix():
    imageOfSix = [[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1], [1, 1, 1, 1, 1, 1]]
    imageOfSix = np.array(imageOfSix)
    return [imageOfSix], [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]

class CNNSoftmax():
    def __init__(self, picShape, classNum, epochNum=1):
        self.classNum = classNum
        self.layers = []  # 用于存储各层参数
        self.picShape = picShape  # 初始化的时候，需要手动输入图片的高和宽，用来推测后面的一系列参数
        self.createNetwork()
        self.epochNum = epochNum

    def createNetwork(self):  # 可以在
        cnnLayer1 = CNN(self.picShape, 1,
                        kernelNum=1, colStride=2, receptiveFieldSize=3, poolingSize=2, poolingStride=2)
        print(cnnLayer1.outputImageShape, cnnLayer1.outputImageNum)

        cnnLayer2 = CNN(cnnLayer1.outputImageShape, cnnLayer1.outputImageNum,
                        kernelNum=1, colStride=2, receptiveFieldSize=3, poolingSize=2, poolingStride=2)
        height, width, num = cnnLayer2.outputImageShape[0], cnnLayer2.outputImageShape[1], cnnLayer2.outputImageNum
        softmaxLayer = Softmax4CNN(height * width * num, self.classNum)
        self.layers = [cnnLayer1, cnnLayer2, softmaxLayer]

    def shuffleData(self, inputList, outputList):
        indexList = list(range(len(inputList)))
        random.shuffle(indexList)
        resInput, resOutput = [], []
        for i in indexList:
            resInput.append(inputList[i])
            resOutput.append(outputList[i])
        return resInput, resOutput

    def calGrad(self, anImage, realLabel):  # 梯度计算
        self.gradList = [None for _ in range(len(self.layers))]  # 每一层的梯度数据
        # self.gradList[-1] = self.layers[-1].calGrad()

    # 更新参数
    def updateWeights(self, gradList):
        pass

    # 使用BP算法，训练模型
    def fit(self, trainingImageList, trainingLabelList):
        for epoch in range(self.epochNum):
            trainingImageList, trainingLabelList = self.shuffleData(trainingImageList, trainingLabelList)
            for i in range(len(trainingImageList)):
                anImage, realLabel = trainingImageList[i], trainingLabelList[i]
                gradList = self.calGrad(anImage, realLabel)
                self.updateWeights(gradList)
                if i % 10 == 0:
                    cost = self.calCost(trainingImageList[:10], trainingLabelList[:10])
                    print("已经学习了", epoch, '轮, cost为', cost)

    def predict(self, anImage):
        anImage = [anImage]
        for layer in self.layers:
            anImage = layer.predict(anImage)
        #            print(anImage)
        return anImage

    #训练过程中需要使用的预测函数，会把各层的输出保留起来，用于计算梯度和误差
    def predict4Train(self, anImage):
        for layer in self.layers:
            anImage = layer.predict4Trian(anImage)

    # 计算损失值
    def calCost(self, trainingImageList, trainingLabelList):
        cost = 0
        for j in range(len(trainingImageList)):
            predLabel = self.predict([trainingImageList[j]])
            realLabel = trainingLabelList[j]
            for i in range(self.classNum): cost -= realLabel[i] * np.log(predLabel[i])
        print("损失值是", cost)



if __name__ == '__main__':
    #     loadData()
    #     testInput, testOutput, trainInput, trainOutPut = loadSimpleData()
    testInput, testOutput = loadImageOfSix()
    #     print(testOutput[:3])
    #     cnn = CNN(kernelNum=5, colStride=2, receptiveFieldSize=3, poolingSize=2, poolingStride=2)
    #     cnn.calOutput(testInput)
    clf = CNNSoftmax([6, 6], 10)
    clf.fit(testInput, testOutput)
    pred = clf.predict(testInput[0])
    print(pred)

