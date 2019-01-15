#实现一个简单的CNN+softmax分类器，其中CNN的深度可以自定义，采用LeNet的连接方式。对手写体数字识别数据进行分类

import pickle
import numpy as np
import random, copy

#加载手写体数字识别数据集
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
    imageOfSix = [[1,1,1,1,1,1], [1,0,0,0,0,0],[1,1,1,1,1,1],[1,0,0,0,0,1],[1,1,1,1,1,1]]
    imageOfSix = np.array(imageOfSix)
    return [imageOfSix], [[0,0,0,0,0,0,1,0,0,0]]

#一个卷积层，
class CNN():
    def __init__(self, inputImageShapeFromFormerLayer, inputImageNumFromFormerLayer, kernelNum = 5, colStride=2, receptiveFieldSize = 3, poolingSize=2, poolingStride=2):
        self.inputImageShapeFromFormerLayer = inputImageShapeFromFormerLayer
        self.inputImageNumFromFormerLayer = inputImageNumFromFormerLayer
        self.kernelNum = kernelNum#本层卷积核的个数
        self.colStride = colStride#卷积步长
        self.receptiveFieldSize = receptiveFieldSize#卷积核的感受野的大小，这里为了简单，采用方形感受野。要求边长为奇数
        #这样便于padding规则的简化，同时便于感受野的定位。
        self.poolingSize = poolingSize#池化操作的采样区域边长
        self.poolingStride = poolingStride
        self.weightListOfKernels = []#存储每个卷积核的权重矩阵
        self.biasList = []#存储每个卷积核的偏置

        self.outputImageNum = None#输出图像的个数
        self.outputImageShape = None#输出图像的形状用于提供给后一层CNN来做相关尺寸的计算，从而推算出最后一层CNN的输出尺寸
        #用来初始化softmax的权重向量
        #初始化参数
        self.initAll()
        
    def initAll(self):
        for _ in range(self.kernelNum):
            weight = np.random.rand(self.receptiveFieldSize, self.receptiveFieldSize)
            bias = np.random.normal()
            self.weightListOfKernels.append(weight)
            self.biasList.append(bias)
#         print(self.weightListOfKernels, self.biasList)
        self.calOutputInfo()

    #基于出入图像的尺寸和数量，以及卷积核等的情况，计算输出的图像的个数和尺寸
    def calOutputInfo(self):
        #生成来自上一层的模拟数据
        imageList = [np.zeros(self.inputImageShapeFromFormerLayer) for _ in range(self.inputImageNumFromFormerLayer)]
        outputImageList = self.predict(imageList)
        self.outputImageNum = len(outputImageList)
        print(outputImageList)
        self.outputImageShape = outputImageList[0].shape
        
    def padding(self, inputImageList):#对图像进行填充
        newImageList = []
        incLength = (self.receptiveFieldSize-1)
        incLengthHalf = int(incLength/2)
        print(inputImageList[0].shape)
        oriHeight, oriWidth = inputImageList[0].shape
        newHeight, newWidth = oriHeight + incLength, oriWidth + incLength
        for anImage in inputImageList:
            newImage = np.zeros((newHeight, newWidth))#创建一个0矩阵，尺寸与填充后的图像大小相同
            newImage[incLengthHalf: newHeight-incLengthHalf, incLengthHalf: newWidth-incLengthHalf] = anImage#把原始图像覆盖到0矩阵的中心位置
            newImageList.append(newImage)
        return newImageList, incLengthHalf
    #激活函数
    def relu(self, regressionRes):
#         print(regressionRes)
        if regressionRes<0: regressionRes=0
        return regressionRes
        
    def colIt(self, newImage, oriHeight, oriWidth, incLengthHalf, weightMatrix, bias, stride):#对图像进行卷积
        outputImage = []#直接用list来接收结果，算是一种偷懒的做法；效率更高的是创建尺寸合适的np.array
        for i in range(incLengthHalf, incLengthHalf + oriHeight, stride):
            tempList = []
            for j in range(incLengthHalf, incLengthHalf + oriWidth, stride):
                pointsInField = newImage[i-incLengthHalf: i+incLengthHalf+1, j-incLengthHalf: j+incLengthHalf+1]
#                 print(np.sum(pointsInField*weightMatrix) + bias)
#                 print(np.sum(pointsInField*weightMatrix))
                output = self.relu(np.sum(pointsInField*weightMatrix) + bias)
                tempList.append(output)
            outputImage.append(tempList)
        outputImage = np.array(outputImage)
        return outputImage

    def padding4Pooling(self, oriImage):
        incLength = (self.poolingSize-1)
        incLengthHalf = int(self.poolingSize/2)
        # print(incLengthHalf)
        oriHeight, oriWidth = oriImage.shape
        newHeight, newWidth = oriHeight + incLength, oriWidth + incLength
        newImage = np.zeros((newHeight, newWidth))#创建一个0矩阵，尺寸与填充后的图像大小相同
        newImage[incLengthHalf: newHeight-incLengthHalf + 1,
                        incLengthHalf: newWidth-incLengthHalf + 1] = oriImage#把原始图像覆盖到0矩阵的中心位置
        return newImage, incLengthHalf

    def pooling(self, anImage):#采样窗口和步长是随意设置的，需要这里自动适应
        oriHeight, oriWidth = anImage.shape
        newImage = []
        padImage, incLengthHalf = self.padding4Pooling(anImage)
        newHeight, newWidth = padImage.shape
        for i in range(incLengthHalf, newHeight, self.poolingStride):
            tempList = []
            for j in range(incLengthHalf, newWidth, self.poolingStride):
                sliceOfImage = padImage[i-incLengthHalf: i+incLengthHalf, j-incLengthHalf: j+incLengthHalf]
                meanV = np.mean(sliceOfImage)
                tempList.append(meanV)
            newImage.append(tempList)
        newImage = np.array(newImage)
        return newImage

    #将一批图片的索引，分成均等分，每个卷积核一份
    def splitImageList4EachKernel(self, numOfOriImage):
        # 然后让每一个卷积核扫描特定的图片，分别得到若干图片的抽象
        oriIndexList = list(range(numOfOriImage))
        numOfInputImage4AKernel = int(numOfOriImage / self.kernelNum)  # 求每一个卷积核需要扫描的图片个数
        imageIndexOfEachKernel = []
        if numOfInputImage4AKernel <= 1:  # 卷积核个数大于图片个数
            tempIndexList = []
            for i in range(int(self.kernelNum / numOfOriImage) + 1):
                tempIndexList += oriIndexList
            for i in range(self.kernelNum):
                imageIndexOfEachKernel.append([tempIndexList[i], tempIndexList[i]])
        else:
            tempIndexList = oriIndexList + oriIndexList
            for i in range(self.kernelNum):
                imageIndexOfEachKernel.append(tempIndexList[i: i + numOfInputImage4AKernel])
        return imageIndexOfEachKernel

    def predict(self, inputImageList):
        outputImageList = []
        #首先填充.这里为了简单，没有提供不填充的选项
#         print(inputImageList)
        newInputImageList, incLengthHalf = self.padding(inputImageList)
        oriHeight, oriWidth = inputImageList[0].shape
        numOfOriImage = len(inputImageList)
#         print(inputImageList)
        imageIndexOfEachKernel = self.splitImageList4EachKernel(numOfOriImage)
        
        for i in range(len(imageIndexOfEachKernel)):
            imageIndexOfThisKernel = imageIndexOfEachKernel[i]
            start, end = imageIndexOfThisKernel[0], imageIndexOfThisKernel[-1]
            for index in range(start, end+1):
                anImage = newInputImageList[index]
                outputImage = self.colIt(anImage, oriHeight, oriWidth, incLengthHalf, self.weightListOfKernels[i],self.biasList[i], self.colStride)
                outputImage = self.pooling(outputImage)
#                 print(outputImage)
#                 print("###################")
                outputImageList.append(outputImage)
        return outputImageList


class Softmax():#需要为cnn的输出做一些改动，比如需要将cnn传过来的抽象图像拉直，拼接成一个向量；
    # 特征的个数就是这个向量的长度。初始化softmax分类器参数的时候，需要接收来自前面的结果。

    def __init__(self, numOfNode, classNum, learningRate=.01, stepNum=10):
        self.weights = None  # 参数矩阵，每一行是一个类别对应的自变量系数
        self.parNum = numOfNode  # 模型里自变量的个数，后面需要初始化
        # 这里为了方便，截距被当作一个取值固定的变量来处理，系数是1.模型输入后，会初始化这个向量
        self.diffFuctions = []  # 存储每个变量对应方向的偏导数
        self.learningRate = learningRate  # 学习率。这里每个参数的学习率是一样的；我们也可以为各个参数设置不同的学习率。
        self.stepNum = stepNum  # 每一批数据学习的步数
        self.classNum = classNum
        self.init()

    def init(self):
        self.weights = [[random.uniform(-0.2, 0.2) for i in range(self.parNum)]
                     for j in range(self.classNum)]  # 初始化模型参数矩阵(self.classNum行self.parNum列)，这里使用0。
        self.weights = np.array(self.weights)  # 处理成numpy的数组，便于进行乘法等运算
        self.bias = [random.uniform(-0.2, 0.2) for j in range(self.classNum)]
        self.bias = np.array(self.bias)

    # 计算一个观测值的输出
    def predict(self, inputImageList):
#         print(inputImageList)
        inputData = np.array(inputImageList).reshape((1, self.parNum))
#         print(inputData)
        probList = np.dot(self.weights, np.transpose(inputData))
        probList = np.transpose(probList)
        probList += self.bias
        probList = probList[0]
        probList = list(probList)  # 从矩阵的第一行才是概率分布列表
        probList = self.softmax(probList)
        maxProb = np.max(probList)
        probList = list(probList)
        maxProbIndex = probList.index(maxProb)
        # 用来做预测的时候，需要将概率值二值化，也就是输出类别标签
        predLabel = [1 if i == maxProbIndex else 0 for i in range(len(probList))]
        return predLabel

    def softmax(self, xList):
        xArray = np.array(xList)
        xArray = np.exp(xArray)
        sumV = sum(xArray)
        if sumV == 0:  # 如果各家概率都是零
            result = xArray * 0
        else:
            result = xArray / sumV
        return result
            
        
class CNNSoftmax():
    def __init__(self, picShape, classNum):
        self.classNum = classNum
        self.layers = []#用于存储各层参数
        self.picShape = picShape#初始化的时候，需要手动输入图片的高和宽，用来推测后面的一系列参数
        self.createNetwork()

    def createNetwork(self):#可以在
        cnnLayer1 = CNN(self.picShape,1)
        print(cnnLayer1.outputImageShape, cnnLayer1.outputImageNum)

        cnnLayer2 = CNN(cnnLayer1.outputImageShape, cnnLayer1.outputImageNum)
        height, width, num = cnnLayer2.outputImageShape[0], cnnLayer2.outputImageShape[1], cnnLayer2.outputImageNum
        softmaxLayer = Softmax(height*width*num, self.classNum)
        self.layers = [cnnLayer1, cnnLayer2, softmaxLayer]
        
        
    def predict(self, anImage):
        anImage = [anImage]
        for layer in self.layers:
            anImage = layer.predict(anImage)
            print(anImage)
        
if __name__ == '__main__':
#     loadData()
#     testInput, testOutput, trainInput, trainOutPut = loadSimpleData()
    testInput, testOutput = loadImageOfSix()
#     print(testOutput[:3])
#     cnn = CNN(kernelNum=5, colStride=2, receptiveFieldSize=3, poolingSize=2, poolingStride=2)
#     cnn.calOutput(testInput)
    clf = CNNSoftmax([6, 6], 10)
    clf.predict(testInput[0])
    
    
    