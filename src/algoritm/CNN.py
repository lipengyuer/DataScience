#实现一个简单的CNN+softmax分类器，其中CNN的深度可以自定义，采用LeNet的连接方式。对手写体数字识别数据进行分类

import pickle
import numpy as np
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
    def __init__(self, kernelNum = 5, colStride=2, receptiveFieldSize = 3, poolingSize=2, poolingStride=2):
        self.kernelNum = kernelNum#本层卷积核的个数
        self.colStride = colStride#卷积步长
        self.receptiveFieldSize = receptiveFieldSize#卷积核的感受野的大小，这里为了简单，采用方形感受野。要求边长为奇数
        #这样便于padding规则的简化，同时便于感受野的定位。
        self.poolingSize = poolingSize#池化操作的采样区域边长
        self.poolingStride = poolingStride
        self.weightListOfKernels = []#存储每个卷积核的权重矩阵
        self.biasList = []#存储每个卷积核的偏置
        #初始化参数
        self.initAll()
        
    def initAll(self):
        for _ in range(self.kernelNum):
            weight = np.random.rand(self.receptiveFieldSize, self.receptiveFieldSize)
            bias = np.random.normal()
            self.weightListOfKernels.append(weight)
            self.biasList.append(bias)
#         print(self.weightListOfKernels, self.biasList)
    
    def padding(self, inputImageList):#对图像进行填充
        newImageList = []
        incLength = (self.receptiveFieldSize-1)
        incLengthHalf = int(incLength/2)
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
        outputImage = np.zeros((oriHeight,oriWidth))
        for i in range(incLengthHalf, incLengthHalf + oriHeight, stride):
            for j in range(incLengthHalf, incLengthHalf + oriWidth, stride):
                pointsInField = newImage[i-incLengthHalf: i+incLengthHalf+1, j-incLengthHalf: j+incLengthHalf+1]
#                 print(np.sum(pointsInField*weightMatrix) + bias)
#                 print(np.sum(pointsInField*weightMatrix))
                outputImage[i-incLengthHalf,j-incLengthHalf+1] = self.relu(np.sum(pointsInField*weightMatrix) + bias)
        return outputImage
    
    def pooling(self, anImage):#采样窗口和步长是随意设置的，需要这里自动适应
        oriHeight, oriWidth = anImage.shape
        newImage = np.zeros((oriHeight-self.poolingSize+1))
        for i in range(0, oriHeight-self.poolingSize+1, self.poolingStride):
            for j in range(0, oriWidth-self.poolingSize+1, self.poolingStride):
                
        
        
    def calOutput(self, inputImageList):
        outputImageList = []
        #首先填充.这里为了简单，没有提供不填充的选项
        newInputImageList, incLengthHalf = self.padding(inputImageList)
        oriHeight, oriWidth = inputImageList[0].shape
        numOfOriImage = len(inputImageList)
#         print(inputImageList)
        #然后让每一个卷积核扫描特定的图片，分别得到若干图片的抽象
        numOfInputImage4AKernel = int(numOfOriImage/self.kernelNum)#求每一个卷积核需要扫描的图片个数
        
        imageIndexOfEachKernel = []
        if numOfInputImage4AKernel<=1:#卷积核个数大于图片个数
            tempIndexList = []
            for i in range(int(self.kernelNum/numOfOriImage)+1):
                tempIndexList += list(range(numOfOriImage))
            for i in range(self.kernelNum):
                imageIndexOfEachKernel.append([tempIndexList[i], tempIndexList[i]])
        else:
            tempIndexList = list(range(numOfOriImage)) + list(range(numOfOriImage))
            for i in range(self.kernelNum):
                imageIndexOfEachKernel.append(tempIndexList[i: i + numOfInputImage4AKernel])            
        
        for i in range(len(imageIndexOfEachKernel)):
            imageIndexOfThisKernel = imageIndexOfEachKernel[i]
            start, end = imageIndexOfThisKernel[0], imageIndexOfThisKernel[-1]
            for index in range(start, end+1):
                anImage = newInputImageList[index]
                outputImage = self.colIt(anImage, oriHeight, oriWidth, incLengthHalf, self.weightListOfKernels[i],self.biasList[i], self.colStride)
                print(outputImage)
                print("###################")
                outputImageList.append(outputImage)
        
        
            
        
        
        
        
        
        
if __name__ == '__main__':
#     loadData()
#     testInput, testOutput, trainInput, trainOutPut = loadSimpleData()
    testInput, testOutput = loadImageOfSix()
    print(testOutput[:3])
    cnn = CNN()
    cnn.calOutput(testInput)
    
    
    