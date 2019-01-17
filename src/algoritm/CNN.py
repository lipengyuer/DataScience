#实现一个简单的CNN+softmax分类器，其中CNN的深度可以自定义，采用LeNet的连接方式。对手写体数字识别数据进行分类

import pickle
import numpy as np
import random, copy

#一个卷积层，
class CNN():
    def __init__(self, outputShapeOFFormerLayer, kernelNum = 5, colStride=2, receptiveFieldSize = 3, \
                           poolingSize=2, poolingStride=2, learningRate = 0.0001):
        self.outputShapeOFFormerLayer = outputShapeOFFormerLayer
        self.learningRate = learningRate
        self.kernelNum = kernelNum#本层卷积核的个数
        self.colStride = colStride#卷积步长
        self.receptiveFieldSize = receptiveFieldSize#卷积核的感受野的大小，这里为了简单，采用方形感受野。要求边长为奇数
        #这样便于padding规则的简化，同时便于感受野的定位。
        self.poolingSize = poolingSize#池化操作的采样区域边长
        self.poolingStride = poolingStride
        self.weightListOfKernels = []#存储每个卷积核的权重矩阵
        self.biasList = []#存储每个卷积核的偏置
        self.grad = None
        self.outputShape = None#输出图像的个数,#输出图像的形状用于提供给后一层CNN来做相关尺寸的计算，从而推算出最后一层CNN的输出尺寸
        #用来初始化softmax的权重向量
        #初始化参数
        self.initAll()
        
    def initAll(self):
        for _ in range(self.kernelNum):
            weight = np.random.rand(self.receptiveFieldSize, self.receptiveFieldSize)
            bias = np.random.normal()
            self.weightListOfKernels.append(weight)
            self.biasList.append(bias)
        # print(self.weightListOfKernels, self.biasList)
        self.weightListOfKernels = np.array(self.weightListOfKernels)
        self.biasList = np.array(self.biasList)
        self.calOutputInfo()

    #基于出入图像的尺寸和数量，以及卷积核等的情况，计算输出的图像的个数和尺寸
    def calOutputInfo(self):
        #生成来自上一层的模拟数据
#         print(self.outputShapeOFFormerLayer)
        kernelNumOfFormerLayer = self.outputShapeOFFormerLayer[0]
        inputImageNumFromFormerLayer = self.outputShapeOFFormerLayer[1]
        height, width = self.outputShapeOFFormerLayer[2], self.outputShapeOFFormerLayer[3]
#         print("上一层的图片尺寸是", height, width)

        imageList = np.zeros((kernelNumOfFormerLayer, inputImageNumFromFormerLayer, height, width))
        outputImageList = self.predict(imageList)
        self.outputShape= outputImageList.shape
        
    def padding(self, inputImageList):#对图像进行填充
        newImageList = []
#         print(inputImageList)
#         print('shape', inputImageList.shape)
        kernelNumOfFormerLayer, inputImageNumOfFormerLayer, inputImageHeightOfFormerLayer, inputImageWidthOfFormerLayer = \
            inputImageList.shape
        oriHeight, oriWidth = inputImageHeightOfFormerLayer, inputImageWidthOfFormerLayer
        newHeight = np.ceil((oriHeight - self.receptiveFieldSize)/self.colStride)*self.colStride + self.receptiveFieldSize
        newWidth = np.ceil((oriWidth - self.receptiveFieldSize)/self.colStride)*self.colStride + self.receptiveFieldSize
        newHeight, newWidth = int(newHeight), int(newWidth)
        newImageList = np.zeros((kernelNumOfFormerLayer, inputImageNumOfFormerLayer, newHeight, newWidth))
        newImageList[:, :, 0:oriHeight, 0: oriWidth] = inputImageList
#         for i in range(0, kernelNumOfFormerLayer, self.colStride):
#             tempImageList4ThisKernel = []
#             for j in range(0, inputImageNumOfFormerLayer, self.colStride):
#                 anImage = inputImageList[i, j, :, :]
#                 newImage = np.zeros((newHeight, newWidth))#创建一个0矩阵，尺寸与填充后的图像大小相同
# #                 print(newImage.shape, newHeight, newWidth, anImage.shape)
#                 newImage[0: oriHeight, 0: oriWidth] = anImage#把原始图像覆盖到0矩阵的中心位置
#                 tempImageList4ThisKernel.append(newImage)
#             newImageList.append(tempImageList4ThisKernel)
#         newImageList = np.array(newImageList)
        return newImageList
    #激活函数
    def relu(self, regressionRes):
        if regressionRes<0: regressionRes=0
        return regressionRes
        
    def colIt(self, newImage, oriHeight, oriWidth, weightMatrix, bias, stride):#对图像进行卷积
        outputImage = []#直接用list来接收结果，算是一种偷懒的做法；效率更高的是创建尺寸合适的np.array
        h = oriHeight-self.receptiveFieldSize
        if h==0: h = 1
        w = oriWidth-self.receptiveFieldSize
        if w==0: w = 1
        for i in range(0, h, stride):
            tempList = []
            for j in range(0, w, stride):
                pointsInField = newImage[i: i+self.receptiveFieldSize, j: j+self.receptiveFieldSize]
#                 print(i, j, pointsInField)
#                 print(i, j, weightMatrix)
#                 print(i,j, newImage.shape, oriWidth-stride)
                output = self.relu(np.sum(pointsInField*weightMatrix) + bias)
                tempList.append(output)
            outputImage.append(tempList)
        outputImage = np.array(outputImage)
        return outputImage

    def padding4Pooling(self, oriImage):
        oriHeight, oriWidth = oriImage.shape
        newHeight = np.ceil(oriHeight/self.poolingStride)*self.poolingStride
        newWidth = np.ceil(oriWidth/self.poolingStride)*self.poolingStride
#         newHeight = int((oriHeight - halfReceptiveFieldSize)/self.colStride)*self.colStride + self.colStride + 1
#         newWidth = int((oriWidth - halfReceptiveFieldSize)/self.colStride)*self.colStride + self.colStride + 1
        newHeight, newWidth  = int(newHeight), int(newWidth)
        newImage = np.zeros((newHeight, newWidth))#创建一个0矩阵，尺寸与填充后的图像大小相同
        newImage[0: oriHeight, 0: oriWidth] = oriImage#把原始图像覆盖到0矩阵的中心位置
        return newImage

    def pooling(self, anImage):#采样窗口和步长是随意设置的，需要这里自动适应
#         print("图片的形状是", anImage.shape)
        oriHeight, oriWidth = anImage.shape
        newImage = []
        padImage = self.padding4Pooling(anImage)
        newHeight, newWidth = padImage.shape
        for i in range(0, newHeight, self.poolingStride):
            tempList = []
            for j in range(0, newWidth, self.poolingStride):
                sliceOfImage = padImage[i: i+self.poolingStride, j: j+self.poolingStride]
                meanV = np.mean(sliceOfImage)
                tempList.append(meanV)
            newImage.append(tempList)
        newImage = np.array(newImage)
        return newImage

    #将一批图片的索引，分成均等分，每个卷积核一份
    def splitImageList4EachKernel(self, inputImageNumOfFormerLayer):
        # 然后让每一个卷积核扫描特定的图片，分别得到若干图片的抽象
        oriIndexList = list(range(inputImageNumOfFormerLayer))
        numOfInputImage4AKernel = int(inputImageNumOfFormerLayer / self.kernelNum)  # 求每一个卷积核需要扫描的图片个数
        imageIndexOfEachKernel = []
        if numOfInputImage4AKernel <= 1:  # 卷积核个数大于图片个数
            tempIndexList = []
            for i in range(int(self.kernelNum / inputImageNumOfFormerLayer) + 1):
                tempIndexList += oriIndexList
            for i in range(self.kernelNum):
                imageIndexOfEachKernel.append([tempIndexList[i], tempIndexList[i]])
        else:
            tempIndexList = oriIndexList + oriIndexList
            for i in range(self.kernelNum):
                imageIndexOfEachKernel.append(tempIndexList[i: i + numOfInputImage4AKernel])
        return imageIndexOfEachKernel

    def predict(self, inputImageList):
        """"cnn接收的，是上一层的输出。而上一层的输出，是各个神经元对应的对所有输入图片的扫描结果,数据的结构是[神经元个数，输入图片个数，图片高，图片宽]"""
        outputImageList = []
#         print("卷积层的输入是", inputImageList)
        #首先填充.这里为了简单，没有提供不填充的选项
#         print(inputImageList)
        newInputImageList = self.padding(inputImageList)
        kernelNumOfFormerLayer, inputImageNumOfFormerLayer, inputImageHeightOfFormerLayer, inputImageWidthOfFormerLayer = \
            inputImageList.shape
        oriHeight, oriWidth = inputImageHeightOfFormerLayer, inputImageWidthOfFormerLayer
        #为每一张图片分分配上一层传过来的图片，分配的原则是:(1)当前层的一个神经元要接收上一层的每一个神经元的输出;
        #(2)只接收上一层一个神经元扫描的到的所有图片中的若干个,从而减少参数。
        #如果有必要，可以进一步减少两层之间的连接数，从而减少参数。
        self.imageIndexOfEachKernel = self.splitImageList4EachKernel(inputImageNumOfFormerLayer)
        
        for i in range(len(self.imageIndexOfEachKernel)):#遍历每一个卷积核
            imageIndexOfThisKernel = self.imageIndexOfEachKernel[i]
            start, end = imageIndexOfThisKernel[0], imageIndexOfThisKernel[-1]
            tempImageList = []#存储当前神经元扫描各个图片得到的结果
#             print("前一层的卷积个个数是", kernelNumOfFormerLayer)
            for j in range(kernelNumOfFormerLayer):#遍历上一层的每一个卷积核
                for index in range(start, end+1):
#                     print(newInputImageList.shape, j, index)
                    anImage = newInputImageList[j, index, :, :]#取出前一层每一个神经元输出的图片中，分给当前层的这个神经元的图片
                    outputImage = self.colIt(anImage, oriHeight, oriWidth, self.weightListOfKernels[i],self.biasList[i], self.colStride)
#                     print(i, j, "卷积的结果是", outputImage)
                    outputImage = self.pooling(outputImage)
#                     print("池化之后的图片是", outputImage)
    #                 print("###################")
                    tempImageList.append(outputImage)
            outputImageList.append(tempImageList)#收集当前神经元扫描各个图片得到的结果
        outputImageList = np.array(outputImageList)
        return outputImageList
        
    def predict4Train(self, inputImageList):
        # 首先填充.这里为了简单，一律填充
        self.traningInput = inputImageList
        self.trainingColOutput = []#卷积核的输出
        self.traningPoolOutput = []#池化层的输出
        newInputImageList = self.padding(inputImageList)
        kernelNumOfFormerLayer, inputImageNumOfFormerLayer, inputImageHeightOfFormerLayer, inputImageWidthOfFormerLayer = \
            inputImageList.shape
        oriHeight, oriWidth = inputImageHeightOfFormerLayer, inputImageWidthOfFormerLayer
        #为每一张图片分分配上一层传过来的图片，分配的原则是:(1)当前层的一个神经元要接收上一层的每一个神经元的输出;
        #(2)只接收上一层一个神经元扫描的到的所有图片中的若干个,从而减少参数。
        #如果有必要，可以进一步减少两层之间的连接数，从而减少参数。
        self.imageIndexOfEachKernel = self.splitImageList4EachKernel(inputImageNumOfFormerLayer)
        #计算前一层输出的像素，与当前层的卷积层输出的对应关系
        self.kernelIndex4EachInputImage = [[] for _ in range(inputImageNumOfFormerLayer)]
        for i in range(len(self.imageIndexOfEachKernel)):#遍历当前层的卷积核
            for picIndex in self.imageIndexOfEachKernel[i]:#遍历这个卷积核连接的图片的index
                self.kernelIndex4EachInputImage[picIndex].append(i)#为这个图片收集后一层的卷积核的索引
                
            
        
        for i in range(len(self.imageIndexOfEachKernel)):#遍历每一个卷积核
            imageIndexOfThisKernel = self.imageIndexOfEachKernel[i]
            start, end = imageIndexOfThisKernel[0], imageIndexOfThisKernel[-1]
            tempColOutputList = []
            tempPoolOutputList = []#存储当前神经元扫描各个图片得到的结果
            for j in range(kernelNumOfFormerLayer):#遍历上一层的每一个卷积核
                for index in range(start, end+1):
                    anImage = newInputImageList[j, index, :, :]#取出上一层每一个神经元输出的图片中，分给当前层的这个神经元的图片
                    outputImage = self.colIt(anImage, oriHeight, oriWidth, self.weightListOfKernels[i],self.biasList[i], self.colStride)
                    tempColOutputList.append(outputImage)
                    outputImage = self.pooling(outputImage)
                    tempPoolOutputList.append(outputImage)
            self.trainingColOutput.append(tempColOutputList)
            self.traningPoolOutput.append(tempPoolOutputList)
        self.trainingColOutput = np.array(self.trainingColOutput)
        self.traningPoolOutput = np.array(self.traningPoolOutput)
        outputImageList = self.traningPoolOutput
        return outputImageList
    
    #计算所有参数的梯度，以及反向传播给前一层所有图片像素点的误差。后一层反向传播过来的误差，是本层池化层的输出误差
    def calGrad(self, errorFromLaterLayer):
        kernelNumOfFormerLayer, picNumOfFormerLayer, formerOutputHeight, formerOutputWidth = self.traningInput.shape
        kernelNumOfThisLayer, picNumOfThisLayer, outputHeight, outputWidth = self.traningPoolOutput.shape
        _, _, kernelOutputHeight, kernelOutputWidth = self.trainingColOutput.shape
        #首先基于池化层输出误差，反推计算卷积核输出像素点的误差。我们统一使用mean pooling
        kernelOutput = np.zeros(self.trainingColOutput.shape)
        for m in range(kernelOutput.shape[0]):
            for n in range(kernelOutput.shape[1]):
                for i in range(0, outputHeight, self.poolingStride):#遍历一副图像的所有像素点(不包括填充部分)
                    for j in range(0, outputWidth, self.poolingStride):
                        kernelOutput[m, n, i: i +  self.poolingStride, j: j+  self.poolingStride] = \
                                          np.ones(( self.poolingStride,  self.poolingStride))* errorFromLaterLayer[m, n, i,j]
#         print("基于池化层输出复原得到的图片是", kernelOutput)
        
        #接下来，基于池化层的输入，也就是卷积层的输出，来计算反向传播到前一层的误差
        self.error2FormerLayer = np.zeros(self.traningInput.shape)
        for i in range(kernelNumOfFormerLayer):
            for j in range(picNumOfFormerLayer):
                indexList = self.kernelIndex4EachInputImage[j]#这个图片对应的后一层的卷积核索引
                for m in range(0, formerOutputHeight - self.receptiveFieldSize, self.colStride):
                    for n in range(0, formerOutputWidth - self.receptiveFieldSize, self.colStride):
                        for laterKernelIndex in indexList:
                            outputOfThisKernel = kernelOutput[laterKernelIndex]
#                             print(outputOfThisKernel.shape, j, m, n)
#                             print(m/self.colStride)
                            self.error2FormerLayer[i, j , m, n] += \
                                self.traningInput[i, j, m, n]* outputOfThisKernel[j, \
                                                      int(m/self.colStride), int(n/self.colStride)]
                                
        #计算这层卷积核的参数的梯度               
        self.grad = np.zeros(self.weightListOfKernels.shape)
        for m in range(kernelNumOfThisLayer):
#             print(kernelNumOfThisLayer, self.weightListOfKernels.shape)
            weights = self.weightListOfKernels[m]
            indexList = self.imageIndexOfEachKernel[m]#这个图片对应的后一层的卷积核索引
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    tempGrad = 0
                    #遍历输入图像，把这个权重的梯度算出来
                    for dim2 in range(errorFromLaterLayer[m].shape[0]):#遍历这个卷积核输出的所有图片
                        for dim3 in range(0, errorFromLaterLayer[m].shape[1]):
                            for dim4 in range(0, errorFromLaterLayer[m].shape[2]):
#                                 print('asdasdaqweqw', errorFromLaterLayer[m].shape)
#                                 print(m, dim2, dim3, dim4)
                                tempGrad += weights[i, j] * errorFromLaterLayer[m, dim2, dim3, dim4]
                    self.grad[m,i,j] = tempGrad
#         print("计算得到的梯度是", self.grad)
    def updateWeights(self):
        # print("CNN", self.__dict__.keys())
        # print("本次更新参数使用 的梯度是",self.grad * self.learningRate )
        # print("当前的参数是", self.weightListOfKernels)
        self.weightListOfKernels -= self.grad * self.learningRate

    def updateWeights4Multi(self, grad):
        # print("CNN", self.__dict__.keys())
        # print("本次更新参数使用 的梯度是",self.grad * self.learningRate )
        # print("当前的参数是", self.weightListOfKernels)
        self.weightListOfKernels -= grad * self.learningRate                           
                        
                        
                
                
        