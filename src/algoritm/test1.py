
import pickle
import numpy as np
from Softmax import Softmax4CNN
import pickle
import numpy as np
import random, copy
from CNN import CNN
import time

class CNNSoftmax():
    def __init__(self, picShape, classNum, epochNum=1, learningRate = 0.001,  workerNum = 3):
        self.workerNum = workerNum
        self.classNum = classNum
        self.layers = []  # 用于存储各层参数
        self.picShape = picShape  # 初始化的时候，需要手动输入图片的高和宽，用来推测后面的一系列参数
        self.epochNum = epochNum
        self.learningRate = learningRate
        self.createNetwork()

    def createNetwork(self):  # 可以在
        print("开始初始化网络")
        cnnLayer1 = CNN([1, 1, self.picShape[0], self.picShape[1]],
                        kernelNum=10, colStride=2, receptiveFieldSize=3, \
                        poolingSize=2, poolingStride=2, learningRate=self.learningRate)
        print("第一个卷积层的输出形状是", cnnLayer1.outputShape)

        cnnLayer2 = CNN(cnnLayer1.outputShape, 
                        kernelNum=5, colStride=1, receptiveFieldSize=3, \
                        poolingSize=2, poolingStride=2, learningRate=self.learningRate)
    
        print("第二个卷积层的输出形状是", cnnLayer2.outputShape)
        kernelNum, picNum, height, width  = cnnLayer2.outputShape
        featureNumOfSoftMax = kernelNum*picNum*height*width
#         print("softmax的特征数是", featureNumOfSoftMax)
        softmaxLayer = Softmax4CNN(featureNumOfSoftMax, self.classNum, learningRate=self.learningRate)
        self.layers = [cnnLayer1, cnnLayer2, softmaxLayer]

    def shuffleData(self, inputList, outputList, rate = 0.2):
        indexList = list(range(len(inputList)))
        random.shuffle(indexList)
        indexList = indexList[:int(rate * len(indexList))]
        resInput, resOutput = inputList[indexList], outputList[indexList]
        return resInput, resOutput

    def calGrad(self, anImage, realLabel):  # 梯度计算
        # self.gradList = [None for _ in range(len(self.layers))]  # 每一层的梯度数据
        outputVector = self.predict4Train(anImage)#执行一次前向过程，记录每一层的输入和输出
        #softmax层的梯度单独计算
        self.layers[-1].calGrad(realLabel)
#         print("softmax层的梯度是", self.gradList[-1])
        for i in range(len(self.layers)-2, -1, -1):#从后向前遍历各层
            thisLayer = self.layers[i]#本层神经元
            laterLayer = self.layers[i+1]#后一层神经元
#             print("正在计算第", i+1, '层的梯度')
            thisLayer.calGrad(laterLayer.error2FormerLayer)
#             print("计算的到的梯度是", thisLayer.grad)
                    
                    
    # 更新参数
    def updateWeights(self):
        for i in range(len(self.layers)):
            # print("正在更新第", i, '层的参数')
            self.layers[i].updateWeights()

    # 使用BP算法，训练模型
    def fit(self, trainingImageList, trainingLabelList):
        trainingImageList, trainingLabelList = np.array(trainingImageList), np.array(trainingLabelList)
        for epoch in range(self.epochNum):
            trainingImageList, trainingLabelList = self.shuffleData(trainingImageList, trainingLabelList, rate=1)
            for i in range(len(trainingImageList)):
                anImage, realLabel = trainingImageList[i], trainingLabelList[i]
                anImage = anImage.reshape((1, 1, anImage.shape[0], anImage.shape[1]))#CNN的输入，第一维对应的是上一层的卷积核，
#                 print("anImage", anImage, anImage.shape)
                #第二维对应的是上一层每一个卷积核对应的输出图片个数。原始图片可以假装来自只有一个卷积核的层，输出图片也只有一个
                self.calGrad(anImage, realLabel)
#                 print("本次使用的梯度是", self.layers[0].grad)
                # print(self.layers[0].__dict__.keys())
                self.updateWeights()
#                 if i % 500 == 0:
# #                     print('asdasdasd', trainingImageList)
#                     cost = self.calCost(trainingImageList, trainingLabelList)
#                     print("已经学习了", epoch, '轮, cost为', cost,\
#                           '本轮的进度是',  i, '/', len(trainingImageList))
            cost = self.calCost(trainingImageList, trainingLabelList)
            print("完成了本轮的训练", epoch, '轮, cost为', cost)
    
    # 更新参数
    def updateWeights4Multi(self, gradList):
        for i in range(len(self.layers)):
            # print("正在更新第", i, '层的参数')
            self.layers[i].updateWeights4Multi(gradList[i])
            
    def fit_multi(self, trainingImageList, trainingLabelList):
        print("开始训练")
        trainingImageListOri, trainingLabelListOri = trainingImageList, trainingLabelList
        from multiprocessing import Pool
        batchSzie = 20
        sliceSize = 100
        initLearningRate = self.learningRate
        print("完成数据准备")
        check_point = 0
        
        for epoch in range(self.epochNum):
            t1 = time.time()
            trainingImageListOri, trainingLabelListOri = \
                                 self.shuffleData(trainingImageListOri, trainingLabelListOri, rate=1)
            for m in range(0, trainingImageListOri.shape[0], sliceSize):
                
                trainingImageList, trainingLabelList = trainingImageListOri[m:m+sliceSize], trainingLabelListOri[m:m+sliceSize]
    #                                    self.shuffleData(trainingImageListOri, trainingLabelListOri, rate=0.1)
                sampleSize = trainingImageListOri.shape[0]
    
                pool = Pool(self.workerNum)
                resList = []
                for i in range(0, len(trainingImageList), batchSzie):
                    trainingImageBatch, trainingLabelBatch = trainingImageList[i: i + batchSzie], trainingLabelList[i:i+batchSzie]
                    res = pool.apply_async(calGrad4Multi, args = (copy.deepcopy(self), trainingImageBatch, trainingLabelBatch))
                    resList.append(res)
                pool.close()
                pool.join()
                gradData = [1 for _ in range(len(self.layers))]
                for res in resList[0:]:
                    gradDataListTemp = res.get()
                    for i in range(len(self.layers)):
    #                     print("各层的对象名是", gradDataListTemp.layers[i])
                        if type(gradData[i])==int: 
                            gradData[i] = gradDataListTemp.layers[i].grad/sampleSize
                        else: 
    #                         print(i, gradData[i].shape, gradDataListTemp.layers[i].grad.shape)
                            gradData[i] += gradDataListTemp.layers[i].grad/sampleSize
                        
    #             print("梯度是", gradData)
                self.learningRate = initLearningRate * ( 1/(1 + np.exp(-0.5 * (1 - epoch**0.5))))
#                 self.learningRate = initLearningRate/(epoch + 1)#np.sqrt(epoch)
                self.updateWeights4Multi(gradData)
                check_point += 1
                if m % 200 == 0:
                    cost = self.calCost(trainingImageList, trainingLabelBatch)
                    print('cost为', cost)
            t2 = time.time()
            if epoch%1==0:
                cost = self.calCost(trainingImageList, trainingLabelBatch)
                print("完成了本轮的训练", epoch, '轮, 耗时是', int(t2-t1) , 'cost为', cost)
#             if check_point%100==0:
#                 import pickle
#                 pickle.dump(self, open('cnnsoftmax.pkl', 'wb'))
            
            
    def predict(self, imageList):
        for layer in self.layers:
            imageList = layer.predict(imageList)
        #            print(anImage)
        return imageList

    #训练过程中需要使用的预测函数，会把各层的输出保留起来，用于计算梯度和误差
    def predict4Train(self, anImage):
        for layer in self.layers:
            anImage = layer.predict4Train(anImage)
#         print("训练输出是", anImage)
        return anImage
                            

    # 计算损失值
    def calCost(self, trainingImageList, trainingLabelList):
        cost = 0
        for j in range(min([20, trainingImageList.shape[0]])):
            anImage = trainingImageList[j].reshape((1, 1,  trainingImageList[j].shape[0],  trainingImageList[j].shape[1]))#CNN的输入，第一维对应的是上一层的卷积核，
            label = trainingLabelList[j]
            predLabel = self.predict4Train(anImage)
            for i in range(self.classNum): cost -= label[i] * np.log(predLabel[i] + 0.0000001)
        return cost

def calGrad4Multi(self, trainingImageList, trainingLabelList):
    for i in range(trainingImageList.shape[0]):
        anImage, realLabel = trainingImageList[i], trainingLabelList[i]
        anImage = anImage.reshape((1, 1, anImage.shape[0], anImage.shape[1]))#CNN的输入，第一维对应的是上一层的卷积核，
#                 print("anImage", anImage, anImage.shape)
        #第二维对应的是上一层每一个卷积核对应的输出图片个数。原始图片可以假装来自只有一个卷积核的层，输出图片也只有一个
        self.calGrad(anImage, realLabel)
    return self


def test2():
    from tensorflow.examples.tutorials.mnist import input_data
    from sklearn.model_selection import train_test_split
    mnist = input_data.read_data_sets('../../data/mnist', one_hot=True)

    inputData = mnist.test.images[:, :].reshape((-1, 28, 28))
    outputData = mnist.test.labels[:, :]
    trainingInput, testInput, traingOutput, testOutput = \
        train_test_split(inputData, outputData, test_size=0.8)
    print(trainingInput.shape)
    with open('cnnsoftmax.pkl','rb') as f:
        clf = pickle.load(f)
#     clf.fit(trainingInput, traingOutput)
#     clf.fit_multi(trainingInput, traingOutput)
    accuracy = [0,0]
#     testInput, testOutput = trainingInput, traingOutput
    for i in range(min(1000, testInput.shape[0])):
        testInputList = testInput[i].reshape((1, 1, inputData.shape[1],inputData.shape[2]))
        pred = clf.predict(testInputList)
        if np.argmax(pred)==np.argmax(testOutput[i]):
            accuracy[0] += 1
        accuracy[1] += 1
        print(accuracy, np.argmax(pred), np.argmax(testOutput[i]))
        
        
if __name__ == '__main__':
    test2()
    




