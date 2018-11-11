# -*- coding:utf-8 -*-
# import scipy
import matplotlib.pyplot as plt
import numpy as np
import pylab
import sys
import os

def showConfusionMatrix(myres, realLabels, classNum):
    predLabel = myres
    realLabel = realLabels

    confusionMatrix = np.zeros((classNum, classNum))
    for i in range(len(predLabel)):
        n = predLabel[i]
        m = realLabel[i]
        confusionMatrix[m - 1, n - 1] += 1
    for i in range(len(confusionMatrix)):
        a = list(confusionMatrix[i])
        print(' '.join(map(lambda x: str(int(x)), a)))