#把原始语料处理成crfpp要求的格式。
#将差距不大的若干种语料融合起来，形成一份较大的语料。
from pyhanlp import *

from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
import pyhanlp

def readTextLines(fileName):
    with open(fileName, 'r', encoding='utf8') as f:
        lines = f.readlines()
    return lines

def character_tagging(line):
    res = ""
    word_list = line.strip().split()
    for word in word_list:
        if len(word) == 1:
            res += word + "\tS\n"
        else:
            res += word[0] + "\tB\n"
            for w in word[1:len(word)-1]:
                res += w + "\tM\n"
            res += word[len(word)-1] + "\tE\n"
    res += "\n"
    return res

def addline(line, fileName):
   with open(fileName, 'a+') as f:
       f.write(line)

def processFilesInDir():
    dirName = 'corpus/'
    trainCorpusFile = 'corpus4WordSeg.txt'
    files = os.listdir(dirName)
    count = 0
    for fileName in files:
        fileName = dirName + fileName
        lines = readTextLines(fileName)
        if fileName in ['as_training.utf8', 'cityu_training.utf8']:
            lines = list(map(lambda x: HanLP.convertToSimplifiedChinese(x), lines))
        for line in lines:
            count += 1
            if count %1000==0:
                print("正在处理第", count, "句话。")
                return
            line = line.split('。')
            for subLine in line[:-1]:
                line = character_tagging(subLine + '。')
                addline(line, trainCorpusFile)

            subLine = character_tagging(line[-1])
            if len(subLine)>10:
                addline(subLine, trainCorpusFile)


if __name__ == '__main__':
    processFilesInDir()


