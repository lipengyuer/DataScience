#endocing:utf-8
def countWordFreq(words):
    wordFreqMap = {}
    for word in words:
        wordFreqMap[word] = wordFreqMap.get(word, 0) + 1
    return wordFreqMap

def getNGrams(line, N=2):
    res = []
    for i in range(len(line)-N):
        res.append(line[i:i+N])
    return res


def readFileAndCount(fileName, label):
    resList = []
    with open(fileName, 'r') as f:
        line = f.readline()
        # print(line)
        while line!='':

            line = line.split('kabukabu')[-1].split(' ')
            line = getNGrams(''.join(line))
            if len(line)< 3:
                line = f.readline()
                continue

            wordFreqMap = countWordFreq(line)
            resList.append({'sentiment': label, 'wordFreq': list(wordFreqMap.items())})
            line = f.readline()
    return resList
import random

def processJingDong():
    negCommentFile = r'negComments.txt'
    posCommentFile = r'posComments.txt'
    wordFreqMapList1 = readFileAndCount(negCommentFile, 1)
    l1 = len(wordFreqMapList1)
    wordFreqMapList2 = readFileAndCount(posCommentFile, 0)
    l2 = len(wordFreqMapList1)
    l = min(l1, l2)
    wordFreqMapList  = wordFreqMapList1[:l] + wordFreqMapList2[:l]
    random.shuffle(wordFreqMapList)
    with open('JingDongCommentsWordFreq.pkl', 'wb') as f:
        pickle.dump(wordFreqMapList, f)

from pyhanlp import HanLP
def wordSeg(text):
    text = text.replace('\n', '').replace('\r', '').replace(' ', '')
    wordPostag = HanLP.segment(text)
    words, postags = [], []
    for line in wordPostag:
        line = str(line)
        res = line.split('/')
        if len(res)!=2:
            continue
        word, postag = line.split('/')
        if len(word) < 3:
            continue
        words.append(word)
        postags.append(postag)
    return words, postags

def readFileAndCountWeibo(fileName, label):
    resList = []
    with open(fileName, 'r') as f:
        line = f.readline()
        # print(line)
        while line!='':

            line = line.split('kabukabu')[-1]
            line = ''.join(line.split(' '))
            line = getNGrams(line)
            # line, postags = wordSeg(line)
            if len(line)==1:
                line = f.readline()
                continue
            wordFreqMap = countWordFreq(line)
            resList.append({'sentiment': label, 'wordFreq': list(wordFreqMap.items())})
            line = f.readline()
    return resList

def processWeibo():
    negCommentFile = r'negative'
    wordFreqMapList1 = readFileAndCountWeibo(negCommentFile, 1)
    l1 = len(wordFreqMapList1)
    posCommentFile = r'positive'
    wordFreqMapList2 = readFileAndCountWeibo(posCommentFile, 0)
    l2 = len(wordFreqMapList1)
    l = min(l1, l2)
    print(l, l1, l2)
    wordFreqMapList  = wordFreqMapList1[:l] + wordFreqMapList2[:l]
    print(len(wordFreqMapList))
    print(list(map(lambda x: x['sentiment'], wordFreqMapList) )[:10])
    random.shuffle(wordFreqMapList)
    with open('WeiboWordFreq.pkl', 'wb') as f:
        pickle.dump(wordFreqMapList, f)

import pickle
if __name__ == '__main__':
    processJingDong()
    processWeibo()
