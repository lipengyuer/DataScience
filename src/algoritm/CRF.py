#条件随机场
#实现CRF的训练和使用，语料的预处理，标注结果的评估

import copy

class CRF():
    
    def __init__(self):
        #定义CRF的参数
        pass
    
    #基于训练语料，估计CRF参数
    def fit(self, sentenceList):
        pass
            
    #基于观测值序列，也就是语句话的字符串列表，使用模型选出最好的隐藏状态序列，并按照分词标记将字符聚合成分词结果
    def predict(self, text): 
        statPathProbMap = {}#存储以各个初始状态打头的概率最大stat路径
        for stat in self.initStatProbDist:#遍历每一个初始状态
            statPath = stat#这是目前积累到的stat路径，也就是分词标记序列
            firstChar = text[0]
            conditionProbOfThisChar = self.charProbDistOfEachStat[stat].get(firstChar, 0.000001)
            statPathProb = self.initStatProbDist[stat] * conditionProbOfThisChar
            statPathProbMapOfThis = {}
            statPathProbMapOfThis[statPath] = statPathProb
            for i in range(1, len(text)):
                char  = text[i]
                tempPathProbMap = {}
                for statValue in self.statValueSet:
                    tempStatPath = statPath + statValue
                    statTrans = statPath[-1] + statValue
                    tempPathProb = statPathProbMapOfThis[statPath]* \
                           self.statTransProbMap.get(statTrans, 0.01)*\
                           self.charProbDistOfEachStat[statValue].get(char, 0.000001)
                    tempPathProbMap[tempStatPath] = tempPathProb
                bestPath = getKeyWithMaxValueInMap(tempPathProbMap)
                statPathProbMapOfThis[bestPath] = tempPathProbMap[bestPath]
                statPath = bestPath
            statPathProbMap[statPath] = statPathProbMapOfThis[statPath]
        bestPath = getKeyWithMaxValueInMap(statPathProbMap)
        res = mergeCharsInOneWord(text, bestPath)
        return res
                    

def getKeyWithMaxValueInMap(dataMap):
    dataList = sorted(dataMap.items(), key=lambda x: x[1], reverse=True)
    theKey = dataList[0][0]
    return theKey
    
#基于分词标记把字符聚合起来，形成分词结果
def mergeCharsInOneWord(charList, tagList):
    wordList = []
    word = ''
    for i in range(len(charList)):
        tag, char = tagList[i], charList[i]
        if tag=='E':
            word += char
            wordList.append(word)
            word = ''
        elif tag=="S":
            word += char
            wordList.append(word)
            word = ''
        else:
            word += char
    return wordList
                
    
dataStr = """
我 S
喜 B
欢 E
吃 S
好 B
吃 M
的 E
， W
因 B
为 E
这 B
些 E
东 B
西 E
好 B
吃 E
。
"""

def loadData(fileName, sentenceNum = 100):
    with open(fileName, 'r') as f:
        line = f.readline()
        corpus = []
        tempSentence = []
        tempTag = []
        count = 0
        while line!=True:
            line = line.replace('\n', '')
            if line=='':#如果这一行没有字符，说明到了句子的末尾
                tempSentence = ''.join(tempSentence)#把字符都连接起来形成字符串，后面操作的时候会快一些
#                 if "习近平" in tempSentence:
#                     print(tempSentence)
                tempTag = ''.join(tempTag)
                corpus.append([tempSentence,tempTag])
#                 print("这句话是", [tempSentence,tempTag])
                tempSentence = []
                tempTag = []
                count += 1
                if count==sentenceNum:#如果积累的句子个数达到阈值，返回语料
                    return corpus
            else:
                line= line.split('\t')
#                 print(line)
                [char, tag] = line[0], line[2]#取出语料的文字和分词标记
                tempSentence.append(char)
                tempTag.append(tag)
            line = f.readline()
    return corpus
                      
import time
if __name__ == '__main__':
    fileName = "/home/pyli/eclipse-workspace/DataScience/src/algoritm/msra_training.txt"
    sentenceList = loadData(fileName, sentenceNum=5000)#加载语料

    model = CRF()
    model.fit(sentenceList)
    res = model.predict(sentenceList[100][0])
    print("分词结果是", res, "真实的分词结果是", )
    
    s = "我是一个粉刷将，粉刷本领强。我要把我的新房子刷的很漂亮。"
    res = model.predict(s)
    testS = ['我是一个粉刷将，粉刷本领强。', '我要把我的新房子刷的很漂亮。',
              '我是一个粉刷将，粉刷本领强。我要把我的新房子刷的很漂亮。',
              '习近平指出，当前我国社会的主要矛盾仍然是人民日益增长的物质需求与不发达的生产力之间的矛盾。']
    for s in testS:
        t1 = time.time()
        res = model.predict(s)
        t2 = time.time()
        print(t2-t1, res)
   
    
    
    
    
    
    