#一个线性链隐马尔可夫模型
#基于已经标注了分词标记的数据，训练一个分词器
import copy

class LinearChainHMM():
    
    def __init__(self):
        #定义隐马尔可夫模型的参数
        self.initStatProbDist = {}#隐藏状态的初始取值的概率分布
        self.statTransProbMap = {}#隐藏状态之间的转移概率矩阵,这里为了简单，理解起来直观，使用map来存储
        self.charProbDistOfEachStat = {}#存储各个隐藏状态下，出现字符的概率分布;每个分布用map来存储
        self.statValueSet = set({})#用来存储隐藏状态的取值水平
    
    #基于训练语料，估计隐马尔可夫模型的3部分参数
    def fit(self, sentenceList):
        sentenceNum = len(sentenceList)
        fistStatINstatTransNum = {}#语料中出现的隐藏状态转换中，第一个状态的频数用来计算这个状态跳转到其他状态的概率分布
        statNumMap = {}#语料中各个隐藏状态出现的次数
        for sentence in sentenceList:
            charsStr, tagsStr = sentence[0], sentence[1]
            
            #统计初始隐藏状态的频数
            initStat = tagsStr[0]#取出第1个字符的隐藏状态取值
            self.initStatProbDist[initStat] = self.initStatProbDist.get(initStat, 0) + 1.
            
            #统计隐藏状态之间转换情况的频数
            for i in range(1, len(tagsStr)):
                firstStat = tagsStr[i]
                fistStat_secondStat = tagsStr[i-1:i+1]#取出连续的两个分词标记,表示从第一个状态转移到第二个状态
                fistStatINstatTransNum[firstStat] = fistStatINstatTransNum.get(firstStat, 0) + 1#发生了一次状态转换
                self.statTransProbMap[fistStat_secondStat] = \
                     self.statTransProbMap.get(fistStat_secondStat, 0) + 1.
                     
            #统计各个隐藏状态下，也就是词性标记下，每个字符出现的频数
            for i in range(0, len(tagsStr)):
                char = charsStr[i]
                tag = tagsStr[i]
                self.statValueSet.add(tag)
                if tag not in self.charProbDistOfEachStat:#如果这个隐藏状态没有收录，就添加上
                    self.charProbDistOfEachStat[tag] = {}
                statNumMap[tag] = statNumMap.get(tag, 0) + 1#统计这个状态取值的个数，后面用来计算字符的概率分布
                self.charProbDistOfEachStat[tag][char] = self.charProbDistOfEachStat[tag].get(char, 0) + 1
                
        #给每一个频数，除上对应的分母，形成概率估计值
        for stat in self.initStatProbDist:#计算各个隐藏状态作为初始状态的概率
            self.initStatProbDist[stat] /= sentenceNum
            
        for firstStat_secondStat in self.statTransProbMap:#计算一个隐藏状态转移到其他状态的概率
            firstStat = firstStat_secondStat[0]
            self.statTransProbMap[firstStat_secondStat] /= fistStatINstatTransNum.get(firstStat, 1)
            
        for tag in self.charProbDistOfEachStat:#遍历每一个词性，也就是隐藏状态
            charProbDist4ThisTag = self.charProbDistOfEachStat[tag]
            for char in charProbDist4ThisTag:#遍历每一个字符，也就是观测值
                charProbDist4ThisTag[char] /= statNumMap[tag]#这个词性下一个字符的出现次数，除以这个词性出现的总次数
                #就是这个词语在这个状态下出现的概率估计值
            
        print("初始状态概率分布" , self.initStatProbDist)
        print("状态转移概率分布" , self.statTransProbMap)
        print("字符出现的条件概率", self.charProbDistOfEachStat)
            
    #基于观测值序列，也就是语句话的字符串列表，使用模型选出最好的隐藏状态序列，并按照分词标记将字符聚合成分词结果
    def predict(self, text): 
        statPathProbMap = {}#存储以各个初始状态打头的概率最大stat路径
        for stat in self.initStatProbDist:#遍历每一个初始状态
            statPath = stat#这是目前积累到的stat路径，也就是分词标记序列
            firstChar = text[0]
            conditionProbOfThisChar = self.charProbDistOfEachStat[stat].get(firstChar, 0.0000001)
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
                           self.charProbDistOfEachStat[statValue].get(char, 0.0000001)
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
    sentenceList = loadData(fileName, sentenceNum=1000)#加载语料

    model = LinearChainHMM()
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
   
    
    
    
    
    
    