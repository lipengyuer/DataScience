from pyhanlp import HanLP
from analysis.algorithm import splitSentence
#获取文本的分词结果和词性标注结果
def wordSeg(text):
    wordPostag = HanLP.segment(text)
    words, postags = [], []
    for line in wordPostag:
        line = str(line)
        word, postag = line.split('/')
        words.append(word)
        postags.append(postag)
    return words, postags

def sentenceWordPostag(textList):
    sentencesList = list(map(lambda x: splitSentence.getSentences(x), textList))
    wordsList, postagsList = [], []
    for sentences in sentencesList:
        wordsListTemp, postagsListTemp = [], []
        for sentence in sentences:
            words, postags = wordSeg(sentence)
            wordsListTemp.append(words)
            postagsListTemp.append(postags)
        wordsList.append(wordsListTemp)
        postagsList.append(postagsListTemp)
    return sentencesList, wordsList, postagsList
