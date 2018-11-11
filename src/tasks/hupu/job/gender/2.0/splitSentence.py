'''
Created on Nov 14, 2016

@author: foxbat
'''
endSet = {'。', '？', '！', '”'}
notEndSet = {'、', '；', '，'}


def loadModel():
    pass


def paragraph2Sentence(paragraph, paragraphIndex, paragraphCount):
    sentences = []
    position = []

    start = 0
    notEndMarker = 0
    pointer = 0
    length = len(paragraph)
    while pointer < length:
        char = paragraph[pointer]
        if char not in endSet:  # 如果该字不是结尾字段,鲁新新修改处
            notEndMarker = pointer  # 非结尾标记 = 当前处
        # 如果距离开始处距离为99，则sentence.content为从该句开始到上一个截止符
        if pointer - start == 99:
            sentence = paragraph[start:notEndMarker + 1].strip()
            sentences.append(sentence)
            start = notEndMarker + 1
        if char in endSet:
            if pointer < length - 1:
                if paragraph[pointer + 1] in endSet:  # 如果分句符之后还是分句符，则pass
                    pass
                else:  # 否则，句子内容为从第二句开始到结束
                    sentence = paragraph[start:pointer + 1].strip()
                    sentences.append(sentence)
                    start = pointer + 1
        if pointer == length - 1:  # 如果是段落最后一个符号则分。
            sentence = paragraph[start:pointer + 1].strip()
            sentences.append(sentence)
        pointer += 1

    sentenceCount = len(sentences)
    for i in range(sentenceCount):
        position.append(
            [paragraphIndex + 1, paragraphCount, i + 1, sentenceCount])
        # sentences[i].id = '-'.join(str(p) for p in sentences[i].position)
    return sentences, position


def content2Sentece(content):
    sentences = []
    position = []
    paragraphs = content.split("\n")
    paragraphCount = len(paragraphs)
    for i in range(paragraphCount):
        paragraphSentences, paragraphPosition = paragraph2Sentence(
            paragraphs[i], i, paragraphCount)
        sentences.extend(paragraphSentences)
        position.extend(paragraphPosition)
    return sentences, position


def getSentences(content):
    sentences = []
    if len(content) > 0:
        contentSentences, contentPosition = content2Sentece(content)
        sentences = contentSentences
    return sentences
