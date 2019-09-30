'''
Created on 2019年9月30日

@author: lipy
'''
#最早的主题模型：LSA
import numpy as np
from pyhanlp import HanLP
from collections import Counter

#适用于小规模语料的LSA
class LSA():
    
    def __init__(self, topic_num=2):
        self.word_id_map = {}#词语序号映射
        self.id_word_map = {}#序号到词语的映射
        self.vocab_size = None
        self.topic_num = topic_num
            
    def segment(self, text):
        word_tag_list = HanLP.segment(text)
        word_list = []
        for word_tag in word_tag_list:
            word, tag = str(word_tag).split('/')
            if tag=='n':
                word_list.append(word)
        return word_list
    
    def word_count(self, word_list):
        word_freq_map = Counter(word_list)
        return word_freq_map

    #遍历语料，得到词汇表，构建特征
    def get_vocab(self, document_list):
        words_list = []
        for doc in document_list:
            words = self.segment(doc)
            words_list.append(words)
            for word in words:
                if word not in self.word_id_map:
                    self.id_word_map[len(self.word_id_map)] = word
                    self.word_id_map[word] = len(self.word_id_map)
                    
                
        self.vocab_size = len(self.word_id_map)
        return words_list
    

    def get_term_freq_matrix(self, words_list):
        term_freq_maxtrix = np.zeros((len(words_list), self.vocab_size))
        for i in range(len(words_list)):
            words = words_list[i]
            term_freq_map = Counter(words)
            for word in words:
                term_freq_maxtrix[i, self.word_id_map[word]] = term_freq_map[word]#这里默认所有的词语都收录到词汇表里了；实际应用的时候，需要考虑OOV
        return term_freq_maxtrix
                
    #基于语料训练模型
    def fit(self, document_list):
        #获取词汇表,以及文档的分词结果
        words_list = self.get_vocab(document_list)
        print('词汇表的大小是', len(self.word_id_map))
        #构建每篇文档的term freq向量
        term_freq_matrix = self.get_term_freq_matrix(words_list)
        print("文档的词频统计结果是一个矩阵，形状是：", term_freq_matrix.shape)
        #矩阵分解
        U, S, V = np.linalg.svd(term_freq_matrix)

        #整理矩阵分解结果
        self.U = U[:, 0:self.topic_num]
        self.S = np.zeros((self.topic_num, self.topic_num))
        for i in range(self.topic_num): self.S[i, i] = S[i]
        self.V = V[0: self.topic_num, :]
  
        self.have_a_look()
        
    
    #看一下模型的效果
    def have_a_look(self):
        #展示主题和词语的对应关系
        word_list = []
        for i in range(self.V.shape[1]): word_list.append(self.id_word_map[i])
        self.print_list(word_list)
        for i in range(self.V.shape[0]):
            self.print_list(self.V[i, :])
            
        #打印每篇文档的主题分布
        print(self.U)
     
    #把一个列表的元素，整齐的打印成一行
    def print_list(self, element_list):
        p = "{0:{" + str(len(element_list)) + "}^10}"#{0:^6}\t{1:{3}^10}\t{2:^6}
        for i in range(1, len(element_list)):
            p += "\t{" + str(i) + ":{" + str(len(element_list)) + "}^10}"
#         print(p)
#         print(element_list)
        es = tuple(list(map(lambda x:str(x)[:10], element_list)) + [chr(12288)])
        print(p.format(*es))
            
            
document_list = ["明天就是国庆节了，祝愿祖国永远繁荣昌盛。",
          "明天就是国庆节了，可以画家看孩子，真开心。",
          "明天可以看国庆大阅兵，真开心。",
          "国庆大阅兵的时候，我在火车上，信号不好，完犊子。",
          "看回放也是极好的，NBA决赛的回放仍然刺激。",
          "国庆大阅兵中，我们可以看到好多新装备。",
          "我的孩子是个调皮鬼，她自己也承认。"]

if __name__ == '__main__':
    topic_model = LSA()
    topic_model.fit(document_list)
    
    