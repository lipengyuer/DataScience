from pyhanlp import HanLP

#用于存储图
class Graph():
    def __init__(self):
        self.linked_node_map = {}#邻接表，
        self.PR_map ={}#存储每个节点的入度
        self.stop_words = set({'我'})
        
    def clean(self):
        self.linked_node_map = {}#邻接表，
        self.PR_map ={}#存储每个节点的入度        
    
    #添加节点
    def add_node(self, node_id):
        if node_id not in self.linked_node_map:
            self.linked_node_map[node_id] = set({})
            self.PR_map[node_id] = 0
        else:
            print("这个节点已经存在")
    
    #增加一个从Node1指向node2的边。允许添加新节点
    def add_link(self, node1, node2):
        if node1 not in self.linked_node_map:
            self.add_node(node1)
        if node2 not in self.linked_node_map:
            self.add_node(node2)
        self.linked_node_map[node1].add(node2)#为node1添加一个邻接节点，表示ndoe2引用了node1
    
    #计算pr
    def get_PR(self, epoch_num=5, d=0.8):#配置迭代轮数，以及阻尼系数
        for i in range(epoch_num):
            for node in self.PR_map:#遍历每一个节点
                self.PR_map[node] = (1-d) + d*sum([self.PR_map[temp_node] for temp_node in self.linked_node_map[node]])#原始版公式
#             print(self.PR_map)
    
    def get_topN(self, top_n):
        topN = sorted(self.PR_map.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return topN
    
    
class TextRankKeyword(Graph):
    
    def segment(self, text):
        word_tag_list = HanLP.segment(text)
        word_tag_list = list(map(lambda x: str(x).split('/'), word_tag_list))
        return word_tag_list
    #对文本分词和磁性标注，取名词，删除停用词，得到候选关键词，并构建词语关系
    def get_word_links(self, text, window_size=5):
        word_tag_list = self.segment(text)
        word_keyword_1 = list(map(lambda x: [x[0], True] if 'n'==x[1] and x[0] not in self.stop_words else [x[0], False], word_tag_list))
        word_link_list = []
        for i in range(len(word_keyword_1)):
            current_word, current_flag = word_keyword_1[i][0], word_keyword_1[i][1]
            if current_flag:
                for j in range(i+1, min(i+1 + window_size, len(word_keyword_1))):
                    temp_word, temp_flag = word_keyword_1[j][0], word_keyword_1[j][1]
                    if temp_flag and current_word!=temp_word:
                        word_link_list.append(tuple(sorted([current_word, temp_word])))
        return word_link_list
    
    def get_keyword_with_textrank(self, text):
        self.clean()
        word_links = self.get_word_links(text)
        for word_pair in word_links:
            self.add_link(word_pair[1], word_pair[0])
        self.get_PR()
        keyword_weight_list = self.get_topN(10)
        return keyword_weight_list

class TextRankSummary(Graph):
    
    #切分句子。simple模式：基于句号分割
    def text2words(self, text):
        sentences = text.replace('\n', '').replace(' ', '').split("。")
        sentences = list(filter(lambda x: len(x)>1, sentences))
        words_list = list(map(lambda x: self.word_segment(x), sentences))
        new_sentences, new_words_list = [], []
        for i in range(len(sentences)):
            if len(words_list[i])>0:
                new_sentences.append(sentences[i])
                new_words_list.append(words_list[i])
        return new_sentences, new_words_list
    
    def word_segment(self, sentence):
        word_tag_list = HanLP.segment(sentence)
        words = []
        for word_tag in word_tag_list:
            word_tag = str(word_tag).split('/')
            if len(word_tag)==2:
                word, tag = word_tag
                if 'n'==tag and word not in self.stop_words:
                    words.append(word)
        return set(words)
            
    #基于字，计算杰卡德相似度
    def sentence_simlarity(self, words1, words2):
        word_set1, word_set2 = set(words1), set(words2)
        simlarity = len(word_set1 & word_set2)/len(word_set2 | word_set2)
        return simlarity
        
    def get_sentence_links(self, sentences):
        
        sentence_link_list = []
        for s_id in range(len(sentences)):
            for s_jd in range(1, len(sentences)):
                if self.sentence_simlarity(sentences[s_id], sentences[s_jd])>0.5:
                    sentence_link_list.append([s_id, s_jd])
        return sentence_link_list
    
    def get_summary_with_textrank(self, text):
        self.clean()
        sentences, words_list = self.text2words(text)
        sentence_link_list = self.get_sentence_links(words_list)
        for link in sentence_link_list:
            self.add_link(link[0], link[1])
        self.get_PR()
        result = self.get_topN(10)
        summary_sentences = [sentences[i] for i,_ in result]
        return summary_sentences
    
if __name__ == '__main__':
    text = ''.join(list(open('test.txt', 'r', encoding='utf8').readlines()))
    keyword_extractor = TextRankKeyword()
    keyword_weight_list = keyword_extractor.get_keyword_with_textrank(text)
    print("关键词是:")
    print(keyword_weight_list)
    print("#################")
    summary = TextRankSummary()
    summary_sentences = summary.get_summary_with_textrank(text)
    print("摘要是：")
    print('\n'.join(summary_sentences))
    
            