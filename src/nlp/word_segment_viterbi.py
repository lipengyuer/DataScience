#使用viterbi算法-最短路径法，来对文本进行分词
import numpy as np
import pickle

class ViterbiSegment():
    
    def __init__(self, mode="train"):
        if mode=="work":
            self.vocab, self.word_distance, self.max_word_len = pickle.load(open("model.pkl", 'rb'))
    
    #基于标注语料，训练一份词语的概率分布，以及条件概率分布————当然最终目的，是得到两个词语之间的连接权重(也可以理解为转移概率)
    #转移概率越大，说明两个词语前后相邻的概率越大，那么，从前一个词转移到后一个词语花费的代价就越小。
    def train_simple(self, default_corpus_size = None):
        #https://raw.githubusercontent.com/liwenzhu/corpusZh/master/corpus/corpus_%E4%B8%80_20140804162433.txt
        self.word_num = {}
        self.word_pair_num = {}
        with open('corpus_segment.txt', 'r', encoding='utf8') as f:
            lines = f.readlines()
            if default_corpus_size!=None: lines = lines[:default_corpus_size]
            print("文档总行数是", len(lines))
            for line in lines:
                line = line.split("\t\t ")
                if len(line)==2:
                    line = line[1].replace('\n', '')
                    words = list(map(lambda x: x.split('/')[0], line.split(' ')))
                    words = list(filter(lambda x: len(x)>0, words))
                    words = ["<start>"] + words + ["<end>"]
                    for word in words:
                        self.word_num[word] = self.word_num.get(word, 0) + 1
                    
                    for i in range(len(words)-1):
                        word_pair = (words[i], words[i+1])#由于要计算的是条件概率，词语先后是需要考虑的
                        self.word_pair_num[word_pair] = self.word_pair_num.get(word_pair, 0) + 1
        #p(AB)=p(A)*p(B|A)=(num_A/num_all)*(num_AB/num_A)=num_AB/num_all。
        #这个权重计算公式的优点是计算效率快；缺点是丢失了num_A带来的信息
        #这个训练算法的效率不太重要；权重包含的信息量尽量大，或者说更精准地刻画词语对的分布，是最重要的事情。
        #hanlp设计了一个权重计算方式,来综合考虑num_A，num_all， num_A带来的信息。
        num_all = np.sum(list(self.word_num.values()))
        word_pair_prob = {}
        for word_pair in self.word_pair_num:
            word_pair_prob[word_pair] = self.word_pair_num[word_pair]/num_all
            
        #由于我们最终要做的是求最短路径，要求图的边权重是一个表示“代价”或者距离的量，即权重越大，两个节点之间的距离就越远。而前面得到的条件概率与这个距离是负相关的
        #我们需要对条件概率求倒数，来获得符合场景要求的权重
        #另外，由于条件概率可能是一个非常小的数，比如0.000001，倒数会很大。我们在运行维特比的时候，需要把多条边的权重加起来——可能遇到上溢出的情况。
        #常用的避免上溢出的策略是去自然对数。
        self.word_distance = {}
        for word_pair in self.word_pair_num:
            word_A, word_B = word_pair
            self.word_distance[word_pair] = np.log(1/word_pair_prob[word_pair])
            
        self.vocab = set(list(self.word_num.keys()))
        self.max_word_len = 0
        for word in self.vocab:
            if len(word)>self.max_word_len: self.max_word_len = len(word)
        
        model = (self.vocab, self.word_distance, self.max_word_len)
        pickle.dump(model, open("model.pkl", 'wb'))
    
    def train_hanlp(self, default_corpus_size = None):
        """
        hanlp里使用的连接器权重计算方式稍微复杂一点，综合考虑了前词出现的概率，以及后词出现的条件规律，有点像全概率p(A)*p(B|A)=p(AB)
#         dSmoothingPara 平滑参数0.1, frequency A出现的频率, MAX_FREQUENCY 总词频
#         dTemp 平滑因子 1 / MAX_FREQUENCY + 0.00001, nTwoWordsFreq AB共现频次
#         -Math.log(dSmoothingPara * frequency / (MAX_FREQUENCY)
#         + (1 - dSmoothingPara) * ((1 - dTemp) * nTwoWordsFreq / frequency + dTemp));
        """

        self.word_num = {}
        self.word_pair_num = {}
        with open('corpus_segment.txt', 'r', encoding='utf8') as f:
            lines = f.readlines()
            if default_corpus_size!=None: lines = lines[:default_corpus_size]
            print("文档总行数是", len(lines))
            for line in lines:
                line = line.split("\t\t ")
                if len(line)==2:
                    line = line[1].replace('\n', '')
                    words = list(map(lambda x: x.split('/')[0], line.split(' ')))
                    words = list(filter(lambda x: len(x)>0, words))
                    words = ["<start>"] + words + ["<end>"]
                    for word in words:
                        self.word_num[word] = self.word_num.get(word, 0) + 1
                    
                    for i in range(len(words)-1):
                        word_pair = (words[i], words[i+1])#由于要计算的是条件概率，词语先后是需要考虑的
                        self.word_pair_num[word_pair] = self.word_pair_num.get(word_pair, 0) + 1

        num_all = np.sum(list(self.word_num.values()))
        dSmoothingPara = 0.1
        dTemp = 1 / num_all + 0.00001
        word_pair_prob = {}
        for word_pair in self.word_pair_num:
            word_A, word_B = word_pair
            #hanlp里的权重计算公式比较复杂，在查不到设计思路的情况下，我们默认hanlp作者是辛苦研制之后，凑出来的~
            word_pair_prob[word_pair] = dSmoothingPara*self.word_num.get(word_A)/ num_all + \
                            (1 - dSmoothingPara)*( (1 - dTemp) *self.word_pair_num[word_pair] / (self.word_num.get(word_A) + dTemp))
            
        #由于我们最终要做的是求最短路径，要求图的边权重是一个表示“代价”或者距离的量，即权重越大，两个节点之间的距离就越远。而前面得到的条件概率与这个距离是负相关的
        #我们需要对条件概率求倒数，来获得符合场景要求的权重
        #另外，由于条件概率可能是一个非常小的数，比如0.000001，倒数会很大。我们在运行维特比的时候，需要把多条边的权重加起来——可能遇到上溢出的情况。
        #常用的避免上溢出的策略是去自然对数。
        self.word_distance = {}
        for word_pair in self.word_pair_num:
            word_A, word_B = word_pair
            self.word_distance[word_pair] = np.log(1/word_pair_prob[word_pair])
            
        self.vocab = set(list(self.word_num.keys()))
        self.max_word_len = 0
        for word in self.vocab:
            if len(word)>self.max_word_len: self.max_word_len = len(word)
        
        model = (self.vocab, self.word_distance, self.max_word_len)
        pickle.dump(model, open("model.pkl", 'wb'))
        
    def generate_word_graph(self, text):
        word_graph = []
        for i in range(len(text)):
            cand_words = []
            window_len = self.max_word_len
            if i + self.max_word_len>=len(text): window_len = len(text)-i + 1
            for j in range(window_len):
                cand_word = text[i: i + j]
                next_index = i + len(cand_word) + 1
                if cand_word in self.vocab:
                    cand_words.append([cand_word, next_index])
            cand_words.append([text[i], i + 2])
            if len(cand_words)>0:
                word_graph.append(cand_words)
#             else:
#                 word_graph.append([[text[i], i + 2]])
        return word_graph
                
    
    def viterbi(self, word_graph):
        path_length_map = {}
        word_graph = [[["<start>", 1]]] + word_graph + [[["<end>", -1]]]
        print(word_graph)
        path_length_map[("<start>", )] = [1, 0]
        
        for i in range(1, len(word_graph)):
            distance_from_start2current = {}
            if len(word_graph[i])==0: continue
            
            for path in list(path_length_map.keys()):
                [next_index_4_later_path, later_distance] = path_length_map[path]
                #print(path, path_length_map)
                
                later_path = list(path)
                if next_index_4_later_path==i:
                    del path_length_map[path]
                    for current_word in word_graph[i]:
                        later_word = path[-1]
                        current_word, next_index = current_word
                        new_path = tuple(later_path + [current_word])
                        new_patn_len = later_distance + self.word_distance.get((later_word, current_word), 100)
                        
                        path_length_map[new_path] = [next_index, new_patn_len]
                        if current_word in distance_from_start2current:
                            if distance_from_start2current[current_word][1]>new_patn_len:
                                distance_from_start2current[current_word] = [new_path, new_patn_len]
                        else:
                            distance_from_start2current[current_word] = [new_path, new_patn_len]
            print(i , len(path_length_map), "distance_from_start2current", distance_from_start2current)
            print("####################")
        sortest_path = distance_from_start2current["<end>"][0]
        sortest_path = sortest_path[1:-1]
        return sortest_path
                    
    def segment(self, text):
        word_graph = self.generate_word_graph(text)
        shortest_path = self.viterbi(word_graph)
        return shortest_path
    
    def evaluation(self):
        pass
import time
if __name__ == '__main__':
    V = ViterbiSegment(mode="work")
    V.train_hanlp(default_corpus_size=None)
    print(V.segment("体操小将王惠莹艰苦拚搏。"))
#     print(V.word_distance[("体操", "小将")])
    t1 = time.time()
    res = V.segment("由于我们最终要做的是求最短路径，要求图的边权重是一个表示“代价”或者距离的量，即权重越大")
    t2 = time.time()
    print(t2-t1, res)
    
    