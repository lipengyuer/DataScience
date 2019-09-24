import pyhanlp

#用于存储图
class Graph():
    def __init__(self):
        self.linked_node_map = {}#邻接表，
        self.PR_map ={}#存储每个节点的入度
        self.stop_words = set({'我'})
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
        self.linked_node_map[node1].add(node2)#为node1添加一个邻接节点，表示ndoe2引用了node1。这里认为两个词语之间的边的权重统一为1。
        self.linked_node_map[node2].add(node1)#因为是无向图，需要给两个节点都添加；邻接节点
            
    #计算pr
    def get_PR(self, epoch_num=10, d=0.5):#配置迭代轮数，以及阻尼系数
        for i in range(epoch_num):
            for node in self.PR_map:#遍历每一个节点
                self.PR_map[node] = (1-d) + d*sum([self.PR_map[temp_node] for temp_node in self.linked_node_map[node]])#原始版公式
    
    def segment(self, text):
        word_tag_list = pyhanlp.HanLP.segment(text)
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
    
    def get_keyword(self, text):
        word_link_list = self.get_word_links(text)        
        for relation in word_link_list:
            self.add_link(relation[0], relation[1])
        self.get_PR()
        
        keyword = sorted(self.PR_map.items(), key=lambda x: x[1], reverse=True)[:10]
        return keyword
        

text = """程序员(英文Programmer)是从事程序开发、维护的专业人员。一般将程序员分为程序设计人员和程序编码人员，但两者的界限并不非常清楚，
特别是在中国。软件从业人员分为初级程序员、高级程序员、系统分析员和项目经理四大类。"""
if __name__ == '__main__':
    graph = Graph()
    res = graph.get_keyword(text)
    print(res)
    
    
    