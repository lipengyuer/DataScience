'''
Created on 2020年1月15日

@author: lipy
'''
#基于双数组实现前缀树
import numpy as np
import time  
from  HashMapTrie import TrieHashMap, Node
import copy
from docutils.nodes import term
#这一版需要解决一个问题，即重复遍历词表导致的DAT构件速度太低。
#需要借鉴图的遍历算法，使用一个描述节点是否已经遍历过的数据结构，来辅助遍历过程。
class HashMapTriePlus(TrieHashMap):
    
    def __init__(self, max_word_len):
        self.root_node = Node("", None)
        self.nodes_list = [[] for k in range(max_word_len)]#用来存储还未遍历过的节点
        self.max_word_len = max_word_len
    
    def add_term(self, element_list):
        current_node = self.root_node
        path_to_this = ""
        for depth in range(len(element_list)):
            element = element_list[depth]
            path_to_this += element
            if current_node.children_node_map == None or element not in current_node.children_node_map:
                new_node = Node(element, current_node, path_to_this = path_to_this)
                current_node.add_children_node(new_node)
                self.nodes_list[depth].append(new_node)
            current_node = current_node.children_node_map[element]
        current_node.set_as_leaf()
        
    def get_path(self, node):
        return node.path_to_this
    
    def print_all(self):
        for i in range(self.max_word_len):
            nodes_this_depth = self.nodes_list[i]
            for node in nodes_this_depth:
                print(node.node_name, node.if_leaf(), end=' ')
            print()
                
class DoubleArrayTrie():
    
    def __init__(self, max_word_len):
        self.base = [1]
        self.check = [0]
        self.size = 1
        self.max_ascii_id = 0
        self.min_ascii_id = 0
        self.max_word_len = max_word_len
        self.hash_trie = HashMapTriePlus(max_word_len)#辅助的hash字典树
    
    def iter_patterns_first(self, term_list):
        for term in term_list:
            self.hash_trie.add_term(term)
            for char in term:
                if ord(char) > self.max_ascii_id: self.max_ascii_id = ord(char) 
                if ord(char)  < self.min_ascii_id: self.min_ascii_id = ord(char) 
        print("最大的ascii码是", self.max_ascii_id, "最小是", self.min_ascii_id)
        self.resize(self.max_ascii_id)
        
        for term in term_list: self.hash_trie.add_term(term)
#         self.hash_trie.print_all()
                
    def build(self, term_list):
        self.iter_patterns_first(term_list)
        former_status = 0
        for node in self.hash_trie.nodes_list[0]:
            b_index = self.base[former_status] + ord(node.node_name)
#             print('b_index', b_index, node.node_name, node.if_leaf())
            self.base[b_index] = -1 if node.if_leaf() else 1
            self.check[b_index] = former_status
#         print(self.base)
        print("完成对第一层的初始化")
        for i in range(self.max_word_len):
            print(i, "这一层的节点个数是", len(self.hash_trie.nodes_list[i]))
            nodes_this_depth = self.hash_trie.nodes_list[i]
            for node in nodes_this_depth:
                path_to_this_node = self.hash_trie.get_path(node)
                former_status = 0
                #执行前面的状态转移过程
                former_status = self.update_stage1(former_status, path_to_this_node)
#                 print('former_status', former_status, self.base[former_status], path_to_this_node)
                self.update(former_status, path_to_this_node, node)
                
        
#             if i==0: break
        indexes = list(range(len(self.base)))
        data = [indexes, self.base, self.check]
        data = np.array(data)
#         print(data)
#         for i in indexes:
#             print(i, self.base[i], self.check[i])
#             print("base", self.base)
#             print("check", self.check)

    #假设一个路径已经添加到了dat中，完成对应的状态转移，然后考虑一个后驱节点
    def update_stage1(self,former_status, parent_path):
        if former_status < 0: former_status = -former_status
        former_base = 0
        for a_char in parent_path:
            former_base = self.base[former_status]
            current_status = former_base + ord(a_char) if former_base>0 else \
                                           -former_base + ord(a_char)
            former_status = current_status#完成状态转换
        return former_status
            
    #考察与node同源的节点的情况
    def update(self, former_status, parent_path, node):
        delta = self.base[former_status]
        delta = delta if delta>0 else -delta
        children_nodes = self.hash_trie.get_children_nodes(parent_path)
        while children_nodes!=False:
            if_clean = True
            for b_char in children_nodes.keys():
                b_index = delta + ord(b_char)
                if b_index >= self.size: self.resize(b_index)
                if self.base[b_index]!=0:
                    if_clean = False
                    break
            if if_clean==True:
                break
            delta += 1
#         if parent_path=="人民":
#             print("###", delta, former_status)
        self.base[former_status] = -delta if self.base[former_status] <0 else delta
#         print(self.base, self.base[former_status], former_status, delta)
        if children_nodes!=False:
            for b_char in children_nodes.keys():
                b_index = delta + ord(b_char)
                self.base[b_index] = -1 if children_nodes[b_char].if_leaf() else 1
                self.check[b_index] = former_status
                      
    def containsKey(self, term):
        start_status = 0
        for a_char in term:
            
            former_base = self.base[start_status]
            new_index = former_base + ord(a_char) if former_base>0 else -former_base + ord(a_char)
            print(a_char, start_status, self.check[new_index])
            if self.check[new_index] == start_status:#如果当前节已经收录，不需要插入，开始考虑下一个状态
                start_status = new_index
                continue
            else:
                return False
        print("判断是否为叶子节点", new_index, self.base[new_index])
        if self.base[new_index] < 0:
            return True
        else:
            return False
        
    def resize(self, new_size):
        self.base += [0] * (new_size - len(self.base) + 10000)
        self.check += [0] * (new_size - len(self.check) + 10000)
        self.size = len(self.check)
        
import pickle
if __name__ == '__main__':
    term_list = list(open(r"C:\Users\lipy\Desktop\work\hanlp_data\data\dictionary\CoreNatureDictionary.txt", 'r', encoding='utf8').readlines())
    term_list = list(map(lambda x: x.split("\t")[0], term_list))
    term_list = term_list[:2] + ["中", "中中", "大人的中国"]
#     term_list = term_list[1650:1690] +term_list[1730:1800] +  ["人民"]
    term_list = list(set(term_list))
    term_list = sorted(term_list)
    print(term_list)
    max_len = 1
    for term in term_list:
        if len(term)> max_len: max_len = len(term)
            
    print("词语的最大长度是", max_len)
    dat = DoubleArrayTrie(max_len)
    t1 = time.time()
    dat.build(term_list)
    t2 = time.time()
    pickle.dump(dat, open("dat.pkl", 'wb'))
    print("构建耗时", t2 - t1)

    dat = pickle.load(open("dat.pkl", 'rb'))
    for term in term_list[:10]:#["人民", "中古", ":", "乙十六", "令人堪忧啊"]:
        char_ids_in_term = [ord(key) for key in term]
        print("检查DAT的功能",  dat.containsKey(term))

#     count = 20000000
# #     print("char_ids_in_term", char_ids_in_term)
#     t1 = time.time()
#     for i in range(count):
#         dat.containsKey("东方美亚")
#     t2 = time.time()
#     print("速度是", int(count/(t2 - t1)))
  