'''
Created on 2020年1月15日

@author: lipy
'''
#基于双数组实现前缀树
import numpy as np
from  HashMapTrie import TrieHashMap
#https://segmentfault.com/a/1190000008877595
char_id_map = {}

class DoubleArrayTrie():
    
    def __init__(self):
        self.base = [1]
        self.check = [0]
        self.size = 1
        self.max_ascii_id = 0
        self.min_ascii_id = 0
        self.hash_trie = TrieHashMap()#辅助的hash字典树
        self.processed_path_set = set()#将已经处理过的路径，即模式片段缓存起来。已经添加到数组中的模式片段，不再添加
    
    def iter_patterns_first(self, term_list):
        for term in term_list:
            self.hash_trie.add_term(term)
            for char in term:
                if char_id_map[char] > self.max_ascii_id: self.max_ascii_id = char_id_map[char]
                if char_id_map[char] < self.min_ascii_id: self.min_ascii_id = char_id_map[char]
        print("最大的ascii码是", self.max_ascii_id, "最小是", self.min_ascii_id)
        self.resize(self.max_ascii_id)
                
    def add_term(self, element_list, depth):
        for term in element_list:
            former_status = 0
            this_depth = 0
            for a_char in term:
                this_depth += 1
                
                current_status = np.abs(self.base[former_status]) + char_id_map[a_char]
                if current_status >= len(self.base) : self.resize(current_status + 10)
                if this_depth < depth: 
                    former_status = current_status
                    continue
                elif this_depth > depth:
                    break
                else:
#                     print(term[: this_depth], self.processed_path_set)
                    if term[: this_depth] in self.processed_path_set: break
                    self.processed_path_set.add(term[: this_depth])
                    if current_status >= len(self.base) : self.resize(current_status + 10)
                    if self.base[current_status]==0:#如果当前状态对应的位置是空的，可以直接添加新节点
                        self.base[current_status] = -1 if this_depth==len(term) else 1#叶子结点的状态取值是负的
                        self.check[current_status] = former_status#check数组更新
                        former_status = current_status#完成状态转换
                    else:
                        if self.check[current_status] == former_status:#如果要添加的节点已经收录了，不需要做什么操作，转换状态即可
                            former_status = current_status
                        else:#如果当前位置存储的状态不为空，需要检查待添加节点的子节点是否可以添加。如果子节点们可以添加，则当前位置存储的位移量是可用的。
                            delta = self.base[int(np.abs(former_status))]
                            ori_temp_start = self.base[int(np.abs(former_status))]
                            parent_path = term[: depth-1]
                            children_node_names = self.hash_trie.get_children_node_names(parent_path)
                            while children_node_names!=False:
                                if_clean = True
                                for b_char in children_node_names:
                                    if delta + char_id_map[b_char] >= len(self.base) : self.resize(delta + char_id_map[b_char] + 10)
                                    if self.base[delta + char_id_map[b_char]]!=0:
                                        if_clean = False
                                        break
                                if if_clean==True:
                                    break
                                delta += 1
                            self.base[former_status] = delta if self.base[former_status] > 0 else -delta 
#                             print(delta, current_status, children_node_names)
                            if children_node_names!=False:
                                for b_char in children_node_names:
#                                     print(this_depth, len(term))
                                    abs_v = np.abs(self.base[ori_temp_start + char_id_map[b_char]])
                                    self.base[delta + char_id_map[b_char]] = abs_v if this_depth!=len(term) else -abs_v
                                    self.check[delta + char_id_map[b_char]] = int(np.abs(former_status))
                                    self.processed_path_set.add(parent_path +b_char )
                            former_status = current_status
        
        indexes = list(range(len(self.base)))
        data = [indexes, self.base, self.check]
        data = np.array(data)
#         print(data)
#         print("base", self.base)
#         print("check", self.check)
    
    def containsKey(self, char_ids_in_term):
        start_status = 0
        for a_char_id in char_ids_in_term:
            new_index = self.base[start_status] + a_char_id if self.base[start_status]>0 else -self.base[start_status] + a_char_id
            if self.base[new_index]==0:#如果位置是空的
                return False
            else:#如果位置不为空，需要处理冲突
                if self.check[new_index] == start_status:#如果当前节已经收录，不需要插入，开始考虑下一个状态
                    start_status = new_index
                    continue
                else:
                    return False
                start_status = new_index
        if self.base[new_index] < 0:
            return True
        else:
            return False
        
    def resize(self, new_size):
        self.base += [0] * (new_size - len(self.base) + 1000)
        self.check += [0] * (new_size - len(self.check) + 1000)
        self.size = len(self.check)
import time   
if __name__ == '__main__':
    term_list = list(open(r"C:\Users\lipy\Desktop\work\hanlp_data\data\dictionary\CoreNatureDictionary.txt", 'r', encoding='utf8').readlines())
    term_list = list(map(lambda x: x.split("\t")[0], term_list))
    term_list = term_list[:100000] + ["人民"]
#     print(term_list)
    max_len = 1
    for term in term_list:
        if len(term)> max_len: max_len = len(term)
        for a_char in term:
            char_id_map[a_char] = len(char_id_map)
#     print(char_id_map)
    hash_trie = TrieHashMap()
    hashMap = {}
    for term in term_list: hashMap[term] = 1
    for term in term_list:
#         print("初始化辅助trie", term) 
        hash_trie.add_term(term)
    print("检查hash字典树", hash_trie.containsKey("人民"))
    print("检查获取子节点的功能", hash_trie.get_children_node_names("人民"))
    dat = DoubleArrayTrie()
    dat.iter_patterns_first(term_list)
    print("完成第一阶段")
    for i in range(1, max_len + 1):
        dat.add_term(term_list, i)
        print(max_len, i, "---------------------")

#     print(dat.containsKey("大力"))
#     print(dat.containsKey("人民"))
#     print(dat.containsKey("人民力量"))
#     print(dat.containsKey("人民力量大"))
#     print(dat.containsKey("哈工大"))
    #######################################
    t1 = time.time()
    count = 200000
    char_ids_in_term = [char_id_map.get(key, 5862) for key in "人民力量大"]
#     print("char_ids_in_term", char_ids_in_term)
    for i in range(count):
        dat.containsKey(char_ids_in_term)
    t2 = time.time()
    print("速度是", count/(t2 - t1))
    t1 = time.time()
    for i in range(count):
        "人民力量大" in hashMap
    t2 = time.time()
    print("速度是", count/(t2 - t1))    
    
