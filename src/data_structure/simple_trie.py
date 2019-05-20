'''
Created on 2019年5月18日
@author: Administrator
'''
#实现一个光板前缀树，用来查找符合条件(前面若干个字相同)的问句

#节点
class Node():
    
    def __init__(self, this_value):
        self.this_value = this_value#当前节点的取值，也就是一个字符
        self.children = {}#当前节点的子节点，dict来存储
        
#用于存储单词的前缀树    
class BaseTrie():
    
    def __init__(self):
        self.children = {'root': Node("root")}
    
    #添加新问句
    def add_new_path(self, new_path):
        temp_node = self.children['root']
        for node in new_path:
            if node in temp_node.children:
                temp_node = temp_node.children[node]
            else:
                new_node = Node(node)
                temp_node.children[node] = new_node
                temp_node = temp_node.children[node]
    #删除一个问句        
    def delete_a_path(self, a_path):
        temp_node = self.children['root']
        for node in a_path:
            if node in temp_node.children:
                if len(temp_node.children)==1:
                    temp_node.children = {}
                    return 
                temp_node = temp_node.children[node]
                
    #召回所有前缀相同的问句
    def get_sim_path(self, a_path):
        res = []
        temp_node = self.children['root'].children.get(a_path[0])
        if temp_node==None: return []
        get_sim_pathes(temp_node,a_path, '',  res)
        return res

    def print_trie(self):
        temp_node = self.children['root']
#         print(temp_node)
        get_pathes(temp_node, '')

#递归地召回前缀树上的所有路径，用于打印
def get_pathes(child_tree, now_path):
    if child_tree.children=={}:
        print(now_path)
    else:
        now_path += child_tree.this_value
        for child_node in child_tree.children:
            get_pathes(child_tree.children[child_node], now_path)

#递归地召回树上，具有特定前缀的所有路径         
def get_sim_pathes(child_tree, left_path, found_path, collector): 
    if len(left_path)>0 and child_tree.this_value!=left_path[0]:
        return 
    elif len(left_path)==0 and child_tree.children=={}: 
            found_path += child_tree.this_value
            collector.append(found_path)
    else:
        found_path += child_tree.this_value
        for child_node in child_tree.children:
            get_sim_pathes(child_tree.children[child_node], left_path[1:],found_path, collector)
     
            
import time         
if __name__ == '__main__':
    #ss = ["我爱北京天安门。", "我爱北京的天安门。"]
    ss = list(open(r'C:\Users\lipy\eclipse-workspace\Lab\src\test\short_long_text\tianlongbabu.txt', 'r', encoding='utf8').readlines())
    a_simple_trie = BaseTrie()
    for s in ss:
        a_simple_trie.add_new_path(list(s))
#     print(a_simple_trie.__dict__)
    #a_simple_trie.print_trie()\
    a_simple_trie.delete_a_path('十九 虽万千人吾往矣\n')
    t1 = time.time()
    print(a_simple_trie.get_sim_path("十九"))
    t2 = time.time()
    print("耗时是", t2-t1)
