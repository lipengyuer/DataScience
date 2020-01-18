'''
Created on 2020年1月15日

@author: lipy
'''
#一个基于hashmap实现的前缀树
#需要支持这样的节点遍历方式:递归地遍历一个子节点的所有子节点；直到所有节点遍历完毕。
import copy

class Node():
    
    def __init__(self, node_name, parent_node=None, path_to_this=None):
        self.node_name = node_name#当前节点存储的元素值
        self.children_node_map = None#当前节点的子节点信息
        self.parent_node = parent_node#父节点引用，用于从任意一个节点出发，获取到达这里的路径。微观上，这种节点可以支持双链表
        self.is_leaf = False
        self.path_to_this = path_to_this

    #是否为叶子结点
    def if_leaf(self):
        return self.is_leaf

    def add_children_node(self, a_node):
        if self.children_node_map==None: self.children_node_map = {}
        self.children_node_map[a_node.node_name] = a_node
        
    def set_as_leaf(self):
        self.is_leaf = True
    
        
class TrieHashMap():
    
    def __init__(self):
        self.root_node = Node("root")

    def add_term(self, element_list):
        current_node = self.root_node
        parent_path = ""
        for element in element_list:
            if current_node.children_node_map == None or element not in current_node.children_node_map:
                new_node = Node(element)
                
                current_node.add_children_node(new_node)
            current_node = current_node.children_node_map[element]
            parent_path += element
        print("这个模式的末尾节点是", current_node.node_name)
        current_node.set_as_leaf()
            
    def containsKey(self, element_list):
        current_node = self.root_node
        for element in element_list:
            if element not in current_node.children_node_map:
                return False 
            current_node = current_node.children_node_map[element]
        if current_node.if_leaf(): return True
        else: return False
    
    def get_children_node_names(self, parent_path):
        current_node = self.root_node
        for element in parent_path:
            if element not in current_node.children_node_map:
                return False 
            current_node = current_node.children_node_map[element]
        if current_node.if_leaf(): return False
        else: return list(current_node.children_node_map.keys())
        
    def get_children_nodes(self, parent_path):
        current_node = self.root_node
        for element in parent_path:
            if element not in current_node.children_node_map:
                return False 
            current_node = current_node.children_node_map[element]
        if current_node.children_node_map==None: return False
        else: return current_node.children_node_map
        
if __name__ == '__main__':
    trie = TrieHashMap()
    trie.add_term([1,2,3,4,5])
    trie.add_term([1,2,3,6,7,8])
    print(trie.containsKey([1,2,3,4,5]))
    print(trie.containsKey([1,2,6,7]))
        
    
    
    
    
    
    