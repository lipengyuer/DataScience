#用于存储图
class Graph():
    def __init__(self):
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
    def get_PR(self, epoch_num=10, d=0.5):#配置迭代轮数，以及阻尼系数
        for i in range(epoch_num):
            for node in self.PR_map:#遍历每一个节点
                self.PR_map[node] = (1-d) + d*sum([self.PR_map[temp_node] for temp_node in self.linked_node_map[node]])#原始版公式
            print(self.PR_map)
            

edges = [[1,2], [3,2], [3,5], [1,3], [2,3], [3, 1], [5,1]]#模拟的一个网页链接网络       
if __name__ == '__main__':
    graph = Graph()
    for edge in edges:
        graph.add_link(edge[0], edge[1])
    graph.get_PR()
    
            
            
            
