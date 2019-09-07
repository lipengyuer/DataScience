#遗传算法
import random
from matplotlib import pyplot as plt
import numpy as np
class GeneticAlgorithm():
    """用遗传算法求三维坐标系中平面的最高点。z=3*x**2 + 0.66*y**2 -0.51*x+0.002"""
    def __init__(self):
        self.dna_num = 1000#每一轮剩下的dna数量
        self.left_num = int(self.dna_num/2)
        self.dnas = []
    
    #适应性函数，用于评价一个个体对环境的适应能力。相当于我们常听说的代价函数——要不断地优化参数，使个体的适应能力越来越强。
    def fitness(self, x,y):
        z=x**2 + 2*x + 1 + y**2 + 2*y + 1#平面方程
        z = -z#平面是向下凹的，取负数就成向上凸了。这个和后面判断适应能力大小是使用的大小判断配套就可以了。
        return z
    
    #随机生成第一代种群。第一代种群如果质量高，距离最优参数比较近，会加快收敛的速度。
    #对研究对象有比较深的了解的话，可以选择更好的初代生成策略。
    def genera_points(self):
        dnas = []
        for _ in range(self.dna_num):
            data = 100*np.random.rand(2)
            dnas.append([data, 0])
        return dnas
    
    #模拟DNA的变异，随机的对一个个体的各个特征进行增减
    def randomly(self, dna):
        a = dna[::]
        if random.uniform(0,1)>0.55:
            flag = 1
            if random.uniform(0,1)>0.5:
                flag = -1
            a[0][1] += np.random.rand()*0.015*flag
            a[0][0] += np.random.rand()*0.015*flag
        return a
    #交叉。就是随机挑选两个DNA，相互“交流”DNA片段，并组合生成新一代DNA。
    def cross(self, dnas):
        a_index = np.random.randint(0, len(dnas)-1)#生成一个随机数，用于从所有的DNA中随机抽取一个
        b_index = np.random.randint(0, len(dnas)-1)#注意，这里允许两次抽取到同一个个体，也就是自交
        c = []
        for index in [[a_index, b_index], [b_index, a_index]]:
            a = dnas[index[0]][0][0]#取这个DNA的第一维
            b = dnas[index[1]][0][1]#取这个DNA的第二维
            c.append([[a,b],self.fitness(a,b)])
        c = sorted(c, key=lambda x: x[1])  
        res = self.randomly(c[-1])#变异一下
        return res
    
    #train,fit,迭代，只是一种叫法。
    def train(self):
        self.dnas = self.genera_points()
        score_list = []
        x_list = []
        y_list = []
        for i in range(2000):
            score = []
            for dna in self.dnas:
                score.append([dna[0], self.fitness(dna[0][0], dna[0][1])])
            score = sorted(score, key=lambda x: x[1])
            good_dnas = score[-self.left_num:]
            new_dnas = []
            print(list(map(lambda x: x[0], good_dnas[-3:])))
            print("##############")
            for dna in good_dnas:
                new_dnas.append(self.cross(self.dnas))
                new_dnas.append(self.cross(self.dnas))
                new_dnas.append(self.cross(self.dnas))
            self.dnas += new_dnas
            random.shuffle(self.dnas)
            self.dnas = self.dnas[:self.dna_num]
            
            score_list.append(self.dnas[-1][1])
            print(len(self.dnas[-1]), len(x_list))
            x_list.append(self.dnas[-1][0][0])
            y_list.append(self.dnas[-1][0][1])
#         
        plt.subplot(211)
        plt.plot(score_list)
        plt.subplot(212)
        plt.plot(x_list, y_list)
        plt.show()
        
if __name__ == '__main__':
    M = GeneticAlgorithm()
    M.train()
    #print(fitness(1,1))
            
                
            
            