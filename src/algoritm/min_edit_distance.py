'''
Created on 2019年10月28日

@author: Administrator
'''
#最小编辑距离
import numpy as np

class ViolentMinEditDistance_v1():
    #枚举所有可行的编辑方案，然后逐一计算代价，最后求最小代价
    def __init__(self, A, B):
        if len(B)<len(A):#为了后面计算简单，这里假设A比B要短
            self.ori_B = A
            self.ori_A = B            
        else:
            self.ori_A = A
            self.ori_B = B
        self.len_ori_B = len(B)
        #，用这个值做标记
        self.edit_path_list = []#用来收集所有可行的编辑方案。这里将每个编辑方案看做一个路径。这样有利于后面理解动态规划算法的”状态“
        
        self.edit_cost_map = {"del": 1, "add": 1, "rep": 2, "keep": 0}
        self.min_edit_distance = -1
        self.best_path = ""
    
    def cal_score(self, path):
        score = 0
        for edit in path:
            score += self.edit_cost_map[edit]
        return score
            
    def fit(self):
        self.min_edit_distance_violent(self.ori_A, 0, [])
        for path in self.edit_path_list:
            score = self.cal_score(path)
            print(path, score)#展示每个编辑路径的内容和得分
            if score<self.min_edit_distance or self.min_edit_distance==-1:
                self.min_edit_distance = score
                self.best_path = path
        print("最小编辑距离是", self.min_edit_distance, self.best_path)
        
    #删除一个字符串的指定位置的字符
    def delete_char(self, a_str, index):   
        return a_str[:index] + a_str[index+1:] 
    
    #在字符串的指定位置添加一个字符
    def add_char(self, a_str, index, new_char):   
        return a_str[:index] + new_char +  a_str[index:] 
    
    #将字符串指定位置的一个字符，替换为另一个字符
    def repalce_char(self, a_str, index, new_char): 
        return a_str[:index] + new_char +  a_str[index + 1:] 
              
    #使用递归的方式生成所有的编辑序列，然后遍历、计算代价，求出最小代价作为最小编辑距离
    def min_edit_distance_violent(self, latest_A, edit_depth, this_path):
        if latest_A==self.ori_B:
            self.edit_path_list.append(this_path)
            return
        elif  edit_depth>=self.len_ori_B:
            return 
        else:
            if edit_depth>=len(latest_A):#如果当前编辑位置超出原始A，无法使用替换或者删除操作，只能添加B对应位置的字符。这是A短于B的情况
                self.min_edit_distance_violent(self.add_char(latest_A, edit_depth, self.ori_B[edit_depth]),\
                                                edit_depth+1, this_path + ['add'])            
            elif edit_depth>=self.len_ori_B:#如果当前编辑位置超出原始A，无法使用替换或者删除操作，只能添加B对应位置的字符。这是A短于B的情况
                self.min_edit_distance_violent(self.delete_char(latest_A, edit_depth, self.ori_B[edit_depth]),\
                                                edit_depth+1, this_path + ['del'])  
            else:
                #删除
                self.min_edit_distance_violent(self.delete_char(latest_A, edit_depth), edit_depth+1, this_path + ['del'])
                #添加
                self.min_edit_distance_violent(self.add_char(latest_A, edit_depth, self.ori_B[edit_depth]),\
                                                edit_depth+1, this_path + ['add'])
                #替换
#                 print(edit_depth, len(latest_A))
                if latest_A[edit_depth]==self.ori_B[edit_depth]:#如果这个位置上，A和B的字符相同，就不用替换了
                    self.min_edit_distance_violent(latest_A, edit_depth+1, this_path + ["keep"])
                else:#如果不同，还是需要替换
                    self.min_edit_distance_violent(self.repalce_char(latest_A, edit_depth, self.ori_B[edit_depth]),\
                                                edit_depth+1, this_path + ['rep'])
            return
        
class ViolentMinEditDistance_v2():
    #在递归的过程中，每一次编辑，顺便基于当前路径的累计代价，计算到达这一步的代价。相比v1版本，这里避免了大量的重复计算
    def __init__(self, A, B):
        if len(B)<len(A):#为了后面计算简单，这里假设A比B要短
            self.ori_B = A
            self.ori_A = B            
        else:
            self.ori_A = A
            self.ori_B = B
        self.len_ori_B = len(B)
        #，用这个值做标记
        self.edit_path_list = []#用来收集所有可行的编辑方案。这里将每个编辑方案看做一个路径。这样有利于后面理解动态规划算法的”状态“
        
        self.min_edit_distance = -1#-1表示还没有初始化
        self.best_path = None
         
    def fit(self):
        self.min_edit_distance_violent(self.ori_A, 0, [], 0)
        for path, score in self.edit_path_list:
            print(path, score)#展示每个编辑路径的内容和得分
            if score<self.min_edit_distance or self.min_edit_distance==-1:
                self.min_edit_distance = score
                self.best_path = path
        print("最小编辑距离是", self.min_edit_distance, self.best_path,"可选的路径数量是", len(self.edit_path_list))
        
    #删除一个字符串的指定位置的字符
    def delete_char(self, a_str, index):   
        return a_str[:index] + a_str[index+1:] 
    
    #在字符串的指定位置添加一个字符
    def add_char(self, a_str, index, new_char):   
        return a_str[:index] + new_char +  a_str[index:] 
    
    #将字符串指定位置的一个字符，替换为另一个字符
    def repalce_char(self, a_str, index, new_char): 
        return a_str[:index] + new_char +  a_str[index + 1:] 
              
    #使用递归的方式生成所有的编辑序列，然后遍历、计算代价，求出最小代价作为最小编辑距离
    def min_edit_distance_violent(self, latest_A, edit_depth, this_path, this_path_score):
        if latest_A==self.ori_B:
            self.edit_path_list.append([this_path,this_path_score])
            return
        elif  edit_depth>=self.len_ori_B:
            return 
        else:
            if edit_depth>=len(latest_A):#如果当前编辑位置超出原始A，无法使用替换或者删除操作，只能添加B对应位置的字符。这是A短于B的情况
                self.min_edit_distance_violent(self.add_char(latest_A, edit_depth, self.ori_B[edit_depth]),\
                                                edit_depth+1, this_path + ['add'], this_path_score + 1)            
            elif edit_depth>=self.len_ori_B:#如果当前编辑位置超出原始A，无法使用替换或者删除操作，只能添加B对应位置的字符。这是A短于B的情况
                self.min_edit_distance_violent(self.delete_char(latest_A, edit_depth, self.ori_B[edit_depth]),\
                                                edit_depth+1, this_path + ['del'], this_path_score + 1)  
            else:
                #删除
                self.min_edit_distance_violent(self.delete_char(latest_A, edit_depth), \
                                               edit_depth+1, this_path + ['del'], this_path_score + 1)
                #添加
                self.min_edit_distance_violent(self.add_char(latest_A, edit_depth, self.ori_B[edit_depth]),\
                                                edit_depth+1, this_path + ['add'], this_path_score + 1)
                #替换
#                 print(edit_depth, len(latest_A))
                if latest_A[edit_depth]==self.ori_B[edit_depth]:#如果这个位置上，A和B的字符相同，就不用替换了
                    self.min_edit_distance_violent(latest_A, edit_depth+1, this_path + ["keep"], this_path_score)
                else:#如果不同，还是需要替换
                    self.min_edit_distance_violent(self.repalce_char(latest_A, edit_depth, self.ori_B[edit_depth]),\
                                                edit_depth+1, this_path + ['rep'], this_path_score + 2)
            return

class DPMinEditDistance():
        
    def __init__(self, A, B):          
        self.A = "#" + A#在A和B的头部添加“#”的目的，是占一个位置，用来表示一个“相同的开始”，
        #即二者在最开始的位置是相等的，编辑代价是0，后面在这个基础上累计
        self.B = "#" + B
        self.A_len = len(self.A)
        self.B_len = len(self.B)
        #，用这个值做标记
        self.edit_path_list = []#用来收集所有可行的编辑方案。这里将每个编辑方案看做一个路径。这样有利于后面理解动态规划算法的”状态“
        
        #初始化编辑路径得分矩阵
        self.step_matrix = np.zeros((self.A_len, self.B_len))
        for i in range(self.A_len): self.step_matrix[i,0] = i#这一列相当于将A的字符全部删除
        for i in range(self.B_len): self.step_matrix[0, i] = i#这一列相当于把B的字符串全部、依次添加到A的末尾

    def fit(self):
        for i in range(1,self.A_len):
            for j in range(1, self.B_len):
                self.step_matrix[i, j] = self.d_i_j(self.step_matrix, i, j)#使用状态更新公式计算到达每一个为指导的代价
        print(self.step_matrix)
        
        #回溯得到最佳编辑方案
        index_A, index_B = self.A_len-1, self.B_len-1
        best_edit_path = []
        print("最小编辑距离是", self.step_matrix[index_A, index_B])
        while index_A>0 and index_B>0:
            best_cost = -1
            best_edit = None
            if index_A-1>-1:
                if self.step_matrix[index_A-1, index_B]<best_cost or best_cost==-1:
                    best_cost = self.step_matrix[index_A-1, index_B]
                    best_edit = 'add'
            if index_B-1>-1:
                if self.step_matrix[index_A, index_B-1]<best_cost or best_cost==-1:
                    best_cost = self.step_matrix[index_A, index_B-1]
                    best_edit = 'del' 
            if index_A-1>-1 and index_B-1>-1:             
                if self.step_matrix[index_A-1, index_B-1]<best_cost or best_cost==-1:
                    best_cost = self.step_matrix[index_A-1, index_B]
                    best_edit = 'rep'
                    
                if self.step_matrix[index_A-1, index_B-1]<self.step_matrix[index_A, index_B] \
                                                                                  or best_cost==-1:
                    best_cost = self.step_matrix[index_A-1, index_B]
                    best_edit = 'keep'     
            if best_edit in ["keep", 'rep']: 
                index_A -= 1
                index_B -= 1
            if best_edit == "del": 
                index_B -= 1
            if best_edit == "add": 
                index_A -= 1
            best_edit_path = [best_edit] + best_edit_path
            print(index_A, index_B, best_edit)
        print("最佳编辑路径是", best_edit_path)    
    
    #计算到达当前状态的最佳路径，对应的总代价    
    def d_i_j(self, step_matrix, i, j): 
        c1 = step_matrix[i-1, j] + 1
        c2 = step_matrix[i, j-1] + 1
        c3 = step_matrix[i-1, j-1] + 1
        if self.A[i]==self.B[j]:
            c3 = step_matrix[i-1, j-1]
        min_c = min(c1, c2, c3)
        return min_c
    #             print(step_matrix)         
if __name__ == '__main__':
    s1 = "小猪饿了"
    s2= "小懒猪饿了"
#     get_min_edit_distance(list(s1), list(s2))
#     vilent_v1 = ViolentMinEditDistance_v1(s1, s2)
#     vilent_v1.fit()
    
    vilent_v2 = ViolentMinEditDistance_v2(s1, s2)
    vilent_v2.fit()

    DP_version = DPMinEditDistance(s1, s2)
    DP_version.fit()
    
    
    
    