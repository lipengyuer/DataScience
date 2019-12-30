'''
Created on 2019年11月28日

@author: lipy
'''
def task1():
    total_money = 3#所有可用资金
    #每个项目的期望收入是相等的；对应的投入也是相等的。收入和投入都经过了归一化。
    earn = [1, 1, 1, 1, 1, 1]
    cost = [1, 1, 1, 1, 1, 1]
    project_num = len(earn)
    
    target_ids = []
    total_income = 0
    total_cost = 0
    import random
    cand_target_ids = list(range(len(earn)))
    random.shuffle(cand_target_ids)
    for project_id in cand_target_ids:
        
        this_earn = earn[project_id]
        this_cost = cost[project_id]
        if total_cost +this_cost<=total_money:
            total_income += this_earn
            total_cost += this_cost
            target_ids.append(project_id)
        else:
            break
    print("要投的项目是", target_ids, "使用的资金量是", total_cost, "期望收入是", total_income)

def task2():
    total_money = 3#所有可用资金
    #每个项目的期望收入不相等；对应的投入是相等的。收入和投入都经过了归一化。
    earn = [1, 2, 3, 1, 5, 2]
    cost = [1, 1, 1, 1, 1, 1]
    project_num = len(earn)
    target_ids = []
    total_income = 0
    total_cost = 0
    cand_target_ids = list(range(project_num))
    project_id_earns = sorted(zip(cand_target_ids,earn), key=lambda x: x[1], reverse=True)
    
    for project_id, this_earn in project_id_earns:
        
        this_cost = cost[project_id]
        if total_cost +this_cost<=total_money:
            total_income += this_earn
            total_cost += this_cost
            target_ids.append(project_id)
        else:
            break
    print("要投的项目是", target_ids, "使用的资金量是", total_cost, "期望收入是", total_income)

def task3():
    total_money = 10
    
    #各个项目的投入金额不相等；它们需要的投资额也不相等
    earn = [1, 2, 3, 1, 5, 2]
    cost = [2, 2, 1, 7, 1, 4]
    project_num = len(earn)
    
    target_ids = []
    total_income = 0
    total_cost = 0
    cand_target_ids = list(range(len(earn)))
    target_ids_plans = [[0], [1]]#可选方案，用0表示不投，1表示投
    
    
    
    ##############生成所有可能的方案###########这实际上是一个前缀树的生长过程
    import copy
    for i in range(project_num-1):
        new_target_ids_plans = copy.deepcopy(target_ids_plans)
        for j in range(len(target_ids_plans)):
            target_ids_plans[j].append(0)
        for j in range(len(new_target_ids_plans)):
            new_target_ids_plans[j].append(1)
        target_ids_plans += new_target_ids_plans
    print("候选方案个数是", len(target_ids_plans))
    #############消耗内存###############
    
    ##########计算每一个方案的总投入和期望收入##############
    import numpy as np
    best_plan = []
    best_income = 0
    best_cost = 0
    for plan in target_ids_plans:
        this_plan_cost = np.dot(np.array(plan), np.array(cost))
        this_plan_earn = np.dot(np.array(plan), np.array(earn))
        if this_plan_cost<=total_money and this_plan_earn>best_income:
            best_income = this_plan_earn
            best_plan = plan
            best_cost = this_plan_cost
    ###########计算量比较大#################        
    print("要投的项目是", best_plan, "使用的资金量是", best_cost, "期望收入是", best_income)
    
    #打印每一个方案
    for i in range(len(target_ids_plans)):
        plan = target_ids_plans[i]
        print(i, plan)#计算5号方案的代价和收入时，可以使用1号方案相应的计算结果——这是各个方案之间存在相关性的结果。
        #各个方案，可以用一个前缀树来存储，共享相同前缀的部分，对应相同的代价和收入
        #换句话说，如果某个前缀对应的代价和收入已经计算过了，它对应的方案可以直接使用已有结果，在这个基础上累加。
        #这样就实现了已有结果的继承，从而避免重复计算
        #前缀树是一个非常适合用来理解和使用动态规划的数据结构。
        #当然，我们可以通过剪枝进一步优化这个算法：当一个路径的代价超过阈值，就停止生长。
    
    #使用前缀树来解决背包问题的一个缺陷，就是在生成所有可行的方案，并计算得到相应代价和收入后，还需要遍历一遍，以找到最佳方案。
        
if __name__ == '__main__':
#     task1()
#     task2()
    task3()
    
    total_money2 = 3
    earn2 = [1, 2, 3, 1, 5, 2]
    cost2 = [1, 1, 1, 1, 1, 1]
    
    total_money3 = 10
    earn3 = [1, 2, 3, 1, 5, 2]
    cost3 = [2, 2, 1, 7, 1, 4]