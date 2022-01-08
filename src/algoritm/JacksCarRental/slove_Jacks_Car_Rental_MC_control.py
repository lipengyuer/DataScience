'''
Created on 2022年1月6日

@author: Administrator
'''
#《RL》65页Example 4.2: Jack’s Car Rental
#使用蒙特卡洛控制方法求解。新增设定：Jack的公司只开10天。
import numpy as np
import copy
from boto3.docs import action
from argparse import Action
from scipy.stats import poisson
import time
import seaborn as sns
import matplotlib.pyplot as plt
import random

running_day_num = 50#假设Jack开若干天就休息一段时间。

max_car_number_available = 20#20
car_rental_company_state_space = list(range(max_car_number_available + 1))#最少0个；最多20个
Jacks_action_space = [5, 4, 3, 2, 1, 0, -1, -2 , -3, -4 ,5]
car_move_costs = 2
cat_rent_return = 10
need_poisson_theta1 = 3
need_poisson_theta2 = 4
return_car_poisson_theta1 = 3
return_car_poisson_theta2 = 2

max_need = 11#最大需求。这个累积概率已经很高了。
gamma = 0.9
Q  = {}
policy = np.zeros((max_car_number_available + 1, max_car_number_available + 1), dtype=np.int32)

cached_probs = {}
        
def get_prob(n, lam):
    global cached_probs
    if (n, lam) not in cached_probs:
        cached_probs[(n, lam)] = poisson.pmf(n, lam)
    return cached_probs[(n, lam)]
    
def get_return(state, needs, action):
    available_car_number = copy.deepcopy(state)
    available_car_number[0] -= action
    available_car_number[1] += action
    for i in range(2):
        if available_car_number[i]>max_car_number_available: available_car_number[i] = max_car_number_available
        if available_car_number[i]<0: available_car_number[i] = 0
        
    return (min(needs[0], available_car_number[0]) + \
            min(needs[1], available_car_number[1])) * cat_rent_return

def get_new_state(morning_state, need, action, night_return_car_nums):
    state = copy.deepcopy(morning_state)
    
    #调配车辆
    state[0] -= action
    state[1] += action
    for i in range(2):
        if state[i]>max_car_number_available: state[i] = max_car_number_available
        if state[i]<0: state[i] = 0
        
    #白天借车
    for i in range(2): state[i] = max(state[i] - need[i], 0)
        
    #将归还的车辆放入货架
    for i in range(2):
        state[i] = min(max_car_number_available, state[i] + night_return_car_nums[i])

    return state

#使用当前策略生成一个episode
def generate_new_episode(policy):
    episode = []
    state = [0, 0]
    for i in range(running_day_num):
        today_needs = [np.random.poisson(need_poisson_theta1), np.random.poisson(need_poisson_theta2)]
        today_return_car_nums = [np.random.poisson(return_car_poisson_theta1), 
                                       np.random.poisson(return_car_poisson_theta2)]
        action = policy[state[0], state[1]]
        state_pi = get_new_state(state, today_needs, action, today_return_car_nums)
        today_return = get_return(state, today_needs, action) - abs(action)*car_move_costs
        episode.append([tuple(state), action, today_return])
        state = state_pi
    return episode


class ReturnStack:
    
    def __init__(self):
        self.mean = 0
        self.returns = []
        
    def add(self, r):
        self.mean = self.mean + (r - self.mean)/(len(self.returns) + 1)
        self.returns.append(r)
    
import threading  
def mente_carlo_control():
    returns = {}
        
    max_epoch = 500000

    for epoch in range(max_epoch):
        
        if epoch%1000==0: print(f"第{epoch}次采样,完成进度{int(100*epoch/max_epoch)}%")
        #生成episode
        an_episode = generate_new_episode(policy)
       
        #评估
        processed_s_a = set({})
        for t in range(len(an_episode)):
            
            state, action, today_return = an_episode[t]
            key = (state, action)
            if key in processed_s_a:continue
            else: processed_s_a.add(key)
            reward = today_return
            cached_dc = 1
            for j in range(t, len(an_episode), 1):
                reward += cached_dc * today_return
                cached_dc *= gamma
                
            if key not in returns: returns[key] = ReturnStack()
            
            returns[key].add(reward)
            
            if state not in Q: Q[state] = {}
            Q[state][action] = returns[key].mean
            
        #提升
        for t in range(len(an_episode)):
            state, _, _ = an_episode[t]
            return_of_each_action = Q[state]
            return_of_each_action = sorted(return_of_each_action.items(), key=lambda x: x[1])
            if random.uniform(0, 1)>0.05:
                policy[state] = return_of_each_action[-1][0]
            else:
                policy[state] = Jacks_action_space[random.randint(0, len(Jacks_action_space)-1)]
        
        
    fig = sns.heatmap(policy, cmap="YlGnBu")
    plt.savefig("figure_policy_" + f"{epoch}" + ".png")
    plt.show()
    plt.close()
if __name__ == '__main__':
    mente_carlo_control()
    

