'''
Created on 2022年1月6日

@author: Administrator
'''
#《RL》65页Example 4.2: Jack’s Car Rental
import numpy as np
import copy
from boto3.docs import action
from argparse import Action
from scipy.stats import poisson
import time
import seaborn as sns
import matplotlib.pyplot as plt


max_car_number_available = 20#20
car_rental_company_state_space = list(range(max_car_number_available + 1))#最少0个；最多20个
Jacks_action_space = [5, 4,3,2,1, 0, -1, -2 , -3, -4 ,5]
car_move_costs = 2
cat_rent_return = 10
need_poisson_theta1 = 3
need_poisson_theta2 = 4
return_car_poisson_theta1 = 3
return_car_poisson_theta2 = 2

max_need = 11#最大需求。这个累积概率已经很高了。
gamma = 0.9
state_values = np.zeros((max_car_number_available + 1, max_car_number_available + 1))
policy = np.zeros(state_values.shape, dtype=np.int32)

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
    for i in range(2):
        state[i] = max(state[i] - need[i], 0)
        
    #将归还的车辆放入货架
    for i in range(2):
        state[i] = min(max_car_number_available, state[i] + night_return_car_nums[i])

    return state
    

#状态价值函数
def state_value_fuction(current_state, action, simple=True):
    state_values_temp = copy.deepcopy(state_values)
    state = copy.deepcopy(current_state)
    time_cost = 0
    t1 = time.time()
    new_value = 0 - abs(action) * car_move_costs
    for need1 in range(max_need):
        for need2 in range(max_need):
            prob_needs = get_prob(need1, need_poisson_theta1) * get_prob(need2, need_poisson_theta2)
            today_needs = [need1, need2]
            r = get_return(state, today_needs, action)
            
            if simple:
                state_pi = get_new_state(state, today_needs, action, \
                                [return_car_poisson_theta1, return_car_poisson_theta2])
                new_value += prob_needs * (r + gamma * state_values_temp[state_pi[0], state_pi[1]])
            else:
                for returned_car_num1 in range(max_need):
                    for returned_car_num2 in range(max_need):                    
                        today_return_car_nums = [returned_car_num1, returned_car_num2]                            
                        state_pi = get_new_state(state, today_needs, action, today_return_car_nums)
                        new_value += prob_needs * \
                                         get_prob(returned_car_num1, return_car_poisson_theta1) * \
                                         get_prob(returned_car_num2, return_car_poisson_theta2)* \
                                         (r + gamma * state_values_temp[state_pi[0], state_pi[1]])
                    
    state_values_temp[state[0], state[1]] = new_value
    t2 = time.time()
#     print("计算一个状态的价值耗时", t2-t1, "计算回报、新状态等的耗时是", time_cost)
    return state_values_temp[state[0], state[1]] 

def policy_iteration():
    
    for epoch in range(5):
        #策略评估
        for prediction_epoch in range(5):
            old_value = copy.deepcopy(state_values)
            for s1 in range(max_car_number_available + 1):
                for s2 in range(max_car_number_available + 1):
                    state = [s1, s2]
                    action = policy[s1, s2]
                    state_values[state] = state_value_fuction(state, action)
                if s1%10==0:
                    print("策略评估完成进度", s1, "/", max_car_number_available)
            max_value_change = abs(old_value - state_values).max()
            print(f"策略评估第{prediction_epoch}次循环，max_value_change", max_value_change)
        
        #策略提升
        for s1 in range(max_car_number_available + 1):
            if s1%10==0:
                print("策略更新完成进度", s1, "/", max_car_number_available)
            for s2 in range(max_car_number_available + 1):
                action_returns = []
                for action in Jacks_action_space:
                    if (0 <= action <= s1) or (-s2 <= action <= 0):
                        action_returns.append(state_value_fuction([s1, s2], action))
                    else:
                        action_returns.append(-np.inf)
                new_action = Jacks_action_space[np.argmax(action_returns)]
                policy[s1, s2] = new_action
        
        fig = sns.heatmap(state_values, cmap="YlGnBu")
        plt.savefig("figure_value_" + f"{epoch}" + ".png")
        plt.close()
        
        fig = sns.heatmap(policy, cmap="YlGnBu")
        plt.savefig("figure_policy_" + f"{epoch}" + ".png")
        plt.close()
                    
if __name__ == '__main__':
    policy_iteration()
    

