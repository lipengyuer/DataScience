'''
Created on 2019年1月24日

@author: pyli
'''
import random
from scipy import stats
#使用EM算法解决一个概率问题。
data ="""正面    反面    正面    正面    正面    反面    正面    反面    正面    反面
反面    正面    正面    反面    正面    正面    反面    正面    反面    正面
正面    反面    反面    正面    正面    反面    正面    反面    正面    正面
正面    反面    正面    正面    反面    正面    正面    反面    正面    反面
反面    正面    反面    正面    正面    正面    正面    正面    反面    正面"""
data = data.split('\n')
data = list(map(lambda x: x.split('    '), data))

#演示前几步的计算过程
def test():
    
    #第一步，假设一个选续A或者B的概率分布，并基于这个概率分布随机决定各轮所用的硬币;
    #然后统计这个假设下，两枚硬币投掷得到正面的概率
    p_A, p_B = 0.4, 0.6
    state = ['B', 'A', 'A', 'B', 'A']#['A' if random.uniform(0,1)<p_A else "B" for i in range(len(data))]
    print("生成的硬币选择为", state)
    p_up_A, p_up_B = 0, 0
    freqMap = {'A': 0, "A_up": 0, "B": 0, "B_up": 0}
    for i in range(len(data)):
        thisState = state[i]
        thisObservationList = data[i]
        for j in range(len(thisObservationList)):
            freqMap[thisState] += 1
            if thisObservationList[j]=="正面":
                freqMap[thisState + '_up'] += 1
    print("统计结果是", freqMap)
    p_up_A, p_up_B = freqMap['A_up']/freqMap['A'], freqMap['B_up']/freqMap['B']
    print(p_up_A, p_up_B)
    
    
    #第2步，基于观测数据，以及两个条件概率，估计每一轮选择两个硬币的概率分布。
    p_A_B_List = [[1, 1] for i in range(len(data))]
    for i in range(len(data)):
        probDist = p_A_B_List[i]
        thisObservationList = data[i]
        tempProb_A, tempProb_B = 1, 1
        for j in range(len(thisObservationList)):
            if thisObservationList[j]=="正面":
                tempProb_A *= p_up_A
                tempProb_B *= p_up_B
            else:
                tempProb_A *= (1- p_up_A)
                tempProb_B *= (1- p_up_B)
        probDist[0] = tempProb_A/(tempProb_A + tempProb_B)
        probDist[1] = tempProb_B/(tempProb_A + tempProb_B)
    print("各轮选择硬币的概率是", p_A_B_List)
    
    #基于各轮选择硬币的概率分布，重新计算一次硬币投出正面的概率，
    p_up_A, p_up_B = 0, 0
    freqMap = {'A': 0, "A_up": 0, "B": 0, "B_up": 0}
    for i in range(len(data)):
        stateProb = p_A_B_List[i]
        thisObservationList = data[i]
        for j in range(len(thisObservationList)):
            freqMap['A'] += stateProb[0]
            freqMap['B'] += stateProb[1]
            if thisObservationList[j]=="正面":
                freqMap['A_up'] += stateProb[0]
                freqMap['B_up'] += stateProb[1]
    print("统计结果是", freqMap)
    p_up_A, p_up_B = freqMap['A_up']/freqMap['A'], freqMap['B_up']/freqMap['B']
    print(p_up_A, p_up_B)

roundNum = 500
import time
def genarateData():
    random.seed = int(time.time())
    prob_A, prob_B= 0.7, 0.3
    prob_up_A, prob_up_B = 0.3, 0.9
    if_A = True#是否选择硬币A
    dataList = []
    coinList = []
    for _ in range(roundNum):
        if random.uniform(0, 1)<prob_A: 
            if_A = True#选择硬币A
            coinList.append("A")
        else: 
            if_A = False#选择硬币B
            coinList.append("B")
        #基于选择的硬币生成正反面序列
        if if_A==True: tempList = [1 if random.uniform(0,1)<prob_up_A else 0for _ in range(10)]
        else: tempList = [1 if random.uniform(0,1)<prob_up_B else 0 for _ in range(10)]
        dataList.append(tempList)
    print("#######开始打印真实参数###########")
    print("隐藏参数，也就是选择硬币A的概率是", prob_A, '选择硬币B的概率是', prob_B)
    print("条件概率，也就是硬币A掷出正面的概率是", prob_up_A, '硬币B掷出正面的概率是', prob_up_B)
    [print(coinList[i], dataList[i]) for i in range(len(coinList))]
    print("#############完成打印真实参数############")
#     dataList = [[1,0,0,0,1,1,0,1,0,1],
#                             [1,1,1,1,0,1,1,1,0,1],
#                             [1,0,1,1,1,1,1,0,1,1],
#                             [1,0,1,0,0,0,1,1,0,0],
#                             [0,1,1,1,0,1,1,1,0,1]]
    return dataList          

from scipy.special import comb, perm        
#基于隐藏参数，估计条件概率                      
def EM_step(prob_A_B_list, prob_up, dataList):
    
    for i in range(len(dataList)):
        observationList = dataList[i]
        tempProbA, tempProbB = 1, 1#使用一枚硬币投掷出这个观测序列的概率
        up_num = 0
        for j in range(len(observationList)):
            if observationList[j]==1:
                tempProbA *= prob_up['A']
                tempProbB *= prob_up['B']
                up_num += 1
            else:
                tempProbA *= (1 - prob_up['A'])
                tempProbB *= (1 - prob_up['B'])
#         tempProbA *= comb(len(observationList), up_num)
#         tempProbB *= comb(len(observationList), up_num)
#         tempProbA = stats.binom.pmf(up_num,len(observationList),prob_up['A'])
#         tempProbB = stats.binom.pmf(up_num,len(observationList),prob_up['B'])
        prob_A_B_this_round = {}
        prob_A_B_this_round['A'] = tempProbA/(tempProbA + tempProbB)
        prob_A_B_this_round['B'] = tempProbB/(tempProbA + tempProbB)
        prob_A_B_list[i] = prob_A_B_this_round
        
    expectationSumMap = {'A_up': 0, 'A_down': 0, 'B_up': 0, "B_down": 0}#两种硬币投掷出正面，和反面的次数期望值之和
    for i in range(len(dataList)):
        observationList = dataList[i]
        prob_A_B_this_round = prob_A_B_list[i]
        for j in range(len(observationList)):
            if observationList[j]==1:
                expectationSumMap['A_up'] += prob_A_B_this_round['A']*1.0
                expectationSumMap['B_up'] += prob_A_B_this_round['B']*1.
            else:
                expectationSumMap['A_down'] += prob_A_B_this_round['A'] * 1.
                expectationSumMap['B_down'] += prob_A_B_this_round['B'] * 1.  
         
    #更新条件概率
    prob_up['A'] = expectationSumMap['A_up']/(expectationSumMap['A_up'] + expectationSumMap['A_down'])
    prob_up['B'] = expectationSumMap['B_up']/(expectationSumMap['B_up'] + expectationSumMap['B_down'])


        
    #基于新的隐藏参数，估计新的条件概率
    

#使用EM算法去估计两个条件概率
def EM():
    dataList = genarateData()#生成观测数据
    #定义一个变量，用来存储隐藏参数，也就是选择硬币的概率分布
    prob_A_B_list = [{"A": 0.7, 'B': 0.3} for _ in range(roundNum)]#这里顺便给了一个初始值
    #定义一个变量，用来存储我们最终要求的概率分布，即硬币A和硬币B各自掷出正面的概率
    prob_up = {"A": 0.6, "B": 0.4}#这里顺便给了一个无意义的取值
    epochNum = 100#迭代执行的次数。这里为了简单，没有使用更加合理的停止条件，即判断参数是否收敛。
    for epoch in range(epochNum):
        #首先执行E步骤，即基于隐藏参数，计算条件概率的期望值
        EM_step(prob_A_B_list, prob_up, dataList)
        print(prob_up)
#         print(prob_A_B_list)
    

if __name__ == '__main__':
    EM()







