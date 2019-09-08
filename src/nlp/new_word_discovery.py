'''
Created on 2019年5月11日

@author: Administrator
很久以前捋的这个算法，现在整理起来比较难受:(1)内容需要温习；（2）代码思路需要使劲回忆。后者的锅还得自己背，没写好注释就会这样。
'''
#针对大规模语料的tf-idf统计。
#语料较大的时候，无法全部加载到内存中，需要逐行读取。
#统计环节比较消耗算力，组好并行化。
#多个进程对多个文件块的统计结果需要合并为一份数据。这个任务只能是一个进程来完成了

import multiprocessing
import os
import numpy as np

min_word_length=2#词语最小长度
max_word_length=6#词语最大长度

#这里假设标点符号和数字不会被用来组成词语。这些符号可以用空格来替换掉
marks = {'，', ',', '。', '.', '?', '？', ':', '：', '!', '！', '%', '”', '"', '“', '、', '—'}
digits = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ', '    '}

#递归地遍历rootDir文件夹里的所有文件
def iter_files(result, rootDir):
    #遍历根目录
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            result.append(file_name)
        for dirname in dirs:
            iter_files(result, dirname)#递归调用自身,只改变目录名称
    return result

#将文本切分为ngram,n可调
def get_ngrams(line, min_N=1, max_N=4):
    if len(line)<=max_N: return []
    res = []
    for j in range(min_N, max_N+1):
        for i in range(len(line)-j):
            res.append(line[i: i + j])
    return res

#判断文本是否包含数字
def contains_digits(line):
    for char in line: 
        if char in digits: return True
    return False

#
def generate_binary_tree(plans, line):
    if len(line)==0:
        return 
    else:
        for i in range(len(plans)):
            plan = plans[i]
            plan_b = plan + [line[0]]
            plans.append(plan_b)
            plan[-1] = plan[-1] + line[0]
        line = line[1:]
        generate_binary_tree(plans, line)

def generate_link_plans(line):
    plans = [[line[0]]]
    generate_binary_tree(plans, line[1:])
    return plans
        
def update_gram_score(ngram_freq_map, left_right_char_freq_map, line):
    for ngram_len in range(1, max_word_length + 1):
        for i in range(len(line) - ngram_len):
            ngram = line[i : i + ngram_len]
            if contains_digits(ngram): continue
            ngram_freq_map[ngram] = ngram_freq_map.get(ngram, 0) +1
            if ngram not in left_right_char_freq_map: 
                left_right_char_freq_map[ngram] = {'left_char': {}, 'right_char': {}}
        line = '*' + line + '*'
        for i in range(1, len(line) - ngram_len-1):
            ngram = line[i : i + ngram_len]
            if contains_digits(ngram): continue
            left_char, right_char = line[i-1:i], line[i + ngram_len: i + ngram_len + 1]
            left_right_char_freq_map[ngram]['left_char'][left_char] = \
                 left_right_char_freq_map[ngram]['left_char'].get(left_char, 0) +1
            left_right_char_freq_map[ngram]['right_char'][right_char] = \
                 left_right_char_freq_map[ngram]['right_char'].get(right_char, 0) +1

import numpy as np
"""在新词发现中，我们需要计算一个文字片段x1,x2,x3,x4,x5,...,x_L的内部凝聚度agg。
agg=num(x1+x2+x3+x4+x5 + ... + x_L)/(所有子片段的频数乘积)
“所有子片段”，指的是我们将文字片段切分之后得到的一个序列——切分的方式有很多种，我们需要挑选
其中乘积最小的那个切分方式得到的序列，并以这个乘积来代表文字片段的凝聚度。
文字片段中的char可以看做一排节点，只有相邻的节点可以创建边来连接——边的状态就是“存在”或“不存在“。
联通的若干节点组成一个子片段，这样就完成了对文字片段的切分。
举例来说,x1和x2的连接方式是2种；x2和x3的连接方式是2种；...。上述文字片段的连接方式有2**(L-1)种，每一种连接方式
对应一种切分方式。
可以首先罗列出所有的切分方式，然后计算得到最小的频数乘积即可。需要考虑使用维特比算法吗？改造一下问题，应该可以使用。
把词边界也表示成一个字符，
"""
def cal_ngram_agglomeration(ngram_freq_map):
    ngram_agglomeration_map = {}
    for ngram in ngram_freq_map:
        if len(ngram)==1 or '*' in ngram: continue
        link_plans = generate_link_plans(ngram)
        
        min_agglomeration = 1
        for plan in link_plans:
            agglomeration = 1
            for gram in plan:
                agglomeration *= ngram_freq_map.get(gram, 1)
            if agglomeration<min_agglomeration:
                min_agglomeration = agglomeration
        #print(min_agglomeration)
        ngram_agglomeration_map[ngram] = ngram_freq_map[ngram]/min_agglomeration
    return ngram_agglomeration_map

def cal_entropy(gram_freq_map):
    entropy = 0
    gram_num = sum(gram_freq_map.values())
    for gram in gram_freq_map:
        prob = gram_freq_map[gram] / gram_num
        entropy -= prob*np.log(prob)
    return entropy
    
def cal_left_right_char_entropy(left_right_char_freq_map):
    left_right_char_entropy_map = {}
    for ngram in left_right_char_freq_map:
        left_char_freq_map = left_right_char_freq_map[ngram]['left_char']
        right_char_freq_map = left_right_char_freq_map[ngram]['right_char']
        left_right_char_entropy_map[ngram] = \
                   cal_entropy(left_char_freq_map) + cal_entropy(right_char_freq_map)
    return left_right_char_entropy_map

def cal_final_score(ngram_agglomeration_map, left_right_char_entropy_map):
    score_map = {}
#     print(left_right_char_entropy_map)
    for ngram in ngram_agglomeration_map:
        if ngram in left_right_char_entropy_map:
            
            score_map[ngram] = ngram_agglomeration_map[ngram] + left_right_char_entropy_map[ngram]**2
    return score_map

def map_filter(a_map, percent = 25):
    threhold_value = np.percentile(list(a_map.values()), 25)
    for key in a_map:
        if a_map[key]<threhold_value:
            del a_map[key]
        
def new_word_discovery(file_list):
    count = 1
    flag = 1
    ngram_freq_map, left_right_char_freq_map = {}, {}
    for file in file_list:
        
        if 'SUCCESS' in file: continue
        print(file)
        with open(file, 'r', encoding='utf8') as f:
            line = f.readline()
            while len(line)>0:
                #print(line)
                count += 1
                for mark in marks: line = line.replace(mark, ' ')
                print(count, len(ngram_freq_map))
                update_gram_score(ngram_freq_map, left_right_char_freq_map, line)
                line = f.readline()
                if count>5000:
                    flag = 0
                    break
        if flag==0:
            print( 'task count ', count)
            break
    print('finished reading.')
#     print(ngram_freq_map)
    ngram_agglomeration_map = cal_ngram_agglomeration(ngram_freq_map)
    map_filter(ngram_agglomeration_map)
    left_right_char_entropy_map = cal_left_right_char_entropy(left_right_char_freq_map)
    map_filter(left_right_char_entropy_map, percent=80)
    ngram_score_map = cal_final_score(ngram_agglomeration_map, left_right_char_entropy_map)
    
    #print(left_right_char_entropy_map)
    words = sorted(ngram_score_map.items(), key=lambda x: x[1], reverse=True)
    words = list(filter(lambda x: x[1]>0, words))[:2000]
    with open('new_words.txt', 'w', encoding='utf8') as f:
        for word in words:
            f.write(str(word[0]) + ' ' + str(word[1]) + '\n')
        
    
if __name__ == '__main__':
    soure_data_dir = r'E:\work\data\稻谷原标注语料'
    files = []
    iter_files(files, soure_data_dir)
    new_word_discovery(files)
  