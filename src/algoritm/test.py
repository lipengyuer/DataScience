#!/usr/bin/python
 
# crf.py (by Graham Neubig)
#  This script trains conditional random fields (CRFs)
#  stdin: A corpus of WORD_POS WORD_POS WORD_POS sentences
#  stdout: Feature vectors for emission and transition properties
 
from collections import defaultdict
from math import log, exp
import sys
import operator
 
# The L2 regularization coefficient and learning rate for SGD
l2_coeff = 1
rate = 10
 
# A dictionary to map tags to integers
tagids = defaultdict(lambda: len(tagids))
tagids["<S>"] = 0
 
############# Utility functions ###################
def dot(A, B):
    return sum(A[k]*B[k] for k in A if k in B)
def add(A, B):
    C = defaultdict(A, lambda: 0)
    # for k, v in A.items(): C[k] += v
    for k, v in B.items(): C[k] += v
    return C
def logsumexp(A):
    k = max(A)
    return log(sum( exp(i-k) for i in A ))+k
 
############# Functions for memoized probability
def calc_feat(x, i, l, r):#提取特征
    return { ("T", l, r): 1, ("E", r, x[i]): 1 }

def calc_e(x, i, l, r, w, e_prob):
    if (i, l, r) not in e_prob:
        e_prob[i,l,r] = dot(calc_feat(x, i, l, r), w)
    return e_prob[i,l,r]

#计算前向变量取值
def calc_f(x, i, l, w, e, f):
    if (i, l) not in f:
        if i == 0:
            f[i,0] = 0
        else:
            prev_states = (range(1, len(tagids)) if i != 1 else [0])
            f[i,l] = logsumexp([
                calc_f(x, i-1, k, w, e, f) + calc_e(x, i, k, l, w, e)
                    for k in prev_states])
    return f[i,l]
#计算后向变量的取值
def calc_b(x, i, r, w, e, b):
    if (i, r) not in b:
        if i == len(x)-1:
            b[i,0] = 0
        else:
            prev_states = (range(1, len(tagids)) if i != len(x)-2 else [0])
            b[i,r] = logsumexp([
                calc_b(x, i+1, k, w, e, b) + calc_e(x, i, r, k, w, e)
                    for k in prev_states])
    return b[i,r]
 
############# Function to calculate gradient ######
def calc_gradient(x, y, w):
    f_prob = {(0,0): 0}#前向算法计算结果
    b_prob = {(len(x)-1,0): 0}#后向算法计算结果
    e_prob = {}
    grad = defaultdict(lambda: 0)
    # Add the features for the numerator
    for i in range(1, len(x)):
        for k, v in calc_feat(x, i, y[i-1], y[i]).items():
            grad[k] += v
    # Calculate the likelihood and normalizing constant
    norm = calc_b(x, 0, 0, w, e_prob, b_prob)
    lik = dot(grad, w) - norm
    # Subtract the features for the denominator
    for i in range(1, len(x)):
        for l in (range(1, len(tagids)) if i != 1 else [0]):
            for r in (range(1, len(tagids)) if i != len(x)-1 else [0]):
                # Find the probability of using this path
                p = exp(calc_e(x, i, l, r, w, e_prob)
                        + calc_b(x, i,   r, w, e_prob, b_prob)
                        + calc_f(x, i-1, l, w, e_prob, f_prob)
                        - norm)
                # Subtract the expectation of the features
                for k, v in calc_feat(x, i, l, r).items():
                    grad[k] -= v * p
    # print grad
    # Return the gradient and likelihood
    return (grad, lik)

def loadData(fileName, sentenceNum = 100):
    with open(fileName, 'r', encoding='utf8') as f:
        line = f.readline()
        tempSentence = []
        tempT = []
        count = 0
        corpus = []
        while line!=True:
            line = line.replace('\n', '')
            if line=='':#如果这一行没有字符，说明到了句子的末尾
                tempSentence = ''.join(tempSentence)#把字符都连接起来形成字符串，后面操作的时候会快一些
#                 if "习近平" in tempSentence:
#                     print(tempSentence)
                tempTag = ' '.join(tempT)
                # print('asd', tempTag)
                corpus.append(tempTag)
#                 corpus.append([tempSentence[:20],tempTag[:20]])
#                 print("这句话是", [tempSentence,tempTag])
                tempSentence = []
                tempT = []
                count += 1
                if count==sentenceNum:#如果积累的句子个数达到阈值，返回语料
                    return corpus
            else:
                line= line.split('\t')
#                 print(line)
                chartag = line[0] + '_' + line[1]#取出语料的文字和分词标记
                tempSentence.append(chartag)
                tempT.append(chartag)
            line = f.readline()
    return corpus

############### Main training loop
if __name__ == '__main__':
    # load in the corpus
    corpus1 = []
    lines = loadData(r"trainingCorpus4wordSeg_part.txt", sentenceNum=100)
    # print(lines)
    for i in range(len(lines)):
        line = lines[i]
        words = [ "<S>" ]
        tags = [   0   ]
        # print(line)
        line = line.strip()
        for w_t in line.split(" "):
            w, t = w_t.split("_")
            words.append(w)
            tags.append(tagids[t])
        words.append("<S>")
        tags.append(0)
        # print(words, tags)
        corpus1.append( (words, tags) )
        # print(line)
    # for 50 iterations
    # print(corpus1)
    w = defaultdict(lambda: 0)
    for iternum in range(1, 50+1):
        grad = defaultdict(lambda: 0)
        # Perform regularization
        reg_lik = 0
        for k, v in w.items():
            grad[k] -= 2*v*l2_coeff
            reg_lik -= v*v*l2_coeff
        # Get the gradients and likelihoods
        lik = 0
        # print(corpus)
        for (x, y) in corpus1:
            my_grad, my_lik = calc_gradient(x, y, w)
            for k, v in my_grad.items():
                grad[k] += v
            lik += my_lik
        l1 = sum( [abs(k) for k in grad.values()] )
        print (sys.stderr, "Iter %r likelihood: lik=%r, reg=%r, reg+lik=%r gradL1=%r" % (iternum, lik, reg_lik, lik+reg_lik, l1))
        # Here we are updating the weights with SGD, but a better optimization
        # algorithm is necessary if you want to use this in practice.
        for k, v in grad.items():
            w[k] += v/l1*rate
 
    # Reverse the tag strings
    strs = list(range(0, len(tagids)))
    for k, v in tagids.items(): strs[v] = k
    # Print the features
    # for k, v in sorted(w.items(), key=operator.itemgetter(1)):
    #     if k[0] == "E":
    #         print( "%s %s %s\t%r" % (k[0], strs[k[1]], k[2], v))
    #     else:
    #         print( "%s %s %s\t%r" % (k[0], strs[k[1]], strs[k[2]], v))
