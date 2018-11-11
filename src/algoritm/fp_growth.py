#实现关联规则

# 数据集
def loadSimpDat():
    simDat = [['r','z','h','j','p'],
              ['z','y','x','w','v','u','t','s'],
              ['z'],
              ['r','x','n','o','s'],
              ['y','r','x','z','q','t','p'],
              ['y','z','x','e','q','s','t','m']]
    return simDat

# 构造成 element : count 的形式
def createInitSet(dataSet):
    retDict={}
    for trans in dataSet:
        key = frozenset(trans)
        if key in retDict:
            retDict[frozenset(trans)] += 1
        else:
            retDict[frozenset(trans)] = 1
    return retDict


# coding:utf-8
from numpy import *


# 本代码来自<机器学习实战>
# 注释的对应的场景是购物篮分析

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur  # 这个item在所有的购物篮中出现的总数
        self.nodeLink = None
        self.parent = parentNode  # needs to be updated
        self.children = {}

    def inc(self, numOccur):  # 增加item的频数
        self.count += numOccur

    def disp(self, ind=1):  # 递归展示树的内容
        print(' ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


# FP构建函数
def createTree(dataSet, minSup=1):
    headerTable = {}  # 头表，用于存储支持度大于阈值的items，并按照支持续倒序排列的结果
    for trans in dataSet:  # 遍历数据集
        for item in trans:  # 遍历这个事务里的所有item
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]  # 记录每个元素项出现的频度
    # 删除支持度地域阈值的item
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del (headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:  # 不满足最小值支持度要求的除去
        return None, None  # 如果没有item符合最低支持度的要求，那么就没有可用的规则了，直接退出
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]  # 给起始节点留位置
    retTree = treeNode('Null Set', 1, None)  # 配置跟节点
    for tranSet, count in dataSet.items():  # 遍历数据集
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]  # 把headerTable里存储的item的频数取出来
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1],
                                                 reverse=True)]  # 按照频数将这个事务里的item倒序排列
            updateTree(orderedItems, retTree, headerTable, count)  # 向fp树添加数据
    return retTree, headerTable


# 将item序列添加到fp树里
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:  # 如果item序列的第一个在当前节点的子节点里，这个子节点的频数加１
        inTree.children[items[0]].inc(count)  # 如果item序列的第一个在当前节点的子节点里dren[items[0]].inc(count)
    else:  # 如果item序列的第一个item不在当前节点的子节点里
        inTree.children[items[0]] = treeNode(items[0], count, inTree)  # 为fp树创建一个新的子节点
        if headerTable[items[0]][1] == None:  # 如果headerTable里这个item还没有指向末个节点的指针，添加上面创建的子节点的指针
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):  # ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):  # treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]  # (sort header table)
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None:
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def mineRuleWithFPTree(itemsLists, suport_min=100):
    initSet = createInitSet(itemsLists)
    myFPtree, myHeaderTab = createTree(initSet, suport_min)
    myFreqList = []
    if myHeaderTab == None:
        return []
    mineTree(myFPtree, myHeaderTab, suport_min, set([]), myFreqList)
    return myFreqList
def mineFreqSet(itemsLists):
    data = createInitSet(itemsLists)


if __name__ == '__main__':
    data = loadSimpDat()
    data = createInitSet(data)
    print(data)