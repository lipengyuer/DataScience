'''
Created on 2019年1月8日

@author: pyli
'''
#实现各种排序算法,对一个列表中的浮点数按照从小到大的顺序排序

#插入排序，将未排序的数据插入到已排序的数据中的合适位置。
#这里需要初始化以排序数字，从而启动后面的过程;另外，假如一个数字比已排序数字都小的话，需要放在最左边
def insertSort(numList):
    leftNum = numList[0]
    if leftNum>numList[1]:  
        leftNum = numList[1]
        rightNum = numList[0]
    else:
        rightNum = numList[1]
    
    res = [leftNum, rightNum]#假定最后一个数字是最大的
    for num in numList[2:]:#遍历前面的所有数字
        ifAllLarger = True
        for i in range(len(res)-1):
            leftNum = res[i]
            rightNum = res[i+1]
            if leftNum <=num <=rightNum:
                ifAllLarger = False
                res = res[:i+1] + [num] + res[i+1:]
                break
        if ifAllLarger==True:
            res = [num] + res
    return res
                
#冒泡
def bubbleSort(numList):
    newnumList = []
#快速

if __name__ == '__main__':
    a = [1,6, 3,2,5,3,5, 2]
    print(insertSort(a))
