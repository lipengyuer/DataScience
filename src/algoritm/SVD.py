#实现奇异值分解.输入是一个numpy矩阵，输出是U,sigma,V矩阵
import numpy as np

#基于矩阵分解的结果，复原矩阵
def rebuildMatrix(U, sigma, V):
    a = np.dot(U, sigma)
    a = np.dot(a, np.transpose(V))
    return a

#基于特征值的大小，对特征值以及特征向量进行排序。倒序排列
def sortByEigenValue(Eigenvalues, EigenVectors):
    index = np.argsort(-1*Eigenvalues)
    Eigenvalues = Eigenvalues[index]
    EigenVectors = EigenVectors[:,index]
    return Eigenvalues, EigenVectors

#对一个矩阵进行奇异值分解
def SVD(matrixA, NumOfLeft=None):
    #NumOfLeft是要保留的奇异值的个数，也就是中间那个方阵的宽度
    #首先求transpose(A)*A
    matrixAT_matrixA = np.dot(np.transpose(matrixA), matrixA)
    #然后求右奇异向量
    lambda_V, X_V = np.linalg.eig(matrixAT_matrixA)
    lambda_V, X_V = sortByEigenValue(lambda_V, X_V)
    #求奇异值
    sigmas = lambda_V
    sigmas = list(map(lambda x: np.sqrt(x) if x>0 else 0, sigmas))#python里很小的数有时候是负数
    sigmas = np.array(sigmas)
    sigmasMatrix = np.diag(sigmas)
    if NumOfLeft==None:
        rankOfSigmasMatrix = len(list(filter(lambda x: x>0, sigmas)))#大于0的特征值的个数
    else:
        rankOfSigmasMatrix =NumOfLeft
    sigmasMatrix = sigmasMatrix[0:rankOfSigmasMatrix, :]#特征值为0的奇异值就不要了

    #计算右奇异向量
    X_U = np.zeros((matrixA.shape[0], rankOfSigmasMatrix))#初始化一个右奇异向量矩阵，这里直接进行裁剪
    for i in range(rankOfSigmasMatrix):
        X_U[:,i] = np.transpose(np.dot(matrixA,X_V[:, i])/sigmas[i])

    #对右奇异向量和奇异值矩阵进行裁剪
    X_V = X_V[:,0:NumOfLeft]
    sigmasMatrix = sigmasMatrix[0:rankOfSigmasMatrix, 0:rankOfSigmasMatrix]
    print(rebuildMatrix(X_U, sigmasMatrix, X_V))
    return X_U, sigmasMatrix, X_V

A = np.array([[0, 0, 0, 2, 2], [0, 0, 0, 3, 3], [0, 0, 0, 1, 1], [1, 1, 1, 0, 0],
              [2, 2, 2, 0, 0], [5, 5, 5, 0, 0], [1, 1, 1, 0, 0]])
if __name__ == '__main__':
    SVD(A, NumOfLeft=3)

    
    