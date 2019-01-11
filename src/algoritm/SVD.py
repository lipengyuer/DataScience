#实现奇异值分解.输入是一个numpy矩阵，输出是U,sigma,V矩阵
import numpy as np

def SVD(matrixA):
    #首先求A*transpose(A)和transpose(A)*A
    matrixA_matrixAT = np.dot(matrixA, np.transpose(matrixA))
#     print(np.transpose(matrixA), matrixA)
    matrixAT_matrixA = np.dot(np.transpose(matrixA), matrixA)
    print(matrixA_matrixAT)
    print(matrixAT_matrixA)
    #求两个矩阵的特征值以及特征向量矩阵
    lambda_U, X_U = np.linalg.eig(matrixA_matrixAT)
    lambda_V, X_V = np.linalg.eig(matrixAT_matrixA)
#     print(lambda_U, X_U)
#     print(lambda_V, X_V)
    #按照特征值大小，对特征值向量和特征向量矩阵进行排序
    index_U = np.argsort(-lambda_U)
    lambda_U, X_U = lambda_U[index_U], X_U[:, index_U]
    index_V = np.argsort(-lambda_V)
    lambda_V, X_V = lambda_V[index_V], X_V[:, index_V]
#     print(index_U)
#     X_V = -X_V
    print("########")
    print(lambda_U)
    print(X_U)
    print("########")
    print(lambda_V)
    print( X_V)
    print("#############")
    #初始化sigma矩阵
    index_U = np.argsort(-lambda_U)
    lambda_U = lambda_U[index_U]
    brandth = np.min([len(lambda_U), len(lambda_V)])
    sigmaMatrix = np.zeros((matrixA.shape[0], brandth))
    for i in range(brandth):
        sigmaMatrix[i,i] = np.sqrt(lambda_U[i])
#     print(sigmaMatrix)
    return X_U, sigmaMatrix, X_V


A = np.array([[2,4],[1, 3],[0, 0],[0, 0]])
# A = np.array([[0,1 ],[1, 1 ],[1, 0]])
# A = np.array([[1,1 ],[1, 1 ],[0, 0]])
# A = np.array([[4,4 ],[-3, 3 ]])
if __name__ == '__main__':
    X_U, sigmaMatrix, X_V = SVD(A)
    #基于U,V,sigma复原A
#     print(X_U)
#     print(sigmaMatrix)
#     print(X_V)
    A = np.dot(X_U, sigmaMatrix)
    A = np.dot(A, np.transpose(X_V))

    print(A)
#     print()
#     print(np.linalg.svd(A))
    
    
    