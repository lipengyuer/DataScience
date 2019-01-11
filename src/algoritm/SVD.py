#实现奇异值分解.输入是一个numpy矩阵，输出是U,sigma,V矩阵
import numpy as np

def exchangeNum(index1, index2):
    num = 0
    for i in range(len(index1)):
        if index1[i]!=index2[i]:
            num+=1
    num -=1
    if num <0:
        num=0
    return num

def rebuildMatrix(U, sigma, V):
    a = np.dot(U, sigma)
    a = np.dot(a, np.transpose(V))
    return a

def sortByEigenValue(Eigenvalues, EigenVectors):
    index = np.argsort(-1*Eigenvalues)
    Eigenvalues = Eigenvalues[index]
    EigenVectors = EigenVectors[:,index]
#     EigenVectors = EigenVectors[index, :]
    return Eigenvalues, EigenVectors

def  changeSignal(aMatrix):
#     print(aMatrix)
    for i in range(aMatrix.shape[1]):
        num = 0
        for j in range(aMatrix.shape[0]):
            if aMatrix[j,i]<0:
                num +=1
        if num>0.5*aMatrix.shape[0]:
#             print('asdasdasd')
            aMatrix[:,i] *=-1
#     print(aMatrix)
    return aMatrix

def SVD(matrixA):
    #首先求A*transpose(A)和transpose(A)*A
    matrixA_matrixAT = np.dot(matrixA, np.transpose(matrixA))
#     print(np.transpose(matrixA), matrixA)
    matrixAT_matrixA = np.dot(np.transpose(matrixA), matrixA)
#     print(matrixA_matrixAT)
#     print(matrixAT_matrixA)
    #求两个矩阵的特征值以及特征向量矩阵
    lambda_U, X_U = np.linalg.eig(matrixA_matrixAT)
    lambda_U, X_U = sortByEigenValue(lambda_U, X_U)
    X_U = changeSignal(X_U)
    print(X_U)

#     print(X_U)

    lambda_V, X_V = np.linalg.eig(matrixAT_matrixA)
    lambda_V, X_V = sortByEigenValue(lambda_V, X_V)
    X_V = changeSignal(X_V)
#     print(X_V)

#     print(X_V)

    a = rebuildMatrix(X_U, np.diag(lambda_U), X_U)
    b = rebuildMatrix(X_V, np.diag(lambda_V), X_V)
#     print("复原", a)
#     print("复原", b)
#     print(lambda_U)
#     print(X_U)
#     print(lambda_V)
#     print(X_V)
#     print(np.dot(X_U, np.transpose(X_U)))
    h = X_U.shape[0]
    b = X_V.shape[0]
    brandth = np.min([h, b])
    sigmaMatrix = np.zeros((h, b))
    print('特征值1是', lambda_U)
    print('特征值2是', lambda_V)

    for i in range(brandth):
        sigmaMatrix[i, i] = np.sqrt(lambda_V[i]) if lambda_V[i]>0 else 0
    print("奇异值是", sigmaMatrix)
    print(rebuildMatrix(X_U, sigmaMatrix, X_V))
    return X_U, sigmaMatrix, X_V

def right(matrixA):
    X_U, lambda_U, X_V = np.linalg.svd(matrixA)
    X_V = np.transpose(X_V)
    h = X_U.shape[0]
    b = X_V.shape[0]
    brandth = np.min([h, b])
    sigmaMatrix = np.zeros((h, b))
    print("####################")
    print("标准的是",X_V)
    
#     print(X_V)
    print('奇异值是', lambda_U)
    for i in range(brandth):
        sigmaMatrix[i, i] = lambda_U[i]
    
#     print(rebuildMatrix(X_U, sigmaMatrix, X_V))
    #求A*transpose(A)和transpose(A)*A的特征值和特征向量，与numpy的结果比较
    matrixA_matrixAT = np.dot(matrixA, np.transpose(matrixA))
    matrixAT_matrixA = np.dot(np.transpose(matrixA), matrixA)
    lambda_U, X_U = np.linalg.eig(matrixA_matrixAT)
    lambda_V, X_V = np.linalg.eig(matrixAT_matrixA)
#     print("单独计算的左奇异矩阵是", X_U)
    print("特征值是", lambda_V)
    print("单独计算的右奇异矩阵是", X_V)

def SVD_v1(matrixA):
    #首先求A*transpose(A)和transpose(A)*A
    matrixA_matrixAT = np.dot(matrixA, np.transpose(matrixA))
#     print(np.transpose(matrixA), matrixA)
    matrixAT_matrixA = np.dot(np.transpose(matrixA), matrixA)
#     print(matrixA_matrixAT)

    lambda_V, X_V = np.linalg.eig(matrixAT_matrixA)
    lambda_V, X_V = sortByEigenValue(lambda_V, X_V)
    X_V = changeSignal(X_V)
    print(X_V)
    sigmas = lambda_V
#     sigmas = filter(lambda x: x>0.001, lambda_V)
    sigmas = list(map(lambda x: np.sqrt(x), sigmas))
#     sigmas = np.array(sigmas)
    X_U = np.zeros((matrixA.shape[0], len(sigmas)))
    print(X_U)

    for i in range(len(sigmas)):
        print(matrixA*X_V[:, i]/sigmas[i])
#     print(X_U)
    
 
A = np.array([[2,4],[1, 3],[0, 0],[0, 0]])
A = np.array([[0,1 ],[1, 1 ],[1, 0]])
# A = np.array([[1,1 ],[1, 1 ],[0, 0]])
# A = np.array([[4,4 ],[-3, 3 ]])
# A = np.array([[1,1 ],[0, 1 ],[1, 0]])
A = np.array([[0, 0, 0, 0],
           [0, 0, 0, 3],
           [0, 0, 0, 0],
           [3, 3, 4, 0]])
A = np.array([[ 0, 0, 0, 0, 5],
              [ 0, 0, 0, 0, 3],
              [0, 0, 0, 2, 2]])
# A = np.array([[0, 0, 0, 2, 2], [0, 0, 0, 3, 3], [0, 0, 0, 1, 1], [1, 1, 1, 0, 0], 
#               [2, 2, 2, 0, 0], [5, 5, 5, 0, 0], [1, 1, 1, 0, 0]])
# A = np.array([[1,1 ],[0, 1 ],[-1, 1]])
if __name__ == '__main__':
#     evec_u, eval_sigma1, evec_part_v = SVD(A)
#     right(A)
    SVD_v1(A)
#     print(evec_u, eval_sigma1, evec_part_v)
#     print(rebuildMatrix(evec_u, eval_sigma1, evec_part_v))
    
    