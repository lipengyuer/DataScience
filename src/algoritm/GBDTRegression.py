from CARTRegression import CARTRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#一个非常简单的GBDT，只有残差拟合模块。
class GBDT():
    
    def __init__(self, max_tree_num=3):
        self.tree_list = []
        self.max_tree_num = max_tree_num
        
    def fit(self, x, y):
        residual = y
        for i in range(self.max_tree_num):
            model = CARTRegression()
            model.fit(x, residual)
            self.tree_list.append(model)
            prediction = model.predict(x)
            residual = residual - prediction
    
    def predict(self, x):
        y = np.zeros(x.shape[0])
        for model in self.tree_list:
            new_pred = np.array(model.predict(x))
            y += new_pred
            print('new_pred', new_pred)
        return y

def MSE(prediction, real_value):
    r = 0
    for i in range(len(prediction)):
        r += (prediction[i]-real_value[i])**2
    return r

if __name__ == '__main__':
    data = pd.read_excel("ENB2012_data2_example.xlsx")
    data = data.sample(frac=1)
    data = data.values
    X, Y = data[: ,0: -2], data[:, -2]
#     plt.plot(X[:,0])
#     plt.show()
#     print(X)
    
    #训练模型
    model = GBDT(max_tree_num=3)#变化CART的个数，可以看到拟合误差(残差平方和降低了一点)
    model.fit(X, Y)
    
    #测试
    prediction = model.predict(X)
    print(prediction)
    res = [Y[i]-prediction[i] for i in range(len(prediction))]
    print("残差平方和是", MSE(prediction, Y))
    plt.plot(Y)
    plt.plot(prediction)
    plt.show()
    
