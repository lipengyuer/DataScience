import sklearn as sk
from sklearn import tree
import copy
from sklearn import datasets
from sklearn.model_selection import train_test_split
#基于sk的决策树，构建一个AdaBoosting框架

class AB():
    
    def __init__(self, tree_num=50):
        self.tree_num = tree_num
        self.tree_list = []#存储决策树
        self.weights_list = []#存储每个决策树的话语权大小
        self.create_trees()
    
    def create_trees(self):
        for _ in range(self.tree_num): 
            a_tree = tree.DecisionTreeClassifier()
            self.tree_list.append(a_tree)
            
    def fit(self, X, Y):
        X_new, Y_new = copy.deepcopy(X), copy.deepcopy(Y)
        for i in range(self.tree_num):
            self.tree_list[i].fit(X_new, Y_new)
            
            #检查错分样本情况，并计算acc
            wrong_sample_index = []
            pred_class = self.tree_list[i].predict(X)
            for j in range(len(Y)):
                if pred_class[j]!=Y[j]:
                    wrong_sample_index.append(j)
            accuracy = 1 - len(wrong_sample_index)/len(Y)
            print("第", i, "棵树的acc是", accuracy)
            self.weights_list.append(accuracy)
            for j in range(len(wrong_sample_index)):
                X_new.append(X[j])
                Y_new.append(Y[j])
    
    def predict(self, X):
        prediction_list = []
        for i in range(self.tree_num):
            pred = self.tree_list[i].predict(X)
            prediction_list.append(pred)
        prediction = []
        for sample_i in range(len(X)):
            labels_for_this_sample = []
            for tree_id in range(self.tree_num):
                labels_for_this_sample.append(prediction_list[tree_id][sample_i])
            final_label = self.get_final_class(labels_for_this_sample)
            prediction.append(final_label)
        return prediction
            
        
    def get_final_class(self, class_labels):
        label_weight_map = {}
        for tree_i in range(len(class_labels)):
            label = class_labels[tree_i]
            label_weight_map[label] = label_weight_map.get(label, 0) + self.weights_list[tree_i]
        label_weight_list = sorted(label_weight_map.items(), key=lambda x: x[1])
        final_label = label_weight_list[-1][0]
        return final_label
        
    def evaluate(self, X, Y):
        prediction = self.predict(X)
        error_count = 0
        for j in range(len(Y)):
            if prediction[j]!=Y[j]:
                error_count += 1
        accuracy = 1 - error_count/len(Y)
        print("测试集中的acc是", accuracy)        

if __name__ == '__main__':
    iris = datasets.load_digits()
    X = iris.data
    Y = iris.target
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.1)
    print(len(Y))
#     
    clf = AB(tree_num=1)
    clf.fit(trainX, trainY)
    clf.evaluate(testX, testY)
    