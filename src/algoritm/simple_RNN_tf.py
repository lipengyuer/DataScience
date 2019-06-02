'''
Created on 2019年6月2日

@author: Administrator
'''
#用tensorflow实现一个直白的RNN
#用这个RNN实现对正弦曲线的刻画：y_t = f(y_t-T-1, ..., y_t-1)
#网络结构是RNN层+多元线性回归。
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#超参数设置
batch_size = 11#训练的时候，一般使用批量梯度下降来加快收敛；有时候也会使用这个策略来避免内存或者现存不足导致的内存溢出
rnn_node_num = 50#RNN神经元个数
input_dim  =1#输入数据的维度
T =20#序列的长度，这里每次会向网络输入一个长20的序列。
output_dim = 1#输出的维度。
lr = 0.0001#学习率
step_num = 1#训练时,遍历数据集的次数

#生成训练数据，取值范围是[-1,1]。
def gernerate_data():
    data = [np.sin(i/20) for i in range(0, 1000)]
#     plt.plot(data)
#     plt.show()
    X = []
    Y = []
    for i in range(len(data) - T - 1):
        X.append(data[i: i  + T])#输入的序列
        Y.append(data[i + T])#输出的取值
    return np.array(X), np.array(Y)

class TFRNN():
    
    def __init__(self, rnn_node_num, input_dim, output_dim, T = 10, learning_rate = 0.001):
        self.T = T#循环的次数，即时间步数
        self.lr = learning_rate#学习率
        self.input_dim = input_dim#输入序列的元素的维度。假如输入是气温序列，那么元素就是一个个气温取值，维度是1；假如是字向量
        #元素就是一个个向量，维度是字向量的长度。
        self.rnn_node_num = rnn_node_num#RNN的神经元的个数。这里只有一层
        self.output_dim = output_dim
        #创建网络
        self.init_dynamic_graph()
    
    #初始化一个计算图
    def init_dynamic_graph(self):
        #设置计算图的输入
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.T, self.input_dim])
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.output_dim])
        self.batch_size = tf.placeholder(dtype=tf.float32, shape=[1])
        
        #设置RNN层的结构
        self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.rnn_node_num)
        init_state = tf.zeros(shape=[self.batch_size[0], self.rnn_cell.state_size])
        output, states = tf.nn.dynamic_rnn(self.rnn_cell, self.X, initial_state=init_state)

        #多元线性回归，把RNN层的self.rnn_node_num个输出综合起来
        self.W = tf.Variable(tf.random_normal([self.rnn_node_num, self.output_dim ]))
        self.b = tf.Variable(tf.zeros([self.output_dim]))
        self.final_output = tf.matmul(output[:, -1, :], self.W) + self.b#多元线性回归
        
        #把正确答案和模型的计算值都拉直，然后计算最小二成损失值
        Y_ = tf.reshape(self.Y, [-1])
        output_ = tf.reshape(self.final_output, [-1])
#         self.loss = -tf.reduce_mean(tf.multiply(output_, tf.log(Y_)))#分类时使用的损失函数
        self.loss = tf.reduce_mean(tf.square(output_ - Y_))#回归任务常用的损失函数
        
        #设置优化函数
        self.train = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        
        #启动上面定义的计算图
        self.sess=tf.Session()
        self.sess.run(tf.initialize_all_variables())

        
    def fit(self, X, Y):
        #把训练数据喂给计算图，驱动计算图的优化动作
        loss, _, final_output= self.sess.run([self.loss, self.train, self.final_output],\
                                              feed_dict={self.X: X, self.Y:Y, self.batch_size: [X.shape[0]]})
        return loss, final_output
    
    def predict(self, X):
        #把数据喂给计算图，把输出值获取出来
        final_output= self.sess.run([self.final_output], feed_dict={self.X: X, self.batch_size: [X.shape[0]]})
        return final_output   

import random
if __name__ == '__main__':
    print("生成训练数据")
    X, Y = gernerate_data()
    print("初始化模型")
    model = TFRNN(rnn_node_num, input_dim, output_dim, T=T ,learning_rate=lr)
    loss_list = []
    print("开始训练")
    for step in range(step_num):
        for i in range(0, X.shape[0] - batch_size, batch_size):
            X_batch = X[i: i + batch_size]
            Y_batch = Y[i: i + batch_size]
            X_batch = np.array(X_batch).reshape([-1, T, input_dim])
            Y_batch = np.array(Y_batch).reshape([-1, 1])
            
            loss_v, final_output = model.fit(X_batch, Y_batch)

            loss_list.append(loss_v)
    plt.plot(loss_list)
    plt.show()#展示损失值的变化趋势
        
