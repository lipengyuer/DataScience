import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import numpy as np
import pylab
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import axis
import sklearn

class LSTMClassifier():
    def __init__(self, class_num, input_size, timestep_size=20, hidden_size=50, layer_num=1, learning_rate=1e-3, ):
        self.lr = learning_rate
        # 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
        self.batch_size = tf.placeholder(tf.int32, [])
        # 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
        self.input_size = input_size
        # 时序持续长度为28，即每做一次预测，需要先输入28行
        self.timestep_size = timestep_size
        # 每个隐含层的节点数
        self.hidden_size = hidden_size
        # LSTM layer 的层数
        self.layer_num = layer_num
        # 最后输出分类类别数量，如果是回归预测的话应该是 1
        self.class_num = class_num
        config = None
        sess = None
        self.learning_rate = None
        self.global_step = None

    def initGraph(self, ifDecrLR=True):
        self.config = tf.ConfigProto()
        self.sess = tf.Session(config=self.config)
        self._X = tf.placeholder(tf.float32, [None, self.input_size])
        self.y = tf.placeholder(tf.float32, [None, self.class_num])
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.X = tf.reshape(self._X, [-1, 1, self.input_size])

        # 调用 MultiRNNCell 来实现多层 LSTM
        self.mlstm_cell = rnn.MultiRNNCell([self.unit_lstm(self.hidden_size) for i in range(self.layer_num)],
                                           state_is_tuple=True)
        # 用全零来初始化state
        self.init_state = self.mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.outputs, self.state = tf.nn.dynamic_rnn(self.mlstm_cell, inputs=self.X, initial_state=self.init_state,
                                                     time_major=False)
        self.h_state = self.outputs[:, -1, :]  # 或者 h_state = state[-1][1]

        self.W = tf.Variable(tf.truncated_normal([self.hidden_size, self.class_num], stddev=0.1), dtype=tf.float32)
        self.bias = tf.Variable(tf.random_normal(shape=[self.class_num]), dtype=tf.float32)
        # self.bias = tf.Variable(tf.constant(0.1, shape=[self.class_num]), dtype=tf.float32)
        self.y_pre = tf.nn.softmax(tf.matmul(self.h_state, self.W) + self.bias)
        # 损失和评估函数

        self.cross_entropy = -tf.reduce_mean(self.y * tf.log(self.y_pre))
        # self.cross_entropy = -tf.reduce_sum(self.y * tf.log(self.y_pre))

        # self.cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pre, labels=self.y))
        #self.train_op = tf.train.MomentumOptimizer(momentum=1,learning_rate=self.lr).minimize(self.cross_entropy)

        if ifDecrLR==True:
            # print("学习率", self.learning_rate)
            batch = tf.shape(self._X)[0]
            self.global_step = tf.Variable(0)
            self.learning_rate = tf.train.exponential_decay(self.lr, self.global_step, decay_steps=self.input_size / batch,
                                                            decay_rate=1,
                                                            staircase=True)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy, global_step=self.global_step)
        else:
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.y_pre, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.sess.run(tf.global_variables_initializer())

    def unit_lstm(self, hidden_size):
        # 定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
        lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=0.6, state_is_tuple=True)
        # 添加 dropout layer, 一般只设置 output_keep_prob
        lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=0.6, output_keep_prob=0.6)
        return lstm_cell

    def fit(self, X, Y):
        self.sess.run(self.train_op, feed_dict={self._X: X, self.y: Y,
                                                self.keep_prob: 0.6, self.batch_size: len(Y)})
        #print("参数值是", self.W)

    def test(self, X, Y):
        train_accuracy, cost, y_pre = self.sess.run([self.accuracy, self.cross_entropy, self.y_pre], feed_dict={
            self._X: X, self.y: Y, self.keep_prob: 0.6, self.batch_size: len(Y)})
        # 已经迭代完成的 epoch 数: mnist.train.epochs_completed
        print("training accuracy %f, cost is %f" % (train_accuracy, cost))
        return y_pre

    def initOneHotEncoder4Y(self, YSample):
        self.oneHotEncoder4Y = OneHotEncoder().fit(YSample)

    def oneHotEncode(self, Y):
        batch_ys = self.oneHotEncoder4Y.transform(Y).todense().astype(np.float32)
        return batch_ys


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import random

    iris = load_iris()
    X, Y = iris.data, iris.target
    Y = list(map(lambda x: [x], Y))
    Y = np.array(Y)
    oneHotEncoder4Y = OneHotEncoder().fit(Y)

    indexList = list(range(len(Y)))
    random.shuffle(indexList)
    trainX, testX = X[indexList[:100], :], X[indexList[100:], :]
    trainY, testY = Y[indexList[:100], :], Y[indexList[100:], :]
    #     print("测试数据是", testY)

    #     print(oneHotEncoder4Y.transform(Y).todense())
    num_feature = len(X[0])
    print("特征的数量是", num_feature)
    clf = LSTMClassifier(3, num_feature)
    clf.initGraph()
    clf.initOneHotEncoder4Y(trainY)
    count = 0
    for i in range(150):
        #         print("本批数据的量是", _batch_size)
        #         batch_ys = np.array(trainY).reshape(_batch_size,1)
        batch_ys = clf.oneHotEncode(trainY)
        batch_xs = np.array(trainX).astype(np.float32)
        clf.fit(batch_xs, batch_ys)

        batch_ys = clf.oneHotEncode(testY)
        batch_xs = np.array(testX).astype(np.float32)
        clf.test(batch_xs, batch_ys)