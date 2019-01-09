'''
Created on 2019年1月8日

@author: pyli
'''
#用rnn+softmax来做分类,这里使用鸢尾花数据集
import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
from matplotlib.pyplot import axis
import matplotlib.pyplot as plt
from _regex_core import name
hidden_size=10
batch_size_num = 10
stepNum = 5
classNum = 3
featureNum = 4
learningRate = 0.1

input_data = tf.placeholder(tf.float32, [None, 1, featureNum], name='input_data')#输入
batch_size = tf.shape(input_data)[0]#动态获取当前这批数据的大小
real_output = tf.placeholder(tf.float32, [None, classNum])#真实的输出值

weights_softmax = tf.Variable(tf.truncated_normal(shape = [hidden_size, classNum]))#softmax层的权重
bais_softmax = tf.Variable(tf.truncated_normal(shape = [classNum]))#softmax层的偏置

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
outputs_1, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                   initial_state=initial_state,
                                   dtype=tf.float32)
outputs_2 = outputs_1
outputs_3 = tf.nn.softmax(tf.matmul(outputs_2[:, -1,:],weights_softmax) + bais_softmax)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=real_output, logits=outputs_3))
train = tf.train.AdadeltaOptimizer(learning_rate=learningRate).minimize(loss)
label = tf.arg_max(outputs_3, 1, name='label')
correctPredNum = tf.equal(tf.arg_max(outputs_3, 1), tf.arg_max(real_output, 1))
accuracy = tf.reduce_mean(tf.to_float(correctPredNum))




fileName = '../../algorithm/iris.data'
with open(fileName, 'r') as f:
    lines = f.readlines()
    lines = list(map(lambda x: x.replace('\n', '').split(','), lines))
outputList = []
inputList = []
for line in lines[:-1]:
    label = line[-1]
    line = list(map(lambda x: float(x)/5, line[:-1]))
    if label=='Iris-virginica':
        outputList.append([1., 0., 0.])
    elif label=='Iris-setosa':
        outputList.append([0., 1., 0.])
    else:
        outputList.append([0., 0., 1.])
    inputList.append([line])
inputList, testInput, outputList, testOutput= train_test_split(inputList, outputList, test_size=0.1)
inputList = np.array(inputList)
outputList = np.array(outputList)
testInput = np.array(testInput)
testOutput = np.array(testOutput)

saver = tf.train.Saver(max_to_keep=4)


config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage
                inter_op_parallelism_threads = 1, 
                intra_op_parallelism_threads = 1,
                log_device_placement=True)
sess = tf.Session(config=config)
sess.run(tf.initialize_all_variables())

lossList = []
for epoch in range(10000):
    for i in range(0, len(inputList), batch_size_num):
        x_batch = inputList[i:i+batch_size_num, :]
        y_batch = outputList[i:i+batch_size_num, :]
        #     print(x_batch)
        #     print(y_batch)
        lossValue, _ = sess.run([loss, train], \
                                            feed_dict= {input_data: x_batch, real_output: y_batch})
#         print(epoch, lossValue)
        accuracyValue, lossValue = sess.run([accuracy, loss], \
                                    feed_dict= {input_data: testInput, real_output: testOutput})
        print(epoch, accuracyValue, lossValue)
        lossList.append(lossValue)
        saver.save(sess, 'model/mymodel', global_step=epoch)

plt.plot(lossList)
plt.show()
    