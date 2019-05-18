import pandas as pd
import numpy as np
rootPath = './data/'
data = pd.read_csv(rootPath + 'happiness_train_complete.csv')
data = data[data['happiness']>0]

def an(feature):
    # print(data['birth'])
    # data['birth'] = 2018 - data['birth']#.astype(int)
    print(data[['happiness', 'birth']].groupby('birth').agg(['min', 'max', 'mean']))
    print(data[['happiness', feature]].groupby([ feature]).agg(['min', 'max', 'mean']))
    print(data[['happiness', feature]].groupby([ feature]).count())

# an('political')
#print(data.groupby(['happiness']).count())
data = data[data['f_birth']<0]
print(min(data['f_birth']))
# print(len(data['marital_now']))
# print(data.isnull().any())
print(len(data))
