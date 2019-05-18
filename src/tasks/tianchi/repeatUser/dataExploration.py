import pandas as pd
import matplotlib.pyplot as plt
def show():
    fileName = 'C:\\Users\\Administrator\\Desktop\\简单任务\回头客项目\\用户涉足的商家数量分布.csv'
    df = pd.read_csv(fileName, sep='\t')
    df.plot.scatter(y='user_num', x='merchant_num')
    plt.show()
    
def whatIsTimestamp():
    fileName = r"c:\\Users\\Administrator\\Desktop\\简单任务\回头客项目\\data_format2\\train_format2_sub.csv"
    df = pd.read_csv(fileName)
#     print(df[df['label']==1])
    tempdf = df[df['user_id']==42393][['user_id', 'merchant_id', 'label', 'activity_log']]
    tempdf.to_csv('c:\\Users\\Administrator\\Desktop\\简单任务\回头客项目\\tempRes.csv', index=0)
    df['activity_log_detail'] = df['activity_log'].apply(lambda x: str(x).split('#')[0].split(':'))
    df['activity_log_detail'] = df['activity_log_detail'].apply(lambda x: -1 if len(x)!=5 else x)
    df = df[df['activity_log_detail']!=-1]
    df['activity_log_detail'] = df['activity_log_detail'].apply(lambda x: x[3]).astype(int)
    df['label'] = df['label'].astype(int)
   # print(df['activity_log_detail'].describe())
   # print('618时的购物', df[df['activity_log_detail']==418].count(), '双十一',  df[df['activity_log_detail']==1111].count())
    #print(df[df['label']==-1].count(), df[df['label']==0].count(), df[df['label']==1].count())
    

if __name__ == '__main__':
    #show()
    whatIsTimestamp()