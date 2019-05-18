#对用户和商家进行画像。如果可能，对商品也进行画像。
import pandas as pd


def processLog(df):
    featureValues, featureNames = [], []
    logs = df['activity_log'].values
    data = []
    for log in logs: 
        if type(log)==str: data += log.split('#')
    data = list(map(lambda x: x.split(':'), data))
    
    
    #是否购买#####################################
    actionNum = len(data)
    timestamp_set =set()
    buyingNum, cartNum, click_num, favor_num = 0, 0, 0, 0
    for line in data: 
        if line[4] =='2': buyingNum += 1
        if line[4]=='1': cartNum+=1
        if line[4]=='0': click_num+=1
        if line[4]=='3': favor_num+=1
        timestamp_set.add(line[-2])
    day_num = len(timestamp_set)
    buy_favor_rate = buyingNum/(favor_num + 0.0000001)
    action_num_per_day = actionNum/(day_num + 0.0000001)
    click_num_per_day = click_num/(day_num + 0.0000001)
    buy_num_per_day = buyingNum/(day_num + 0.0000001)
    buying_prob = buyingNum/(click_num + 0.0000001)
    buying_prob_after_cart = buyingNum/(cartNum+0.0000001)
    buying_prob_after_click = buyingNum/(click_num+0.0000001)
    featureValues += [click_num_per_day, favor_num, cartNum, buyingNum, buying_prob, buying_prob_after_cart, 
                      buying_prob_after_click, action_num_per_day, buy_num_per_day,
            buy_favor_rate]
    featureNames += ['click_num_per_day', 'favor_num','cartNum','buyingNum', 'buying_prob', 'buying_prob_after_cart', 
                     'buying_prob_after_click', 'action_num_per_day'
                , 'buy_num_per_day', 'buy_favor_rate']
    
    
    
    #######商品类型####################################
    cat_num, brand_num,  cat_set, brand_set = 0, 0, set(), set()
    
    for line in data:
        if line[4] in ['2', '3']:
            cat_set.add(line[1])
            brand_set.add(line[2])
    cat_num, brand_num = len(cat_set), len(brand_set)
    featureValues += [cat_num, brand_num]
    featureNames += ['cat_num', 'brand_num']
    ##################
    return featureValues, featureNames

#对用户进行画像，并把结果写到csv文件中，便于后续进行联表操作
def personas4User(fileName):
    def stastic(data4user):
        
        merchant_num = data4user['merchant_id'].count()
        user_id = data4user['user_id'].iloc[0]
        featuresFromLog, logFeatureNames = processLog(data4user)
        res = [ [user_id, merchant_num/10] + featuresFromLog]
        res = pd.DataFrame(res, columns=['user_id', 'merchant_num'] + logFeatureNames)
        return res
    print("读取数据。")

    df = pd.read_csv(fileName)
    print("开始分组。")
    res = df.groupby(by=['user_id'])
    print("开始统计")
    res = res.apply(stastic)
#     print(df.drop(['activity_log'], axis=1))
#     print(res.reset_index(drop=True))
#     print(res['merchant_num'])
#     print(res.isnull())
    res = res.reset_index(drop=True)
    df = df.reset_index(drop=True).merge(res, on=['user_id'])
#     print(df.columns)
#     print(df.drop(['activity_log'], axis=1))
    return res


def personas4Merchant():
    pass

import pandas as pd
def get_merchant_id():
    trainDataFile = './data_format2/train_format2_sub.csv'
    df = pd.read_csv(trainDataFile)
    merchant_id_index = {}
    count = 0
    for merchant_id in df['merchant_id'].values:
        merchant_id_index[merchant_id] = count
        count += 1
    return merchant_id_index

if __name__ == '__main__':
    fileName = r"train_format2_sub.csv"
#     fileName = r"c:\\Users\\Administrator\\Desktop\\简单任务\回头客项目\\data_format2\\train_format2.csv"
#     res = personas4User(fileName)
#     print(res)
    a = get_merchant_id()
    print(a)
    
    
    