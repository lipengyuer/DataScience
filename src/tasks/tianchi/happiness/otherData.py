import  pandas as pd
import numpy as np

rootPath = './data/'

gdpDataFile = rootPath + '2016gdpData.txt'
code_province = """1 = 上海市; 2 = 云南省; 3 = 内蒙古自治区; 4 = 北京市; 5 = 吉林省; 6 = 四川省; 7 = 天津市; 8 = 宁夏回族自治区; 9 = 安徽省; 10 = 山东省; 11 = 山西省; 12 = 广东省; 13 = 广西壮族自治区; 14 = 新疆维吾尔自治区; 15 = 江苏省; 16 = 江西省; 17 = 河北省; 18 = 河南省; 19 = 浙江省; 20 = 海南省; 21 = 湖北省; 22 = 湖南省; 23 = 甘肃省; 24 = 福建省; 25 = 西藏自治区; 26 = 贵州省; 27 = 辽宁省; 28 = 重庆市; 29 = 陕西省; 30 = 青海省; 31 = 黑龙江省"""
province_code_map = {}
code_province = code_province.replace(' ', '').split(';')
for _ in code_province:
    code, province = _.split('=')
    province_code_map[province] = int(code)

with open(rootPath + 'regionNames.txt', 'r', encoding='utf8') as f:
    provinceNames = f.readlines()
    provinceNames = list(map(lambda x: x.replace('\n', '').split('，'), provinceNames))
shortProvinceName2Long = {}
for _ in provinceNames: shortProvinceName2Long[_[1]] = _[0]

def addOtherData(data):
    gdpData = processGDP()
    #rint(data.size, 'asd')
    data = pd.merge(data, gdpData, on=['province'], how='left')
    data = data.fillna(-1)
    data['popIndex'] = data['provinceGDP']/ data['perGDPEachProvince']
    data['incomeGap'] = data['income'] - data['perGDPEachProvince']
    data['incomeGapRate'] = data['income'] / data['perGDPEachProvince']
    data['familyIncomeGap'] = data['family_income']/ data['perGDPEachProvince']

    universityData = processUniversityData()
    universityData = universityData.reset_index(drop=True)
    data = pd.merge(data, universityData, on=['province'], how='left')
    # print(data.values.shape, 'qweqweqwe')
    incomeEachProvince = processPeopleIncome()

    data = pd.merge(data, incomeEachProvince, on=['province'], how='left')
    # print(incomeEachProvince.values.shape)
    # print(data['provinceAvIncome2015'])
    data['ratio2avIncome'] = data['income']/data['provinceAvIncome2014']
    data['incomeGap2Av'] = data['income'] - data['provinceAvIncome2014']
    data['inc_exp_gap_raio'] = (data['inc_exp'] - data['income'])/data['provinceAvIncome2014']
    data['familyIncomme2AvRate'] = data['provinceAvIncome2014']/data['family_income']

    housePrice = processHousePrice()
    data = pd.merge(data, housePrice, on=['province'], how='left')
    data['housePriceIncomeRate'] = data['housePrice']/data['income']
    data['if_house_exp_I'] = data['housePriceIncomeRate'].apply(lambda x: 1 if x<1 else 0)
    data['if_house_exp_II'] = data['housePriceIncomeRate'].apply(lambda x: 1 if 1<=x < 2 else 0)
    data['if_house_exp_III'] = data['housePriceIncomeRate'].apply(lambda x: 1 if 2<=x <5 else 0)
    data['if_house_exp_III'] = data['housePriceIncomeRate'].apply(lambda x: 1 if x > 5 else 0)
    data['housePriceAvGDPRate'] = data['housePrice'] / data['perGDPEachProvince']
    data['housePriceAvIncome'] = data['housePrice'] / data['provinceAvIncome2014']
    data['housePriceFamilyIncomeRate'] = data['housePrice'] /data['family_income']

    lifeLongData = processLifeLong()
    # print(lifeLongData[['realBirth', 'totalLifeLong']])
    #print('age is ', data['realBirth'])

    data = pd.merge(data, lifeLongData[['realBirth', 'totalLifeLong']], on=['realBirth'], how='left')
    # print(data[['realBirth', 'totalLifeLong']].describe())
    # print(lifeLongData[['realBirth', 'totalLifeLong']])
    # print(lifeLongData[lifeLongData['realBirth']==24])
    # print("1111111111111111111111111111")
    return data

def processGDP():

    with open(gdpDataFile, 'r', encoding='utf8') as f:
        gdpData = f.readlines()[1:]
    provinceGDPMap, perGDPEachProvinceMap = {}, {}
    for line in gdpData:
        _, province1, totalGDP, province2, GDPPer = line.split(' ')
        if province1 in province_code_map:
            code = province_code_map[province1]
            provinceGDPMap[code] = float(totalGDP)/1000#100billion
        if province2 in province_code_map:
            code = province_code_map[province2]
            perGDPEachProvinceMap[code] = float(GDPPer) / 10000  # 10000
    provinceGDP = pd.DataFrame(list(provinceGDPMap.items()), columns=['province', 'provinceGDP'])
    #print(provinceGDP)
    perGDPEachProvince = pd.DataFrame(list(perGDPEachProvinceMap.items()), columns=['province', 'perGDPEachProvince'])
    GDPDATA = pd.merge(provinceGDP, perGDPEachProvince, on=['province'])
    return GDPDATA

def processUniversityData():


    with open(rootPath + 'universityData.txt', 'r', encoding='utf8') as f:
        universityData = f.readlines()[1:]
    universityData = map(lambda x: x.split('	'), universityData)
    universityData = list(map(lambda x: [x[1],float(x[2])], universityData))
    universityData = pd.DataFrame(universityData, columns=['universityName', 'universityScore'])
    #print(universityData)
    with open(rootPath + 'universityRegion.txt', 'r', encoding='utf8') as f:
        universityRegion = f.readlines()[1:]
    universityRegion = map(lambda x: x.split('	'), universityRegion)
    universityRegion = list(map(lambda x: [x[0], shortProvinceName2Long[x[2]]], universityRegion))
    universityRegion = list(map(lambda x: [x[0], province_code_map[x[1]]], universityRegion))
    universityRegion = pd.DataFrame(universityRegion, columns=['universityName', 'province'])
    #print(universityRegion)
    universityDataDetail = pd.merge(universityData, universityRegion, on=['universityName'])
    #print(universityDataDetail)
    universityNum =  universityDataDetail.groupby('province').count()
    universityScore = universityDataDetail[['province', 'universityScore']].groupby('province').agg(['max', 'min', 'mean'])
    # print(universityScore['universityScore'])
    universityNum['universityNumInProvince'] = universityNum['universityName']
    universityNum['universitySocreInProvince_min'] = universityScore['universityScore']['min'].astype(float)
    universityNum['universitySocreInProvince_max'] = universityScore['universityScore']['max'].astype(float)
    universityNum['universitySocreInProvince_mean'] = universityScore['universityScore']['mean'].astype(float)

    universityNum = universityNum.drop(['universityName', 'universityScore' ], axis=1)
    universityNum = universityNum.assign(**universityNum.index.to_frame())

    #print(universityNum)
    return universityNum

def processPeopleIncome():
    with open(rootPath + 'income_each_province.txt', 'r', encoding='utf8') as f:
        data = f.readlines()
    data = list(map(lambda x: x.replace('\n', '').split('\t'), data))
    columns = data[0]
    data = list(map(lambda x: [province_code_map.get(x[0], -1)] + x[1:], data[1:]))
    data = list(filter(lambda x: x[0]!=-1, data))
    data = pd.DataFrame(data, columns=columns)
    data = data.drop(['2017', '2016'], axis=1)

    data['provinceAvIncome2013'] = data['2013'].astype(float)/10000
    data['provinceAvIncome2014'] = data['2014'].astype(float)/10000
    data['provinceAvIncomeChange2014'] = data['2014'].astype(float)/10000 - data['2013'].astype(float)/10000
    data['provinceAvIncomeChange2013'] = data['2013'].astype(float)/10000 - data['2012'].astype(float)/10000
    data['provinceAvIncomeChange'] = data['2014'].astype(float)/10000 - data['2008'].astype(float)/10000
    data = data[['province', 'provinceAvIncome2014', 'provinceAvIncome2013', 'provinceAvIncomeChange2014',
                 'provinceAvIncomeChange2013','provinceAvIncomeChange']]

    return data

def processHousePrice():
    with open(rootPath + 'housePrice.txt', 'r', encoding='utf8') as f:
        data = f.readlines()
    data = list(map(lambda x: x.replace('\n', '').split('\t'), data))
    data = list(map(lambda x: [province_code_map.get(shortProvinceName2Long[x[2]], -1), float(x[3])/10000], data[1:]))
    data = list(filter(lambda x: x[0]!=-1, data))
    data = pd.DataFrame(data, columns=['province', 'housePrice'])
    return data

def processLifeLong():
    data = pd.read_csv(rootPath + 'life_long_data.txt', delimiter='\t')
    data.loc[2, 'man'] = (66.28+69.30)/2
    data.loc[2, 'woman'] = (69.27 + 71.80)/2
    res = []
    for year in range(1900, 2015):
        if year < 1981:
            res.append(data.iloc[0].values)
            res[-1][1] =1
        if 1981<=year<1990:
            res.append(data.iloc[1].values)
            res[-1][1] = 2
        if 1990<=year<1996:
            res.append(data.iloc[2].values)
            res[-1][1] = 3
        if 1996<=year<2000:
            res.append(data.iloc[3].values)
            res[-1][1] = 4
        if 2000<=year<2005:
            res.append(data.iloc[4].values)
            res[-1][1] = 5
        if 2005<=year<=2010:
            res.append(data.iloc[5].values)
            res[-1][1] = 6
        if 2010<=year<=2015:
            res.append(data.iloc[6].values)
            res[-1][1] = 7
        res[-1][0] = year

    lifeLong = pd.DataFrame(res, columns = ['birth', 'totalLifeLong', 'manLifeLong', 'womanLifeLong'])
    lifeLong = lifeLong.astype(float)
    lifeLong['birth'] = 2015 - lifeLong['birth']
    lifeLong['realBirth'] = lifeLong['birth'].astype(int)
    return lifeLong


if __name__ == '__main__':

    #processGDP()
    #processUniversityData()
    #processHousePrice()
    processLifeLong()
