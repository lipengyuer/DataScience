import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
import otherData

from sklearn.utils.class_weight import compute_class_weight

features = None

def  processHeight(h):
    if h<150: return 1
    elif 150<h<160: return 2
    elif 160<h<170: return 3
    elif 170<h<180: return 4
    else: return 5

def processWeight(w):
    if w<70: return 1
    elif 70<w<90: return 2
    elif 90<w<110: return 3
    elif 110<w<130: return 4
    else: return 5

def ifTooShort(gender, height):
    if gender==1:
        if height<165: return 1
        else: return 0
    else:
        if height<160: return 1
        else: return 0

def ifTooHight(gender, height):
    if gender==1:
        if height>195: return 1
        else: return 0
    else:
        if height<185: return 1
        else: return 0

def ifTooHeavy(gender, weight):
    if gender==1:
        if weight>200: return 1
        else: return 0
    else:
        if weight>140: return 1
        else: return 0

def ifTooLight(gender, weight):
    if gender==1:
        if weight<100: return 1
        else: return 0
    else:
        if weight<70: return 1
        else: return 0

def processAge(age):
    if age<25: return 1
    if 25<=age<30: return 2
    if 30<=age<40: return 3
    if 40 <= age < 60: return 4
    if 60 <= age < 70: return 5
    if age>=70: return 6

def featureEngineering(data):
    global features
    data['no_income'] = data['income'].apply(lambda x: 1 if x <0  else 0)
    data['income'] = data['income'].apply(lambda x: 1000000 if x > 1000000 else x)\
                             .apply(lambda x: 1000 if x <1000  else x)
    data['income'] = data['income'] / 10000
    data['inc_exp'] /= 10000
    data['inc_exp_gap'] = data['inc_exp']/data['income']
    data['inc_exp_gap_neg'] = data['inc_exp_gap'].apply(lambda x: 1 if x<1 else 0)
    data['no_s_income'] = data['s_income'].apply(lambda x: 1 if x < 0 else 0)
    data['s_income'] = data['s_income'].apply(lambda x: 1000000 if x>1000000 else x)\
                             .apply(lambda x: 10000 if x < 1000 else x)
    data['s_income'] = data['s_income'] / 10000
    data['ratio_s_income'] = data['income']/ data['s_income']
    data['no_family_income'] = data['family_income'].apply(lambda x: 1 if x < 0 else 0)
    data['family_income'] = data['family_income'].apply(lambda x: 1000000 if x>1000000 else x).\
                                                    apply(lambda x: 1000 if x<1000 else x)
    data['family_income'] = data['family_income'] / 10000
    data['ratio_income_in_family'] = data['income']*data['family_m']/data['family_income']
    data = data.fillna(-1)
    # data['s_birth'] = data['s_birth'].apply(lambda x: 2015 if x<1900 else x)
    # data['f_birth'] = data['f_birth'].apply(lambda x: 2015 if x < 1900 else x)
    # data['m_birth'] = data['m_birth'].apply(lambda x: 2015 if x < 1900 else x)
    data['birth'] = data['birth'].apply(lambda x: 2015 if x < 1900 else x)

    data['s_birth'] = data.apply(lambda x: x['birth'] if x['s_birth']<0 else x['s_birth'], axis=1)
    data['f_birth'] = data.apply(lambda x: x['birth']-20 if x['f_birth']<0 else x['f_birth'], axis=1)
    data['m_birth'] = data.apply(lambda x: x['birth']-20 if x['m_birth']<0 else x['m_birth'], axis=1)

    data['f_m_birth_delta'] = data['f_birth'] - data['m_birth']
    data['f_birth_delta'] = data['birth'] - data['f_birth']
    data['m_birth_delta'] = data['birth'] - data['m_birth']
    data['birth'] = 2015 - data['birth']
    data['s_birth'] = 2015 - data['s_birth']
    data['f_birth'] = 2015 - data['f_birth']
    data['m_birth'] = 2015 - data['m_birth']
    data['is_nianqingren'] = data['birth'].apply(lambda x: 1 if x < 25 and x>20 else 0)
    data['is_laoren'] = data['birth'].apply(lambda x: 1 if x > 70 else 0)
    data['if_retired'] = data['birth'] - 60
    data['if_retired'] = data['if_retired'].apply(lambda x: 0 if x < 0 else 1)
    data['s_birth'] = data['s_birth'].apply(lambda x: 100 if x > 90 else x)
    data['birth'] = data['birth'].apply(lambda x: 100 if x > 100 else x)
    data['s_birth'] = data['s_birth'].apply(processAge)
    data['birth'] = data['birth'].apply(processAge)

    data = data.drop(['f_birth', 's_birth'], axis=1)



    print(data['f_birth_delta'].describe())
    print(data[data['f_birth_delta']<0])
    data['inc_exp'] = data['inc_exp'].apply(lambda x: 10000 if x <0 else x)
    data['inc_exp'] = data['inc_exp']/10000

    data['inc_ability'] =  data['inc_ability'].apply(lambda x: 2 if x< 0 else x)
    data['marital_1st'] = data['marital_1st'].apply(lambda x: 2015 if x==9997 else x)
    data['marital_now'] = data['marital_now'].apply(lambda x: 2015 if x == 9997 else x)
    data['marital_1st'] = 2015 - data['marital_1st']
    data['marital_now'] = 2015 - data['marital_now']
    data['bmi'] = data['weight_jin'] / (data['height_cm'] * data['height_cm'] / 20000)
    data['is_too_short'] = data.apply(lambda x: ifTooShort(x.gender, x.height_cm), axis=1)
    data['is_too_heavy'] = data.apply(lambda x: ifTooHeavy(x.gender, x.weight_jin), axis=1)
    data['is_too_hight'] = data.apply(lambda x: ifTooHight(x.gender, x.height_cm), axis=1)
    data['is_too_light'] = data.apply(lambda x: ifTooLight(x.gender, x.weight_jin), axis=1)
    data['height_cm'] = data['height_cm'].apply(processHeight)
    data['weight_jin'] = data['weight_jin'].apply(processWeight)
    data['join_party'] = data['join_party'].apply(lambda x: 2015 if x==-1 else x)
    data['join_party'] = 2015 - data['join_party']
    data['join_party_if'] = data['join_party'].apply(lambda x: 1 if x ==0 else 0)
    data['house'] = data['house'].apply(lambda x: -1 if x==96 else x)
    data['family_income'] = data['family_income'].apply(lambda x: -1 if x == 99999996 else x)
    data['floor_area_per'] = data['floor_area']/data['family_m']

    data['class_change_before'] = data['class'] - data['class_10_before']
    data['class_change_after'] = data['class_10_after'] - data['class']
    data['class_change214'] = data['class'] - data['class_14']

    data['edu_yr'] = 2015- data['edu_yr']
    data['age_highest_edu'] = data['edu_yr'] - data['birth']
    data['f_edu'] = data['f_edu'].apply(lambda x: 1 if x < 0 else x)
    data['m_edu'] = data['m_edu'].apply(lambda x: 1 if x < 0 else x)
    data['edu'] = data['edu'].apply(lambda x: 1 if x < 0 else x)
    data['f_m_edu_gap'] = data['f_edu'] - data['m_edu']
    data['f_edu_gap'] = data['edu'] - data['f_edu']
    data['m_edu_gap'] = data['edu'] - data['m_edu']

    data['status_peer_low'] = data['status_peer'].apply(lambda x: 1 if x==3 else 0)
    data['status_peer_high'] = data['status_peer'].apply(lambda x: 1 if x == 1 else 0)
    data['status_3_before_low'] = data['status_3_before'].apply(lambda x: 1 if x == 3 else 0)
    data['status_3_before_high'] = data['status_3_before'].apply(lambda x: 1 if x == 1 else 0)
    data['inc_ability_in'] = data['inc_ability'].apply(lambda x: 1 if x == 4 else 0)

    data['trust_5_neg'] = data['trust_5'].apply(lambda x: 1 if x == 1 else 0)
    data['trust_8_neg'] = data['trust_5'].apply(lambda x: 1 if x == 1 else 0)
    data['trust_13_neg'] = data['trust_5'].apply(lambda x: 1 if x == 1 else 0)

    data['family_status_neg'] = data['family_status'].apply(lambda x: 1 if x == 1 else 0)
    data['first_job_age'] = data['birth'] - data['work_yr']
    data['equity_neg'] = data['equity'].apply(lambda x: 1 if x == 1 else 0)
    data['depression_neg'] = data['depression'].apply(lambda x: 1 if x == 1 else 0)
    data['body_metal_neg'] = data.\
                         apply(lambda x: 1 if x['depression'] == 1 and x['health_problem']==1 else 0,\
                               axis=1)
    data['leisure'] = data.apply(lambda x: 1 if x['leisure_1'] + x['leisure_2'] +\
            x['leisure_3'] + x['leisure_4'] + x['leisure_5'] < 10 else 0, axis=1)
    pub_servce_fs = ['public_service_1','public_service_2','public_service_3','public_service_4','public_service_5','public_service_6','public_service_7','public_service_8','public_service_9']
    data[pub_servce_fs] = data[pub_servce_fs].applymap(lambda x: 50 if x < 0 else x)
    data[pub_servce_fs] /= 20
    data[pub_servce_fs] = data[pub_servce_fs].applymap(lambda x: int(x))
    trustFs = ['trust_' + str(i) for i in range(1, 14)]
    data[trustFs] = data[trustFs].applymap(lambda x: 1 if x<0 else x)
    stopFeatures = ['invest_other', 'property_other', 'edu_other', 'survey_time',
                    'city', 'county', 'nationality']




    data = data.drop(stopFeatures, axis=1)

    # data = data.applymap(lambda x: 0 if type(x)==str else x in y)
    #print(data.dtypes)
    #print(data.values.shape, 'qwe')
    data = otherData.addOtherData(data)
    # features = ['income', 'gender', 'religion', 'edu']

    if features==None:
        features = []
        for featureName in data.columns:
            #print(featureName, data.dtypes[featureName])
            # if data.dtypes[featureName]==int:
           # if 'income'  in featureName or 'GDP'  in featureName:
                features.append(featureName)
    # print(features)
    # features = ['province', 'gender', 'religion', 'religion_freq', 'edu', 'edu_status', 'edu_yr', 'income', 'political', 'join_party', 'floor_area', 'property_0', 'property_1', 'property_2', 'property_3', 'property_4', 'property_5', 'property_6', 'property_7', 'property_8', 'height_cm', 'weight_jin', 'health', 'health_problem', 'depression', 'hukou', 'hukou_loc', 'media_1', 'media_2', 'media_3', 'media_4', 'media_5', 'media_6', 'leisure_1', 'leisure_2', 'leisure_3', 'leisure_4', 'leisure_5', 'leisure_6', 'leisure_7', 'leisure_8', 'leisure_9', 'leisure_10', 'leisure_11', 'leisure_12', 'socialize', 'relax', 'learn']

    data = data[features]

    #otherData

    return data

def stastics(data):
     print(data.groupby(['happiness']).count())


def loadData(fileName):
    data = pd.read_csv(fileName)
    data = data.drop(['id'], axis=1)
    stastics(data)
    data = data[data['happiness']>0]
    #print(data[['f_birth']])

    data = data.fillna(-1)
    y = data[['happiness']]
    data = data.drop(['happiness'], axis=1)

    x = featureEngineering(data)
    print("age is : ", x['birth'].describe())
    # print(x.columns)
    # trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2)
    return x,y#trainX, testX, trainY, testY

def loadContestData(fileName):
    data = pd.read_csv(fileName)
    res = data[['id']]
    data = data.drop(['id'], axis=1)
    data = featureEngineering(data)
    return res, data

def evaluation(pred, y):
    count = 0
    cost = 0
    for i in range(len(y)):
        if pred[i]==y[i][0]: count += 1
        cost += (pred[i]-y[i][0])**2
        #print(pred[i], y[i][0])
    cost /= len(y)
    #print(count/len(y))
    return count/len(y), cost

from sklearn.tree import DecisionTreeClassifier
import time
def KFoldTest(trainX, trainY):
    kf = KFold(n_splits=10, random_state=int(time.time()))
    totalAcc = 0
    totalTrainingAcc = 0
    cost = 0
    trainingCost = 0
    for trainIndex, testIndex in kf.split(trainX):
        # print(trainX.size, trainY.size)
        trainInput, trainOutput = trainX.iloc[trainIndex], trainY.iloc[trainIndex]
        # print(trainInput['edu_yr'])
        testInput, testOutput = trainX.iloc[testIndex], trainY.iloc[testIndex]
        weight = compute_class_weight('balanced',[1,2,3,4,5], list(map(lambda x: x[0], trainOutput.values)))
        weight = [[i+1, weight[i]] for i in range(len(weight))]
        weight = dict(weight)
        weight = {1:1/104, 2:1/497, 3:1/1159, 4:1/4818, 5:1/1410}
        for key in weight: weight[key] = weight[key]**0.5
        clf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                     class_weight=weight, n_jobs=8)
        # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
        #                          algorithm="SAMME",
        #                          n_estimators=200)
        # clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=.01, max_depth=10,
        #                                  random_state=int(time.time()), max_features=0.3,\
        #                                  min_samples_leaf=20, subsample=0.9)
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes = (30, ), random_state = 1, max_iter=1000)
        clf.fit(trainInput, trainOutput.values)

        pred = clf.predict(trainInput)
        #print(pred)
        # for n in trainOutput.values: print(n)
        trainacc, traincostn = evaluation(pred, trainOutput.values)

        pred = clf.predict(testInput)
        # print(pred)
        acc, costn = evaluation(pred, testOutput.values)
        print("training acc", trainacc, traincostn,trainOutput.size,  'testing acc', acc, costn ,testOutput.size)
        totalAcc += acc
        cost += costn
        totalTrainingAcc += trainacc
        trainingCost += traincostn
    print("k-fold crossvalidation:", totalAcc/10, 'cost is', cost/10)
    print("in training is ", totalTrainingAcc/10, trainingCost/10)


if __name__ == '__main__':
    rootPath = '/opt/dev/tianchi/happiness/'
    trainX, trainY = loadData(rootPath + 'happiness_train_complete.csv')
    #print(trainX)
    # trainX, trainY = loadData(rootPath + 'happiness_train_abbr.csv')
    # trainX = abs(trainX)
    # selector = SelectKBest(chi2, k=80)  # Ñ¡Ôñk¸ö×î¼ÑÌØÕ÷
    # selector.fit(trainX, trainY)
    # trainX = selector.transform(trainX)
    KFoldTest(trainX, trainY)
   #  print("train size:", trainX.size)
   #  # clf = RandomForestClassifier(n_estimators=100, max_depth=1)
   #  clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.01, max_depth=10,
   #                                   random_state=int(time.time()), max_features=0.5, \
   #                                   min_samples_leaf=2, subsample=0.9)
   #  #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (500, 2), random_state = 1)
   #  clf.fit(trainX, trainY)
   # # print(data)
   #  res, contData = loadContestData(rootPath + 'happiness_test_complete.csv')
   #  # contData = selector.transform(contData)
   #  labels = clf.predict(contData)
   #  res['happiness'] = labels
   #  res = res[['id', 'happiness']]
   #  res.to_csv(rootPath + 'myRes.csv', index=0)


    """happiness              ...                         
1           104        ...                      104
2           497        ...                      497
3          1159        ...                     1159
4          4818        ...                     4818
5          1410        ...                     1410
find happiness==4 at first ...
"""

