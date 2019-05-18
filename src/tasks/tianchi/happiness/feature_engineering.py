import pandas as pd
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
import otherData
import itsData
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV

features = None


def findNearInt(f):
    anInt = int(f)
    gaps = [anInt - f, anInt + 1 - f]
    if np.abs(gaps[0]) < np.abs(gaps[1]):
        return anInt
    else:
        return anInt + 1

def processHeight(h):
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

def cal_BMI_index(h, w):
    h = h/100
    w = w/2
    if h<1 or h>3: h=1.7
    if w<30 or w>150: w=75
    bmi = w/h**2
    return bmi
    
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

def inf_edu_year(birth, edu_year):
    res = birth+18 if edu_year < 0 else edu_year
    return res
     
def featureEngineering(data):
    global features
    data = data.fillna(-1)
    data['no_income'] = data['income'].apply(lambda x: 1 if x <0  else 0)
    data['income'] = data['income'].apply(lambda x: 1000000 if x > 1000000 else x)\
                             .apply(lambda x: 5000 if x <1000  else x)
    data['income'] = data['income'] / 10000
    data['inc_exp'] /= 10000
    data['inc_exp_gap'] = data['inc_exp']/data['income']
    data['inc_exp_gap_rate'] = (data['inc_exp']-data['income'])/data['income']
    data['inc_exp_gap_neg'] = data['inc_exp_gap'].apply(lambda x: 1 if x<1 else 0)
    data['no_s_income'] = data['s_income'].apply(lambda x: 1 if x < 0 else 0)
    data['s_income'] = data['s_income'].apply(lambda x: 1000000 if x>1000000 else x)\
                             .apply(lambda x: 10000 if x < 1000 else x)
    data['s_income'] = data['s_income'] / 10000
    data['ratio_s_income'] = data['income']/ data['s_income']
    data['gap_s_income'] = data['income'] - data['s_income']
    data['ratio_gap_s_income'] = (data['income'] - data['s_income'])/data['income']
    
    data['no_family_income'] = data['family_income'].apply(lambda x: 1 if x < 0 else 0)
    data['family_income'] = data['family_income'].apply(lambda x: 1000000 if x>1000000 else x).\
                                                    apply(lambda x: 1000 if x<1000 else x)
    data['family_income'] = data['family_income'] / 10000
    data['gap_family_income'] = data['income']-data['family_income']
    data['ratio_gap_family_income'] = (data['income']-data['family_income'])/data['family_income']
    data['ratio_income_in_family'] = data['income']/data['family_income']
    data['gap_family_income_plas_s'] = data['income'] + data['s_income'] -data['family_income']
    data['ratio_gap_family_income_plas_s'] = (data['income'] + data['s_income'] -data['family_income'])/data['family_income']
    data['ratio_income_in_family_plas_s'] = (data['income'] + data['s_income'])/data['family_income']
    data['ratio_income_in_family_plas_s'] = data['ratio_income_in_family_plas_s'].apply(lambda x: 1 if x>1 else x)
    data['ratio_income_in_family'] = data['ratio_income_in_family'].apply(lambda x: 1 if x>1 else x)

    data = data.fillna(-1)
    data['birth'] = data['birth'].apply(lambda x: 1980 if x < 1900 else x)

    data['s_birth'] = data.apply(lambda x: x['birth'] if x['s_birth']<0 else x['s_birth'], axis=1)
    data['f_birth'] = data.apply(lambda x: x['birth']-23 if x['f_birth']<0 else x['f_birth'], axis=1)
    data['m_birth'] = data.apply(lambda x: x['birth']-23 if x['m_birth']<0 else x['m_birth'], axis=1)
    data['f_birth'] = data.apply(lambda x: x['birth']-23 if abs(x['birth']-x['f_birth'])>50\
                                      else x['f_birth'], axis=1)
    data['m_birth'] = data.apply(lambda x: x['birth']-23 if abs(x['birth']-x['m_birth'])>50\
                                      else x['m_birth'], axis=1)
    data['f_m_birth_delta'] = data['f_birth'] - data['m_birth']
    data['f_birth_delta'] = data['birth'] - data['f_birth']
    data['f_birth_delta'] = data['f_birth_delta'].apply(lambda x: 20 if x<10 else x)
    data['m_birth_delta'] = data['birth'] - data['m_birth']
    data['m_birth_delta'] = data['m_birth_delta'].apply(lambda x: 20 if x<10 else x)
    data['birth'] = 2015 - data['birth']
    data['age'] = data['birth']
    data['s_birth'] = 2015 - data['s_birth']
    data['f_birth'] = 2015 - data['f_birth']
    data['m_birth'] = 2015 - data['m_birth']
    data['is_nianqingren'] = data['birth'].apply(lambda x: 1 if x < 25 and x>20 else 0)
    data['is_laoren'] = data['birth'].apply(lambda x: 1 if x > 70 else 0)
    data['if_retired'] = data['birth'] - 60
    data['if_retired'] = data['if_retired'].apply(lambda x: 0 if x < 0 else 1)
    data['s_birth'] = data['s_birth'].apply(lambda x: 100 if x > 90 else x)
    data['birth'] = data['birth'].apply(lambda x: 100 if x > 100 else x)
    data['realBirth'] = data['birth']
    data['s_birth_stage'] = data['s_birth'].apply(processAge)
    data['birth_stage'] = data['birth'].apply(processAge)

    data = data.drop(['f_birth', 's_birth'], axis=1)

    data['inc_ability'] =  data['inc_ability'].apply(lambda x: 2 if x< 0 else x)
    data['marital_1st'] = data['marital_1st'].apply(lambda x: 2015 if x==9997 or x<0 else x)
    data['marital_now'] = data['marital_now'].apply(lambda x: 2015 if x == 9997 or x<0 else x)
    data['marital_1st'] = data['marital_1st'].apply(lambda x: x if x <100 else 2015-x)
    data['marital_now'] = data['marital_now'].apply(lambda x: x if x <100 else 2015-x)
    data['marital_1st'] = data['marital_1st'].apply(lambda x: 0 if x <0 else x)
    data['marital_now'] = data['marital_now'].apply(lambda x: 0 if x <0 else x)
    data['is_too_short'] = data.apply(lambda x: ifTooShort(x.gender, x.height_cm), axis=1)
    data['is_too_heavy'] = data.apply(lambda x: ifTooHeavy(x.gender, x.weight_jin), axis=1)
    data['is_too_hight'] = data.apply(lambda x: ifTooHight(x.gender, x.height_cm), axis=1)
    data['is_too_light'] = data.apply(lambda x: ifTooLight(x.gender, x.weight_jin), axis=1)
    data['bmi_index'] = data.apply(lambda x: cal_BMI_index(x.height_cm, x.weight_jin), axis=1)
    data['bmi_index_too_high'] = data['bmi_index'].apply(lambda x: 1 if x>30 else 0)
    data['bmi_index_too_low'] = data['bmi_index'].apply(lambda x: 1 if x<18 else 0)
    data['height_cm'] = data['height_cm'].apply(processHeight)
    data['weight_jin'] = data['weight_jin'].apply(lambda x: x*2 if x<80 else x)
    data['weight_jin'] = data['weight_jin'].apply(processWeight)
    data['join_party'] = data['join_party'].apply(lambda x: 2015 if x<1900 else x)
    data['join_party'] = 2015 - data['join_party']
    print("最大的党龄是", max(data['join_party']))
    data['join_party_if'] = data['join_party'].apply(lambda x: 1 if x ==0 else 0)
    data['house'] = data['house'].apply(lambda x: -1 if x==96 else x)
    data['family_m']  = data['family_m'].apply(lambda x: 2 if x<0  else x) 
    data['family_m']  = data['family_m'].apply(lambda x: 10 if x>10  else x) 
    data['floor_area'] = data['floor_area'].apply(lambda x: 50 if x<0 else x)
    data['floor_area_per'] = data['floor_area']/data['family_m']
    data['floor_area_big_house'] = data['floor_area'].apply(lambda x: 1 if x>140 else 0)
    data['floor_area_small_house'] = data['floor_area'].apply(lambda x: 1 if x<20 else 0)
    data['floor_area_small'] = data['floor_area_per'].apply(lambda x: 1 if x<5 else 0)
    
    data['class'] = data['class'].apply(lambda x: 6 if x <0 else x) 
    data['class_10_before'] = data['class_10_before'].apply(lambda x: 6 if x <0 else x) 
    data['class_10_after'] = data['class_10_after'].apply(lambda x: 6 if x <0 else x) 
    data['class_14'] = data['class_14'].apply(lambda x: 6 if x <0 else x) 
    data['class_change_before'] = data['class'] - data['class_10_before']
    data['class_change_after'] = data['class_10_after'] - data['class']
    data['class_change214'] = data['class'] - data['class_14']

    data['edu_yr'] = data.apply(lambda x:inf_edu_year(x.birth, x.edu_yr), axis=1)
    data['high_edu_1'] = data['edu_yr'].apply(lambda x: 1 if x>13 else 0)
    data['high_edu_2'] = data['edu_yr'].apply(lambda x: 1 if x>15 else 0)
    data['age_highest_edu'] = data['realBirth'] - data['edu_yr']
    data['f_edu'] = data['f_edu'].apply(lambda x: 6 if x < 0 or x==14 else x)
    data['m_edu'] = data['m_edu'].apply(lambda x: 6 if x < 0 or x==14 else x)
    
    data['f_high_edu'] = data['f_edu'].apply(lambda x: 1 if x in [12, 13] else 0)
    data['m_high_edu'] = data['m_edu'].apply(lambda x: 1 if x in [12, 13] else 0)
        
    data['edu'] = data['edu'].apply(lambda x: 4 if x < 0 or x==14 else x)
    data['edu_status'] = data['edu_status'].apply(lambda x: 4 if x<0 else x)
    data['edu_status_yiwujiaoyu_lost'] = data.apply(lambda x: 1 if x.edu<5 and x.edu_status in [2, 3] else 0, axis=1)
    data['edu_status_gaozhong_lost'] = data.apply(lambda x: 1 if 5<=x.edu<10 and x.edu_status in [2, 3] else 0, axis=1)
    data['edu_status_daxue_lost'] = data.apply(lambda x: 1 if 10<=x.edu<13 and x.edu_status in [2, 3] else 0, axis=1)
    data['edu_status_master_lost'] = data.apply(lambda x: 1 if x.edu==13 and x.edu_status in [2, 3] else 0, axis=1)
    data['edu_status_yiwujiaoyu_got'] = data.apply(lambda x: 1 if x.edu<5 and x.edu_status==4 else 0, axis=1)
    data['edu_status_gaozhong_got'] = data.apply(lambda x: 1 if 5<=x.edu<10 and x.edu_status==4 else 0, axis=1)
    data['edu_status_daxue_got'] = data.apply(lambda x: 1 if 10<=x.edu<13 and x.edu_status==4 else 0, axis=1)
    data['edu_status_master_got'] = data.apply(lambda x: 1 if x.edu==13 and x.edu_status==4 else 0, axis=1)

    data['f_m_edu_gap'] = data['f_edu'] - data['m_edu']
    data['f_edu_gap'] = data['edu'] - data['f_edu']
    data['m_edu_gap'] = data['edu'] - data['m_edu']
    data['if_evening_school'] = data['edu_other'].apply(lambda x: 1 if x=='夜校' else 0)
    data['s_edu'] = data['s_edu'].apply(lambda x: 6 if x < 0 else x)

    data['status_peer_low'] = data['status_peer'].apply(lambda x: 1 if x==3 else 0)
    data['status_peer_high'] = data['status_peer'].apply(lambda x: 1 if x == 1 else 0)
    data['status_3_before_low'] = data['status_3_before'].apply(lambda x: 1 if x == 3 else 0)
    data['status_3_before_high'] = data['status_3_before'].apply(lambda x: 1 if x == 1 else 0)
    data['inc_ability_in'] = data['inc_ability'].apply(lambda x: 1 if x == 4 else 0)

    data['trust_5_neg'] = data['trust_5'].apply(lambda x: 1 if x == 1 else 0)
    data['trust_8_neg'] = data['trust_5'].apply(lambda x: 1 if x == 1 else 0)
    data['trust_13_neg'] = data['trust_5'].apply(lambda x: 1 if x == 1 else 0)

    data['family_status_neg'] = data['family_status'].apply(lambda x: 1 if x == 1 else 0)
    data['first_job_age'] = data['realBirth'] - data['work_yr']
    data['equity_neg'] = data['equity'].apply(lambda x: 1 if x == 1 else 0)
    data['depression_neg'] = data['depression'].apply(lambda x: 1 if x == 1 else 0)
    data['body_metal_neg'] = data.\
                         apply(lambda x: 1 if x['depression'] == 1 and x['health_problem']==1 else 0,\
                               axis=1)
    data['leisure'] = data.apply(lambda x: 1 if x['leisure_1'] + x['leisure_2'] +\
            x['leisure_3'] + x['leisure_4'] + x['leisure_5'] < 10 else 0, axis=1)
    leisure_type = ['leisure_1','leisure_2','leisure_3','leisure_4','leisure_5','leisure_6',
                    'leisure_7','leisure_8','leisure_9','leisure_10','leisure_11','leisure_12']
    data[leisure_type] = data[leisure_type].applymap(lambda x: 3 if x<0 else x)

    pub_servce_fs = ['public_service_1','public_service_2','public_service_3','public_service_4','public_service_5','public_service_6','public_service_7','public_service_8','public_service_9']
    
    data[pub_servce_fs] = data[pub_servce_fs].applymap(lambda x: 50 if x < 0 else x)
    data[pub_servce_fs] /= 20
    data[pub_servce_fs] = data[pub_servce_fs].applymap(lambda x: int(x))
    trustFs = ['trust_' + str(i) for i in range(1, 14)]
    data[trustFs] = data[trustFs].applymap(lambda x: 1 if x<0 else x)
    
    print("正在处理房产情况。")
    property_other_clf = itsData.property_other_data()
    data['property_other_info'] = data['property_other'].apply(property_other_clf.predict)
    data['invest_other_info'] = data['invest_other'].apply(lambda x: 1 if type(x)==str else 0)

    data['religion_freq'] = data['religion_freq'].apply(lambda x: 0 if x<0 else x)

  
    data['political_people'] = data['political'].apply(lambda x: 1 if x<3 else 0)
    data['political_party'] = data['political'].apply(lambda x: 1 if x==4 else 0)
    data['political_minzhu'] = data['political'].apply(lambda x: 1 if x==3 else 0)
    data['s_political_people'] = data['s_political'].apply(lambda x: 1 if x<3 else 0)
    data['s_political_party'] = data['s_political'].apply(lambda x: 1 if x==4 else 0)
    data['s_political_minzhu'] = data['s_political'].apply(lambda x: 1 if x==3 else 0)
    data['f_political_people'] = data['f_political'].apply(lambda x: 1 if x<3 else 0)
    data['f_political_party'] = data['f_political'].apply(lambda x: 1 if x==4 else 0)
    data['f_political_minzhu'] = data['f_political'].apply(lambda x: 1 if x==3 else 0)
    data['m_political_people'] = data['m_political'].apply(lambda x: 1 if x<3 else 0)
    data['m_political_party'] = data['m_political'].apply(lambda x: 1 if x==4 else 0)
    data['m_political_minzhu'] = data['m_political'].apply(lambda x: 1 if x==3 else 0)

    #小时候服务的工作情况
    data['f_work_14_qushi'] = data['f_work_14'].apply(lambda x: 1 if x==16 else 0)
    data['m_work_14_qushi'] = data['m_work_14'].apply(lambda x: 1 if x==16 else 0)
    data['f_work_14_boss'] = data['f_work_14'].apply(lambda x: 1 if x==10 else 0)
    data['m_work_14_boss'] = data['m_work_14'].apply(lambda x: 1 if x==10 else 0)
    data['f_work_14_worker'] = data['f_work_14'].apply(lambda x: 1 if x<6 else 0)
    data['m_work_14_worker'] = data['m_work_14'].apply(lambda x: 1 if x<6 else 0)
    data['f_work_14_cannot_work'] = data['f_work_14'].apply(lambda x: 1 if x==13 else 0)
    data['m_work_14_cannot_work'] = data['m_work_14'].apply(lambda x: 1 if x==13 else 0)
    data['f_work_14_no_job'] = data['f_work_14'].apply(lambda x: 1 if x==12 else 0)
    data['m_work_14_no_job'] = data['m_work_14'].apply(lambda x: 1 if x==12 else 0)    
    data['f_work_14_geti'] = data['f_work_14'].apply(lambda x: 1 if x==8 else 0)
    data['m_work_14_geti'] = data['m_work_14'].apply(lambda x: 1 if x==8 else 0)   
    
    data['status_peer'] = data['status_peer'].apply(lambda x: 2 if x<0 else x)  
    data['status_3_before'] = data['status_3_before'].apply(lambda x: 2 if x<0 else x)  
    data['view'] = data['view'].apply(lambda x: 3 if x<0 else x)     
    data['neighbor_familiarity'] = data['neighbor_familiarity'].apply(lambda x: 3 if x<0 else x)    
        
    data['gender_m'] = data['gender'].apply(lambda x: 1 if x==1 else 0)
    data['gender_f'] = data['gender'].apply(lambda x: 1 if x==2 else 0)
    data['nationality'] = data['nationality'].apply(lambda x: 1 if x <0 else x)
    data['nationality_han'] = data['nationality'].apply(lambda x: 1 if x in [1, 3] else 0)
    data['nationality_xinjiang'] = data['nationality'].apply(lambda x: 1 if x in [7, 5, 4] else 0)
    data['nationality_meng'] = data['nationality'].apply(lambda x: 1 if x in [2, 6, 8] else 0)
    data['religion'] = data['religion'].apply(lambda x: 0 if x<0 else x)
    data['health'] = data['health'].apply(lambda x: 3 if x<0 else x)
    data['health_problem'] = data['health_problem'].apply(lambda x: 1 if x<0 else x)
    data['depression'] = data['depression'].apply(lambda x: 1 if x<0 else x)
    
    data['hukou_loc_local'] = data['hukou_loc'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
    data['hukou_loc_not_local'] = data['hukou_loc'].apply(lambda x: 1 if x not in [1, 2, 3] else 0)
    data['s_hukou_nong'] = data['s_hukou'].apply(lambda x: 1 if x==1 else 0)
    data['s_hukou_cheng'] = data['s_hukou'].apply(lambda x: 1 if x in [4, 5] else 0)
    data['s_hukou_jun'] = data['s_hukou'].apply(lambda x: 1 if x==6 else 0)
    data['s_hukou_other'] = data['s_hukou'].apply(lambda x: 1 if x in [2,3,8] else 0)
    data['s_hukou_no'] = data['s_hukou'].apply(lambda x: 1 if x==7 or x<0 else 0)

    data['hukou_nong'] = data['hukou'].apply(lambda x: 1 if x==1 else 0)
    data['hukou_cheng'] = data['hukou'].apply(lambda x: 1 if x in [4, 5] else 0)
    data['hukou_jun'] = data['hukou'].apply(lambda x: 1 if x==6 else 0)
    data['hukou_other'] = data['hukou'].apply(lambda x: 1 if x in [2,3,8] else 0)
    data['hukou_no'] = data['hukou'].apply(lambda x: 1 if x==7 or x<0 else 0)
    
    media_types = ['media_1','media_2','media_3','media_4','media_5', 'media_6']
    data[media_types] = data[media_types].applymap(lambda x: 3 if x<0 else x)
    
    data['socialize'] = data['socialize'].apply(lambda x: 3 if x <0 else x)
    data['relax'] = data['relax'].apply(lambda x: 3 if x <0 else x)
    data['learn'] = data['learn'].apply(lambda x: 3 if x <0 else x)
    data['social_neighbor'] = data['social_neighbor'].apply(lambda x: 3 if x <0 else x)
    data['social_friend'] = data['social_friend'].apply(lambda x: 3 if x <0 else x)
    data['socia_outing'] = data['socia_outing'].apply(lambda x: 3 if x <0 else x)
    data['equity'] = data['equity'].apply(lambda x: 4 if x <0 else x)
    
    data['work_status_boss'] = data['work_status'].apply(lambda x: 1 if x ==1 else 0) 
    data['work_status_geti'] = data['work_status'].apply(lambda x: 1 if x ==2 else 0) 
    data['work_status_yuangong'] = data['work_status'].apply(lambda x: 1 if x in [3,4,5] else 0)    
    data['work_status_jiazu'] = data['work_status'].apply(lambda x: 1 if x in [6, 7] else 0) 
    data['work_status_qita'] = data['work_status'].apply(lambda x: 1 if x in [8, 9] or x<0  else 0) 

    data['s_work_status_boss'] = data['s_work_status'].apply(lambda x: 1 if x ==1 else 0) 
    data['s_work_status_geti'] = data['s_work_status'].apply(lambda x: 1 if x ==2 else 0) 
    data['s_work_status_yuangong'] = data['s_work_status'].apply(lambda x: 1 if x in [3,4,5] else 0)    
    data['s_work_status_jiazu'] = data['s_work_status'].apply(lambda x: 1 if x in [6, 7] else 0) 
    data['s_work_status_qita'] = data['s_work_status'].apply(lambda x: 1 if x in [8, 9] or x<0  else 0) 
    
    data['work_yr_nong'] = data['work_yr'].apply(lambda x: 1 if  x<0  else 0) 
    data['work_yr_not_nong'] = data['work_yr'].apply(lambda x: 0 if x<0  else x)
    
    data['work_type_all_day'] = data['work_type'].apply(lambda x: 1 if x==1  else 0) 
    data['work_type_part_time'] = data['work_type'].apply(lambda x: 1 if x==2 or x<0  else 0) 
    data['s_work_type_all_day'] = data['s_work_type'].apply(lambda x: 1 if x==1  else 0) 
    data['s_work_type_part_time'] = data['s_work_type'].apply(lambda x: 1 if x==2 or x<0  else 0) 
       
    data['work_manage_boss'] = data['work_manage'].apply(lambda x: 1 if x==1  else 0) 
    data['work_manage_mid'] = data['work_manage'].apply(lambda x: 1 if x==2  else 0) 
    data['work_manage_worker'] = data['work_manage'].apply(lambda x: 1 if x==3  else 0) 
    data['work_manage_free'] = data['work_manage'].apply(lambda x: 1 if x==4 or x<0  else 0) 
    
    insur_info = ['insur_1', 'insur_2', 'insur_3', 'insur_4']
    data[insur_info] = data[insur_info].applymap(lambda x: 2 if x<0  else x) 
    

    data['family_status']  = data['family_status'].apply(lambda x: 3 if x<0  else x) 
    data['house']  = data['house'].apply(lambda x: 0 if x<0  else x) 

#     data['car_no']  = data['car'].apply(lambda x: 1 if x<0 or x==2  else 0) 
#     data['car_have']  = data['car'].apply(lambda x: 1 if x==1  else 0) 
    
    data['son'] = data['son'].apply(lambda x: 1 if x<0  else x) 
    data['daughter'] = data['daughter'].apply(lambda x: 1 if x<0  else x) 
    data['minor_child'] = data['minor_child'].apply(lambda x: 1 if x<0  else x) 
    
    data['marital_1'] = data['marital'].apply(lambda x: 1 if x==1  else x)
    data['marital_2'] = data['marital'].apply(lambda x: 1 if x==2 else x)
    data['marital_3'] = data['marital'].apply(lambda x: 1 if x==3  else x)
    data['marital_4'] = data['marital'].apply(lambda x: 1 if x==4  else x)
    data['marital_5'] = data['marital'].apply(lambda x: 1 if x==5  else x)
    data['marital_6'] = data['marital'].apply(lambda x: 1 if x==6  else x)
    data['marital_7'] = data['marital'].apply(lambda x: 1 if x==7  else x)
    data['marital_8'] = data['marital'].apply(lambda x: 1 if x<0  else x)
    
    data['s_work_exper'] = data['s_work_exper'].apply(lambda x: 1 if x<0  else x)
    
    data = otherData.addOtherData(data)
    stopFeatures = ['invest_other', 'property_other', 'edu_other', 'survey_time', 'realBirth'\
                    , 'survey_type', 'province', 'city', 'county', 'gender', 'hukou_loc', 'work_status'\
                    ,'work_yr', 'work_manage', 'work_type','s_work_type',  'marital', 'car', 'invest_6', 's_hukou', 'hukou'\
                    ,'s_work_status', 'political', 's_political', 'f_political', 'm_political', 'f_work_14'\
                    ,'m_work_14', 'nationality', 'car', 'edu_yr']
    data = data.drop(stopFeatures, axis=1)

    if features==None:
        features = []
        for featureName in data.columns:
                features.append(featureName)
    
    data = data[features]
    return data

