# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:56:04 2020

@author: finup
"""


import sys
sys.path.append(r'F:\Python\CreditScoreCard')


import numpy as np
import pandas as pd
import datetime
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib

from base import load_from_csv
from base import variable_char_type, cross_table
from base import to_exist_excel, to_new_excel
from base import cols_one_hot
from explore import variable_plot, variable_summary
from clean import drop_variables_by_class, drop_variables_by_unique_value, drop_variables_by_missing_value, \
                  drop_variables_by_overcenter_value




##======================================================================================================##

#### 百融
BR_ApplyEvaluate = pd.read_csv(r'F:\data\三方提测样本返回\BR_ApplyEvaluate.csv', header=0, skiprows=[1])
BR_ApplyEvaluate = BR_ApplyEvaluate.drop(['cell','name',
                                          'cus_num', 'sl_user_date', 'user_time', 'custApiCode', 'swift_number',
                                          'cus_username', 'code', 'flag_applyevaluate'], axis=1)

BR_ApplyEvaluate = BR_ApplyEvaluate.drop_duplicates()
BR_ApplyEvaluate = BR_ApplyEvaluate.rename(columns = {'id':'id_no', 'user_date':'apply_time'})

BR_ApplyEvaluate = BR_ApplyEvaluate.drop_duplicates()
dup_ser = BR_ApplyEvaluate.groupby(['id_no','apply_time'])['id_no'].count()
BR_ApplyEvaluate = BR_ApplyEvaluate[~BR_ApplyEvaluate['id_no'].isin(dup_ser[dup_ser>1].unstack().index.tolist())]


#### 探知
tz_gz = load_from_csv(filePath = r'F:\data\三方提测样本返回\TZ_gongZhai.csv', charTrans=False)
tz_gz = tz_gz.rename(columns = {'mobile':'cell_id', 'idCard':'id_no', 'backTime':'apply_time',
                                     '风险原因':'tz_risk_reason',
                                     '风险等级':'tz_risk_rank',
                                     '近一个月注册平台数':'tz_last1_reg_plat_cnt',
                                     '近一个月注册平台类型(银行)':'tz_last1_reg_plat_bank_cnt',
                                     '近一个月注册平台类型(保险)':'tz_last1_reg_plat_insure_cnt',
                                     '近一个月注册平台类型(互联网金融)':'tz_last1_reg_plat_itfin_cnt',
                                     '近一个月注册平台类型(消费金融)':'tz_last1_reg_plat_consume_cnt',
                                     '近一个月注册平台类型(其他金融)':'tz_last1_reg_plat_other_cnt',
                                     '近一个月申请平台数':'tz_last1_apl_plat_cnt',
                                     '近一个月申请次数':'tz_last1_apl_cnt',
                                     '近一个月申请平台类型(银行)':'tz_last1_apl_plat_bank_cnt',
                                     '近一个月申请平台类型(保险)':'tz_last1_apl_plat_insure_cnt',
                                     '近一个月申请平台类型(互联网金融)':'tz_last1_apl_plat_itfin_cnt',
                                     '近一个月申请平台类型(消费金融)':'tz_last1_apl_plat_consume_cnt',
                                     '近一个月申请平台类型(其他金融)':'tz_last1_apl_plat_other_cnt',
                                     '近一个月使用平台数':'tz_last1_used_plat_cnt',
                                     '近一个月使用次数':'tz_last1_used_cnt',
                                     '近一个月使用平台类型(银行)':'tz_last1_used_plat_bank_cnt',
                                     '近一个月使用平台类型(保险)':'tz_last1_used_plat_insure_cnt',
                                     '近一个月使用平台类型(互联网金融)':'tz_last1_used_plat_itfin_cnt',
                                     '近一个月使用平台类型(消费金融)':'tz_last1_used_plat_consume_cnt',
                                     '近一个月使用平台类型(其他金融)':'tz_last1_used_plat_other_cnt',
                                     '近三个月注册平台数':'tz_last3_reg_plat_cnt',
                                     '近三个月注册平台类型(银行)':'tz_last3_reg_plat_bank_cnt',
                                     '近三个月注册平台类型(保险)':'tz_last3_reg_plat_insure_cnt',
                                     '近三个月注册平台类型(互联网金融)':'tz_last3_reg_plat_itfin_cnt',
                                     '近三个月注册平台类型(消费金融)':'tz_last3_reg_plat_consume_cnt',
                                     '近三个月注册平台类型(其他金融)':'tz_last3_reg_plat_other_cnt',
                                     '近三个月申请平台数':'tz_last3_apl_plat_cnt',
                                     '近三个月申请次数':'tz_last3_apl_cnt',
                                     '近三个月申请平台类型(银行)':'tz_last3_apl_plat_bank_cnt',
                                     '近三个月申请平台类型(保险)':'tz_last3_apl_plat_insure_cnt',
                                     '近三个月申请平台类型(互联网金融)':'tz_last3_apl_plat_itfin_cnt',
                                     '近三个月申请平台类型(消费金融)':'tz_last3_apl_plat_consume_cnt',
                                     '近三个月申请平台类型(其他金融)':'tz_last3_apl_plat_other_cnt',
                                     '近三个月使用平台数':'tz_last3_used_plat_cnt',
                                     '近三个月使用次数':'tz_last3_used_cnt',
                                     '近三个月使用平台类型(银行)':'tz_last3_used_plat_bank_cnt',
                                     '近三个月使用平台类型(保险)':'tz_last3_used_plat_insure_cnt',
                                     '近三个月使用平台类型(互联网金融)':'tz_last3_used_plat_itfin_cnt',
                                     '近三个月使用平台类型(消费金融)':'tz_last3_used_plat_consume_cnt',
                                     '近三个月使用平台类型(其他金融)':'tz_last3_used_plat_other_cnt',
                                     '近六个月注册平台数':'tz_last6_reg_plat_cnt',
                                     '近六个月注册平台类型(银行)':'tz_last6_reg_plat_bank_cnt',
                                     '近六个月注册平台类型(保险)':'tz_last6_reg_plat_insure_cnt',
                                     '近六个月注册平台类型(互联网金融)':'tz_last6_reg_plat_itfin_cnt',
                                     '近六个月注册平台类型(消费金融)':'tz_last6_reg_plat_consume_cnt',
                                     '近六个月注册平台类型(其他金融)':'tz_last6_reg_plat_other_cnt',
                                     '近六个月申请平台数':'tz_last6_apl_plat_cnt',
                                     '近六个月申请次数':'tz_last6_apl_cnt',
                                     '近六个月申请平台类型(银行)':'tz_last6_apl_plat_bank_cnt',
                                     '近六个月申请平台类型(保险)':'tz_last6_apl_plat_insure_cnt',
                                     '近六个月申请平台类型(互联网金融)':'tz_last6_apl_plat_itfin_cnt',
                                     '近六个月申请平台类型(消费金融)':'tz_last6_apl_plat_consume_cnt',
                                     '近六个月申请平台类型(其他金融)':'tz_last6_apl_plat_other_cnt',
                                     '近六个月使用平台数':'tz_last6_used_plat_cnt',
                                     '近六个月使用次数':'tz_last6_used_cnt',
                                     '近六个月使用平台类型(银行)':'tz_last6_used_plat_bank_cnt',
                                     '近六个月使用平台类型(保险)':'tz_last6_used_plat_insure_cnt',
                                     '近六个月使用平台类型(互联网金融)':'tz_last6_used_plat_itfin_cnt',
                                     '近六个月使用平台类型(消费金融)':'tz_last6_used_plat_consume_cnt',
                                     '近六个月使用平台类型(其他金融)':'tz_last6_used_plat_other_cnt',
                                     '近十二个月注册平台数':'tz_last12_reg_plat_cnt',
                                     '近十二个月注册平台类型(银行)':'tz_last12_reg_plat_bank_cnt',
                                     '近十二个月注册平台类型(保险)':'tz_last12_reg_plat_insure_cnt',
                                     '近十二个月注册平台类型(互联网金融)':'tz_last12_reg_plat_itfin_cnt',
                                     '近十二个月注册平台类型(消费金融)':'tz_last12_reg_plat_consume_cnt',
                                     '近十二个月注册平台类型(其他金融)':'tz_last12_reg_plat_other_cnt',
                                     '近十二个月申请平台数':'tz_last12_apl_plat_cnt',
                                     '近十二个月申请次数':'tz_last12_apl_cnt',
                                     '近十二个月申请平台类型(银行)':'tz_last12_apl_plat_bank_cnt',
                                     '近十二个月申请平台类型(保险)':'tz_last12_apl_plat_insure_cnt',
                                     '近十二个月申请平台类型(互联网金融)':'tz_last12_apl_plat_itfin_cnt',
                                     '近十二个月申请平台类型(消费金融)':'tz_last12_apl_plat_consume_cnt',
                                     '近十二个月申请平台类型(其他金融)':'tz_last12_apl_plat_other_cnt',
                                     '近十二个月使用平台数':'tz_last12_used_plat_cnt',
                                     '近十二个月使用次数':'tz_last12_used_cnt',
                                     '近十二个月使用平台类型(银行)':'tz_last12_used_plat_bank_cnt',
                                     '近十二个月使用平台类型(保险)':'tz_last12_used_plat_insure_cnt',
                                     '近十二个月使用平台类型(互联网金融)':'tz_last12_used_plat_itfin_cnt',
                                     '近十二个月使用平台类型(消费金融)':'tz_last12_used_plat_consume_cnt',
                                     '近十二个月使用平台类型(其他金融)':'tz_last12_used_plat_other_cnt',
                                     '近二十四个月注册平台数':'tz_last24_reg_plat_cnt',
                                     '近二十四个月注册平台类型(银行)':'tz_last24_reg_plat_bank_cnt',
                                     '近二十四个月注册平台类型(保险)':'tz_last24_reg_plat_insure_cnt',
                                     '近二十四个月注册平台类型(互联网金融)':'tz_last24_reg_plat_itfin_cnt',
                                     '近二十四个月注册平台类型(消费金融)':'tz_last24_reg_plat_consume_cnt',
                                     '近二十四个月注册平台类型(其他金融)':'tz_last24_reg_plat_other_cnt',
                                     '近二十四个月申请平台数':'tz_last24_apl_plat_cnt',
                                     '近二十四个月申请次数':'tz_last24_apl_cnt',
                                     '近二十四个月申请平台类型(银行)':'tz_last24_apl_plat_bank_cnt',
                                     '近二十四个月申请平台类型(保险)':'tz_last24_apl_plat_insure_cnt',
                                     '近二十四个月申请平台类型(互联网金融)':'tz_last24_apl_plat_itfin_cnt',
                                     '近二十四个月申请平台类型(消费金融)':'tz_last24_apl_plat_consume_cnt',
                                     '近二十四个月申请平台类型(其他金融)':'tz_last24_apl_plat_other_cnt',
                                     '近二十四个月使用平台数':'tz_last24_used_plat_cnt',
                                     '近二十四个月使用次数':'tz_last24_used_cnt',
                                     '近二十四个月使用平台类型(银行)':'tz_last24_used_plat_bank_cnt',
                                     '近二十四个月使用平台类型(保险)':'tz_last24_used_plat_insure_cnt',
                                     '近二十四个月使用平台类型(互联网金融)':'tz_last24_used_plat_itfin_cnt',
                                     '近二十四个月使用平台类型(消费金融)':'tz_last24_used_plat_consume_cnt',
                                     '近二十四个月使用平台类型(其他金融)':'tz_last24_used_plat_other_cnt',
                                     '长周期共债评分':'tz_long_debt_score',
                                     '短周期共债评分':'tz_short_debt_score'
                                    })
tz_gz = tz_gz.drop(['name', 'cell_id','reqToken','code'], axis=1)
tz_gz = tz_gz.drop_duplicates()
dup_ser = tz_gz.groupby(['id_no','apply_time'])['id_no'].count()
tz_gz = tz_gz[~tz_gz['id_no'].isin(dup_ser[dup_ser>1].unstack().index.tolist())]
tz_gz['id_no'] = tz_gz['id_no'].map(lambda x: x.upper())


## 黑瞳（高危客群）
df_heit = pd.read_csv(r'F:\data\三方初步入选数据\HT_data.csv')
df_heit = df_heit.drop(['mobile_md5','name','phone','loan_date'], axis=1)
df_heit = df_heit.rename(columns={'id':'id_no', 'apply_date':'apply_time'})
df_heit['apply_time'] = df_heit['apply_time'].apply(lambda x: str(datetime.datetime.strptime(x,'%Y/%m/%d').date()))
df_heit = df_heit.rename(columns = {'cause_code':'ht_cause_code',
                                     'loandemand_bank':'ht_loandemand_bank',
                                     'loandemand_nonbank':'ht_loandemand_nonbank',
                                     'loandemand':'ht_loandemand',
                                     'csumcap':'ht_csumcap',
                                     'activescenario':'ht_activescenario',
                                     'hicorescore_g':'ht_hicorescore_g',
                                     'riskwarning':'ht_riskwarning',
                                     'risklevel':'ht_risklevel',
                                     'suspect':'ht_suspect',
                                     'repaycap':'ht_repaycap'})

heit_blacklist = df_heit[['id_no', 'apply_time', 'ht_suspect']]
heit_blacklist = heit_blacklist.drop_duplicates()
heit_blacklist['id_no'] = heit_blacklist['id_no'].map(lambda x: x.upper())


## 探知（探真逾期分析报告）
tz_yuqi = load_from_csv(filePath = r'F:\data\三方提测样本返回\TZ_yuQi.csv', charTrans=False)
tz_yuqi = tz_yuqi.drop(['name', 'mobile', 'reqToken', 'code'], axis=1)
tz_yuqi = tz_yuqi.rename(columns = { 'idCard':'id_no', 
                                     'backTime':'apply_time',
                                     'm1逾期提醒事件累计次数':'tz_m1_message_event_cnt',
                                     'm1逾期已还事件累计次数':'tz_m1_repaid_event_cnt',
                                     'm3逾期提醒平台累计次数':'tz_m3_message_plat_cnt',
                                     'm3逾期已还平台累计次数':'tz_m3_repaid_plat_cnt',
                                     'm3逾期提醒事件累计次数':'tz_m3_overdue_alarm_cnt',
                                     'm3逾期已还事件累计次数':'tz_m3_overdue_repaid_cnt',
                                     'm6逾期提醒平台累计次数':'tz_m6_message_plat_cnt',
                                     'm6逾期已还平台累计次数':'tz_m6_repaid_plat_cnt',
                                     'm6逾期提醒事件累计次数':'tz_m6_overdue_alarm_cnt',
                                     'm6逾期已还事件累计次数':'tz_m6_overdue_repaid_cnt',
                                     'm6最大借贷类逾期提醒金额等级':'tz_m6_message_amt_rank',
                                     'm6最大逾期已还时长等级':'tz_m6_repaid_time_rank',
                                     'm6最大逾期时长等级':'tz_m6_time_rank',
                                     '最近一次逾期提醒事件距当前时间天数':'tz_last_overdue_event_days',
                                     '最近一次逾期已还事件距当前时间天数':'tz_last_repaid_event_days'})

tz_yuqi = tz_yuqi.drop_duplicates()
dup_ser = tz_yuqi.groupby(['id_no','apply_time'])['id_no'].count()
tz_yuqi = tz_yuqi[~tz_yuqi['id_no'].isin(dup_ser[dup_ser>1].unstack().index.tolist())]
tz_yuqi['id_no'] = tz_yuqi['id_no'].map(lambda x: x.upper())



## target
df_target = pd.read_csv(r'F:\data\三方提测样本返回\final_encrypt_sample.csv')
df_target = df_target[['request_id', 'date', 'encode_id_no', 'if_loan', 'if_7p', 'if_now_m1p']]
df_target = df_target.rename(columns = {'date':'apply_time', 'encode_id_no':'id_no'})

## 百融、探知和黑瞳数据合并
df_table = df_target.merge(BR_ApplyEvaluate, on=['id_no','apply_time'], how='left')
df_table = df_table.merge(tz_gz, on=['id_no','apply_time'], how='left')
df_table = df_table.merge(tz_yuqi, on=['id_no','apply_time'], how='left')
df_table = df_table.merge(heit_blacklist, on=['id_no','apply_time'], how='left')

del BR_ApplyEvaluate, tz_gz, tz_yuqi, heit_blacklist, df_heit, df_target, dup_ser



##======================================================================================================##
##  定义target
df = df_table.copy()
df = df[df['if_7p'].notnull()]
df = df.drop(['id_no','if_loan','if_now_m1p'], axis=1)
df = df.rename(columns={'if_7p': 'TargetBad'})


#=========================================基础参数设置==========================================
file_path = 'F:\Python\Test'
var_time_by = 'apply_time'
var_key = 'request_id'
var_target = 'TargetBad'


#==========================================变量字符类型核实===========================================
var_type_df = variable_char_type(inDf=df, 
                                 keyVarList=[var_key], 
                                 TargetVarList=[var_target], 
                                 unScaleVarList=[])


#=========================================样本确定============================================
## 样本分析
bad_rate_df = cross_table(df, var_time_by, var_target)
bad_rate_df['BadRate'] = round(bad_rate_df[1.0]/(bad_rate_df['Total'] - bad_rate_df['MissCnt']),4)

## 样本确定
df['SampleType'] = df[var_time_by].apply(lambda x: 'INS' if x <= '2020-06-01' 
                                                         else 'OOT' if x <= '2020-06-28' else '' )
sample_bad_df = cross_table(df, 'SampleType', var_target)
sample_bad_df['BadRate'] = round(sample_bad_df[1.0]/(sample_bad_df['Total'] - sample_bad_df['MissCnt']),4)
print("INS和OOT逾期率分布： \n", sample_bad_df)

ins_raw_df = df[df['SampleType'] == 'INS']
model_raw_df = df[df['SampleType'].isin(['INS','OOT'])]
model_raw_df = model_raw_df.drop('apply_time',axis=1)

del bad_rate_df, sample_bad_df



#==========================================特征分析===========================================

# 确定变量类型及变量值的数量
unscale_var_ls = [var_key,var_target,var_time_by,'SampleType']
ins_raw_var_class_df = variable_char_type(inDf = ins_raw_df, 
                           keyVarList = [var_key], 
                           TargetVarList = [var_target],
                           unScaleVarList = unscale_var_ls)

# 输出变量分布图（连续变量和分类变量两类进行输出）
variable_plot(inDf = ins_raw_df, 
              inVarClassDf = ins_raw_var_class_df, 
              savUrl = file_path)

# 输出变量的统计特征
var_explore_result = variable_summary(inDf=ins_raw_df,inVarClassDf=ins_raw_var_class_df)


# 连续变量统计特征excel输出
to_new_excel(filePath = file_path,
             fileName = 'VarExploreResult',
             sheetName = '连续变量统计特征',
             dataFrm = var_explore_result['contStatSummary'])
# 分类变量分布及占比excel输出
to_exist_excel(filePath = file_path,
               fileName = 'VarExploreResult',
               sheetName= '分类变量分布',
               dataFrm=var_explore_result['classFreqSummary'])



#==========================================无效变量剔除========================================= 

# 按照变量类型确定需要剔除的变量列表
drop_var_list_by_class = drop_variables_by_class(
                        inVarClassDf = ins_raw_var_class_df, 
                        toDropVarClassList = ['UnScale', 'Date', 'Droped'])
# 确定只有一个值的变量列表
drop_var_list_by_unique = drop_variables_by_unique_value(
                        inVarClassDf = ins_raw_var_class_df)
# 按照缺失值比例确定需要剔除的变量列表
drop_var_list_by_missing = drop_variables_by_missing_value(
                        inVarClassDf = ins_raw_var_class_df,
                        dropMissRatePoint = 0.9)
# 剔除分类样本过于集中的变量
drop_var_list_by_overcenter = drop_variables_by_overcenter_value(
                        inVarDistDf = var_explore_result['classFreqSummary'],
                        dropOverCenterPoint = 0.95)

# 在建模样本中剔除需要剔除的变量
drop_var_list = drop_var_list_by_class+drop_var_list_by_unique+drop_var_list_by_missing+drop_var_list_by_overcenter
ins_clean_df = ins_raw_df.drop(drop_var_list, axis=1)

# 剔除变量后，更新建模变量类型及样本量数据框
ins_clean_class_df = ins_raw_var_class_df[~ins_raw_var_class_df['index'].isin(drop_var_list)]
drop_by_freq_df = ins_raw_var_class_df[ins_raw_var_class_df['index'].isin(drop_var_list)]

print('===================  无效变量剔除完成！ =====================')
times = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('=================== ', times)

# 阶段性结果输出
to_new_excel(filePath = file_path,
             fileName = '无效变量基本条件剔除',
             sheetName = '剔除变量',
             dataFrm = drop_by_freq_df)
to_exist_excel(filePath = file_path,
               fileName = '无效变量基本条件剔除',
               sheetName = '保留变量',
               dataFrm = ins_clean_class_df)

del drop_var_list_by_class, drop_var_list_by_unique, drop_var_list_by_missing, drop_var_list_by_overcenter
del drop_var_list


'''
#==========================================缺失值填充===========================================
## object变量缺失值填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] == 'object']['index'].tolist():
    ins_clean_df[var_name] = ins_clean_df[var_name].fillna('Missing')

## 数值型变量缺失值0填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] != 'object']['index'].tolist():
    ## 对缺失值比例小于5%的变量进行填充
    if ins_clean_class_df[ins_clean_class_df['index'] == var_name]['NmissRate'].tolist()[0] <= 0.05:
        #median_value = ins_clean_df[var_name].median()
        ins_clean_df[var_name] = ins_clean_df[var_name].fillna(0)



#==========================================ALl-Sample缺失值填充===========================================
sample_df = model_raw_df.copy()
## object变量缺失值填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] == 'object']['index'].tolist():
    sample_df[var_name] = sample_df[var_name].fillna('Missing')

## 数值型变量缺失值0填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] != 'object']['index'].tolist():
    ## 对缺失值比例小于5%的变量进行填充
    if ins_clean_class_df[ins_clean_class_df['index'] == var_name]['NmissRate'].tolist()[0] <= 0.05:
        #median_value = ins_clean_df[var_name].median()
        sample_df[var_name] = sample_df[var_name].fillna(0)
'''




#===========================================GBDT============================================
#============================================================================================#

from model import gbdt_grid_cv
from sklearn.ensemble import GradientBoostingClassifier

#---------------INS缺失值填充
## object变量缺失值填充
gbdt_ins_clean_df = ins_clean_df.copy()
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] == 'object']['index'].tolist():
    gbdt_ins_clean_df[var_name] = gbdt_ins_clean_df[var_name].fillna('Missing')

## 数值型变量缺失值0填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] != 'object']['index'].tolist():
    ## 对缺失值比例小于5%的变量进行填充
    if ins_clean_class_df[ins_clean_class_df['index'] == var_name]['NmissRate'].tolist()[0] <= 0.05:
        #median_value = ins_clean_df[var_name].median()
        gbdt_ins_clean_df[var_name] = gbdt_ins_clean_df[var_name].fillna(0)



#-----------------ALl-Sample缺失值填充
gbdt_sample_df = model_raw_df.copy()
## object变量缺失值填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] == 'object']['index'].tolist():
    gbdt_sample_df[var_name] = gbdt_sample_df[var_name].fillna('Missing')

## 数值型变量缺失值0填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] != 'object']['index'].tolist():
    ## 对缺失值比例小于5%的变量进行填充
    if ins_clean_class_df[ins_clean_class_df['index'] == var_name]['NmissRate'].tolist()[0] <= 0.05:
        #median_value = ins_clean_df[var_name].median()
        gbdt_sample_df[var_name] = gbdt_sample_df[var_name].fillna(0)


## 调参
x_list = [x for x in gbdt_ins_clean_df.columns.tolist() if x not in [var_key, var_target, var_time_by, 'SampleType']]
catgory_list = [x for x in ins_clean_class_df[ins_clean_class_df['Dtypes']=='object']['index'].tolist() if x in x_list]
gbdt_ins_x = cols_one_hot(inDf = gbdt_ins_clean_df[x_list], varLst = catgory_list)
gbdt_ins_x = gbdt_ins_x.fillna(0)

param_grid = {'n_estimators': [50],
              'learning_rate': [0.03],
              'max_depth': [4],
              'subsample': [0.9],
              'max_features': [0.8], #fraction、sqrt、log2、None
              'min_samples_split':[50],
              'random_state':[1000]
              }
gbdt_grid_res = gbdt_grid_cv(gbdt_ins_x, ins_clean_df[var_target], param_grid, scoring='roc_auc', 
                             n_jobs=4, cv=4, validation_fraction=0.1)
## n_estimators=0.02 learning_rate=100

## 最优参数模型
gbdt_params = {'n_estimators': 50,
              'learning_rate': 0.03,
              'max_depth': 4,
              'subsample': 0.9,
              'max_features': 0.8, #fraction、sqrt、log2、None
              'min_samples_split': 50,
              'random_state': 1000}
gbdt_model = GradientBoostingClassifier(**gbdt_params)
gbdt_model.fit(gbdt_ins_x, ins_clean_df[var_target])

## 保存模型
joblib.dump(gbdt_model, r'F:\Python\Test\model_result\gbdt.m')
gbdt_model = joblib.load(r'F:\Python\Test\model_result\gbdt.m')

## 变量重要性
gbdt_feature_importance = pd.Series(gbdt_model.feature_importances_, index=gbdt_ins_x.columns.tolist(), name='values').sort_values(ascending=False)
gbdt_feature_importance = gbdt_feature_importance[gbdt_feature_importance>0].map(lambda x: '{:.5f}'.format(x))
#gbdt_feature_importance.plot(kind='bar', title='Feature Importances')

## 预测评估
gbdt_sample_x = cols_one_hot(inDf = gbdt_sample_df[x_list], varLst = catgory_list)
gbdt_sample_x = gbdt_sample_x[gbdt_ins_x.columns.tolist()]
gbdt_sample_x = gbdt_sample_x.fillna(0)

gbdt_sample_pred = gbdt_sample_df.copy()
gbdt_sample_pred['y_pred'] = gbdt_model.predict_proba(gbdt_sample_x)[:,1]
gbdt_ins_pred = gbdt_sample_pred[gbdt_sample_pred['SampleType']=='INS']
gbdt_oot_pred = gbdt_sample_pred[gbdt_sample_pred['SampleType']=='OOT']
print('INS-ROC: ', roc_auc_score(gbdt_ins_pred[var_target], gbdt_ins_pred['y_pred']))
print('OOT-ROC: ', roc_auc_score(gbdt_oot_pred[var_target], gbdt_oot_pred['y_pred']))

## 特征工程
alg = gbdt_model
x_df = gbdt_ins_x
alg_method = 'gbdt'
(gbdt_leaf_df, gbdt_leaf_feature_df) = alg_to_feature(alg, x_df, alg_method)



#=========================================XGBoost============================================
#============================================================================================#

from model import xgb_grid_cv
import xgboost as xgb

## 调参
x_list = [x for x in ins_clean_df.columns.tolist() if x not in [var_key, var_target, var_time_by, 'SampleType']]
catgory_list = [x for x in ins_clean_class_df[ins_clean_class_df['Dtypes']=='object']['index'].tolist() if x in x_list]
xgb_ins_x = cols_one_hot(inDf = ins_clean_df[x_list], varLst = catgory_list)

xgb_ins = xgb.DMatrix(xgb_ins_x.values, 
                      label=ins_clean_df[var_target].values)
params = {'booster': 'gbtree',
          'learning_rate': 0.1,
          'gamma': 0, #进一步分割，损失函数减少的阈值
          'max_depth': 6,
          'min_child_weight': 1,
          'subsample': 1, #总体抽样比例
          'sampling_method': 'uniform',
          'colsample_bytree': 1, #每颗树的构建抽样比例
          'lambda': 1, #L2惩罚项
          'alpha': 0, #L1惩罚项
          'objective': 'binary:logistic',
          'growth_policy': 'depthwise',  # lossguide(leaf-wise)
          'eval_metric': 'auc',
          'seed': 1000
         }
grid_params = {'lambda': [1,3,5]}
xgboost_res = xgb_grid_cv(params, xgb_ins, grid_params,
                num_boost_round=10000, early_stopping_rounds = 30, nfold=4, metrics='auc')

## 最优参数模型
params = {'booster': 'gbtree',
          'learning_rate': 0.03,
          'n_estimators': 190,
          'gamma': 0,
          'max_depth': 3,
          'min_child_weight': 30,
          'subsample': 1,
          'sampling_method': 'uniform',
          'colsample_bytree': 1,
          'lambda': 3,
          'alpha': 0,
          'objective': 'binary:logistic',
          'growth_policy': 'depthwise',  # lossguide(leaf-wise)
          'eval_metric': 'auc',
          'seed': 1000
         }
xgb_model = xgb.XGBClassifier(**params)
xgb_model.fit(xgb_ins_x.values, ins_clean_df[var_target].values)

## 保存模型
joblib.dump(xgb_model, r'F:\Python\Test\model_result\xgb.pkl')
xgb_model = joblib.load(r'F:\Python\Test\model_result\xgb.pkl')


## 变量重要性
xgb_feature_importance = pd.Series(xgb_model.feature_importances_, index=xgb_ins_x.columns.tolist(), name='values').sort_values(ascending=False)
xgb_feature_importance = xgb_feature_importance[xgb_feature_importance>0]
#xgb_feature_importance.plot(kind='bar', title='Feature Importances')

## 预测评估
xgb_sample_df = model_raw_df.copy()
x_list = [x for x in ins_clean_df.columns.tolist() if x not in [var_key, var_target, var_time_by, 'SampleType']]
catgory_list = [x for x in ins_clean_class_df[ins_clean_class_df['Dtypes']=='object']['index'].tolist() if x in x_list]
xgb_sample_x = cols_one_hot(inDf = xgb_sample_df[x_list], varLst = catgory_list)
xgb_sample_x = xgb_sample_x[xgb_ins_x.columns.tolist()]

xgb_sample_pred = xgb_sample_df.copy()
xgb_sample_pred['y_pred'] = xgb_model.predict_proba(xgb_sample_x.values)[:,1]
xgb_ins_pred = xgb_sample_pred[xgb_sample_pred['SampleType']=='INS']
xgb_oot_pred = xgb_sample_pred[xgb_sample_pred['SampleType']=='OOT']
print('INS-ROC: ', roc_auc_score(xgb_ins_pred[var_target], xgb_ins_pred['y_pred']))
print('OOT-ROC: ', roc_auc_score(xgb_oot_pred[var_target], xgb_oot_pred['y_pred']))

## 特征工程
alg = xgb_model
x_df = xgb_ins_x
alg_method = 'xgboost'
(xgb_leaf_df, xgb_leaf_feature_df) = alg_to_feature(alg, x_df, alg_method)



#===========================================lightGBM============================================
#============================================================================================#

from model.LightGBM import lgb_grid_cv
import lightgbm as lgb
from base import cols_to_label, cols_been_label

x_list = [x for x in ins_clean_df.columns.tolist() if x not in [var_key, var_target, var_time_by, 'SampleType']]
catgory_list = [x for x in ins_clean_class_df[ins_clean_class_df['Dtypes']=='object']['index'].tolist() if x in x_list]
(lgb_ins_x, lgb_ins_code) = cols_to_label(inDf = ins_clean_df[x_list], varLst=catgory_list)

dtrain = lgb.Dataset(lgb_ins_x, label=ins_clean_df[var_target], categorical_feature=catgory_list)
params = {
        'boosting_type': 'goss',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'max_depth': 10,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'max_bin': 255,
        'verbose': 0
        }
grid_params = {'boosting_type': ['goss','dart']}
lgb_res = lgb_grid_cv(params, dtrain, grid_params, categorical_feature=catgory_list,
                num_boost_round=500, early_stopping_rounds = 30, nfold=4, metrics='auc')

## 最优参数模型
params = {
        'boosting_type': 'goss',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 20,
        'max_depth': 4,
        'learning_rate': 0.05,
        'num_boost_round': 55,
        'feature_fraction': 0.9,
        'max_bin': 50,
        'verbose': 0
        }
lgb_model = lgb.LGBMClassifier(**params)
lgb_model.fit(lgb_ins_x, ins_clean_df[var_target].values, categorical_feature=catgory_list)

## 保存模型
joblib.dump(lgb_model, r'F:\Python\Test\model_result\lgb.pkl')
lgb_model = joblib.load(r'F:\Python\Test\model_result\lgb.pkl')


## 变量重要性
lgb_feature_importance = pd.Series(lgb_model.feature_importances_, index=lgb_ins_x.columns.tolist(), name='values').sort_values(ascending=False)
lgb_feature_importance = lgb_feature_importance[lgb_feature_importance>0]
#lgb_feature_importance.plot(kind='bar', title='Feature Importances')

## 预测评估
lgb_sample_df = model_raw_df.copy()
x_list = [x for x in ins_clean_df.columns.tolist() if x not in [var_key, var_target, var_time_by, 'SampleType']]
lgb_sample_x = cols_been_label(inDf = lgb_sample_df[x_list], varLabelDf=lgb_ins_code)

lgb_sample_pred = lgb_sample_df.copy()
lgb_sample_pred['y_pred'] = lgb_model.predict_proba(lgb_sample_x.values)[:,1]

lgb_ins_pred = lgb_sample_pred[lgb_sample_pred['SampleType']=='INS']
lgb_oot_pred = lgb_sample_pred[lgb_sample_pred['SampleType']=='OOT']
print('INS-ROC: ', roc_auc_score(lgb_ins_pred[var_target], lgb_ins_pred['y_pred']))
print('OOT-ROC: ', roc_auc_score(lgb_oot_pred[var_target], lgb_oot_pred['y_pred']))

## 特征工程
alg = lgb_model
x_df = lgb_ins_x
alg_method = 'lightgbm'
(lgb_leaf_df, lgb_leaf_feature_df) = alg_to_feature(alg, x_df, alg_method)



#=========================================CatBoost============================================
#============================================================================================#

from model import cat_grid_cv
import catboost as cbt

## 调参    
x_list = [x for x in ins_clean_df.columns.tolist() if x not in [var_key, var_target, var_time_by, 'SampleType']]
catgory_list = [x for x in ins_clean_class_df[ins_clean_class_df['Dtypes']=='object']['index'].tolist() if x in x_list]
ct_ins_df = ins_clean_df.copy()
for col in catgory_list:
    ct_ins_df[col] = ins_clean_df[col].fillna('Miss') 

pool = cbt.Pool(ct_ins_df[x_list], 
                label=ct_ins_df[var_target],
                cat_features = catgory_list
                )
params = {'loss_function': 'Logloss',
          'learning_rate': 0.03,
          'iterations': 1000,
          'l2_leaf_reg': 3,
          'depth': 6,
          'min_data_in_leaf': 1,
          'grow_policy': 'SymmetricTree',
          'custom_metric': 'AUC:hints=skip_train~false',
          'eval_metric': 'AUC'}
grid_params = None
catboost_res = cat_grid_cv(pool, params, grid_params, early_stopping_rounds=30, nfold=4)

## 最优参数模型
params = {'loss_function': 'Logloss',
          'learning_rate': 0.08,
          'iterations': 217,
          'l2_leaf_reg': 7,
          'depth': 4,
          'min_data_in_leaf': 30,
          'grow_policy': 'SymmetricTree',
          'custom_metric': 'AUC:hints=skip_train~false',
          'eval_metric': 'AUC'}
cbt_model = cbt.CatBoostClassifier(**params)
cbt_model.fit(ct_ins_df[x_list], 
              ct_ins_df[var_target],
              cat_features=catgory_list,
              verbose = 0)

## 保存模型
joblib.dump(cbt_model, r'F:\Python\Test\model_result\cbt.pkl')
cbt_model = joblib.load(r'F:\Python\Test\model_result\cbt.pkl')

## 变量重要性
cbt_feature_importance = pd.Series(cbt_model.feature_importances_, index=x_list, name='values').sort_values(ascending=False)
cbt_feature_importance = cbt_feature_importance[cbt_feature_importance>0].map(lambda x: '{:.5f}'.format(x))
#cbt_feature_importance.plot(kind='bar', title='Feature Importances')

## 预测评估

cbt_sample_df = model_raw_df.copy()
for col in catgory_list:
    cbt_sample_df[col] = cbt_sample_df[col].fillna('Miss') 

cbt_sample_df['y_pred'] = cbt_model.predict_proba(cbt_sample_df[x_list])[:,1]
cbt_ins_pred = cbt_sample_df[cbt_sample_df['SampleType']=='INS']
cbt_oot_pred = cbt_sample_df[cbt_sample_df['SampleType']=='OOT']
print('INS-ROC: ', roc_auc_score(cbt_ins_pred[var_target], cbt_ins_pred['y_pred']))
print('OOT-ROC: ', roc_auc_score(cbt_oot_pred[var_target], cbt_oot_pred['y_pred']))

## 特征工程
alg = cbt_model
x_df = ct_ins_df[x_list]
alg_method = 'catboost'
(cbt_leaf_df, cbt_leaf_feature_df) = alg_to_feature(alg, x_df, alg_method)





print('INS-ROC: ', roc_auc_score(gbdt_ins_pred[var_target], gbdt_ins_pred['y_pred']))
print('OOT-ROC: ', roc_auc_score(gbdt_oot_pred[var_target], gbdt_oot_pred['y_pred']))

print('INS-ROC: ', roc_auc_score(xgb_ins_pred[var_target], xgb_ins_pred['y_pred']))
print('OOT-ROC: ', roc_auc_score(xgb_oot_pred[var_target], xgb_oot_pred['y_pred']))

print('INS-ROC: ', roc_auc_score(lgb_ins_pred[var_target], lgb_ins_pred['y_pred']))
print('OOT-ROC: ', roc_auc_score(lgb_oot_pred[var_target], lgb_oot_pred['y_pred']))

print('INS-ROC: ', roc_auc_score(cbt_ins_pred[var_target], cbt_ins_pred['y_pred']))
print('OOT-ROC: ', roc_auc_score(cbt_oot_pred[var_target], cbt_oot_pred['y_pred']))

















