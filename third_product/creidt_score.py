


import sys
sys.path.append(r'F:\Python\CreditScoreCard')


import numpy as np
import pandas as pd
import datetime

#from base import *

from base import load_from_csv
from base import variable_char_type
from base import cross_table
from base import to_exist_excel, to_new_excel
from explore import variable_plot
from explore import variable_summary
from clean import drop_variables_by_class, drop_variables_by_unique_value, drop_variables_by_missing_value, \
                  drop_variables_by_overcenter_value
from analysis import corr_df_cal, corr_static_select
from analysis import iv_df_auto_calculate

from bins import step_bin
from bins import rate_bin_transfer
from selects import continuous_power_select, order_power_select, nominal_power_select
from selects import var_df_predict_psi, var_predict_psi_select
from woe import woe_transfer

from model import lr_forward_select, lr_sklearn_model, lr_hypothesis_test
from model import model_prob_evaluation, model_score_evaluation, score_calculate, scorecards
from oot import lr_formula_deployment, predict_compare, psi_cal, oot_prob_evaluation
 




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




##  定义target

df = df_table.copy()
df = df[df['if_loan'].notnull()]
df = df.drop(['id_no','if_loan','if_7p'], axis=1)
df = df.rename(columns={'if_now_m1p': 'TargetBad'})



#=========================================基础参数设置============================================
file_path = 'F:\Python\Test'
var_time_by = 'apply_time'
var_key = 'request_id'
var_target = 'TargetBad'


#==========================================变量字符类型核实===========================================

var_type_df = variable_char_type(inDf=df, 
                                 keyVarList=[var_key], 
                                 TargetVarList=[var_target], 
                                 unScaleVarList=[])

#var_comment_df = pd.read_csv("F:\\Python\\AutoBuildScorecard\\ModelCreditReport\\variable_name.csv",
#            skiprows = 1,
#            sep=',',
#            names = ['VarName', 'Comments'],
#            encoding = 'gb2312',
#            low_memory=False)
#var_type_df = var_type_df.merge(var_comment_df, left_on = 'index', right_on = 'VarName', how='left').drop(['VarName'],axis=1)



#=========================================样本确定============================================
## 样本分析
bad_rate_df = cross_table(df, var_time_by, var_target)
bad_rate_df['BadRate'] = round(bad_rate_df[1.0]/(bad_rate_df['Total'] - bad_rate_df['MissCnt']),4)

## 样本确定
df['SampleType'] = df[var_time_by].apply(lambda x: 'INS' if x <= '2020-05-23' 
                                                         else 'OOT' if x <= '2020-06-13' else '' )
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




#==========================================缺失值填充===========================================
## object变量缺失值填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] == 'object']['index'].tolist():
    ins_clean_df[var_name] = ins_clean_df[var_name].fillna('Missing')

## 数值型变量缺失值中位数填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] != 'object']['index'].tolist():
    ## 对缺失值比例小于5%的变量进行填充
    if ins_clean_class_df[ins_clean_class_df['index'] == var_name]['NmissRate'].tolist()[0] <= 0.05:
        median_value = ins_clean_df[var_name].median()
        ins_clean_df[var_name] = ins_clean_df[var_name].fillna(median_value)



#==========================================变量相关性分析========================================= 

corr_rst = corr_df_cal(inDf = ins_clean_df, varTypeDf = ins_clean_class_df)


drop_con_corr_ls = corr_static_select(inCorrDf=corr_rst['continue_cor_df'],
                                      selectVar='corr_coef',
                                      selectPoint=0.8)
drop_class_corr_ls = corr_static_select(inCorrDf=corr_rst['class_chisq_df'],
                                        selectVar='Chisq_DfStat',
                                        selectPoint=100)

ins_corred_class_df = ins_clean_class_df[~ins_clean_class_df['index'].isin(drop_con_corr_ls+drop_class_corr_ls)]
ins_corred_df = ins_clean_df.drop(drop_con_corr_ls+drop_class_corr_ls, axis=1)

print('===================  变量相关性分析完成！ =====================')
times = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('=================== ', times)

# 阶段性结果输出
continue_cor_df = corr_rst['continue_cor_df']
class_chisq_df = inCorrDf=corr_rst['class_chisq_df']
continue_cor_df['varname1_drop'] = continue_cor_df['varname1'].map(lambda x: 1 if x in drop_con_corr_ls else 0)
class_chisq_df['varname1_drop'] = class_chisq_df['varname1'].map(lambda x: 1 if x in drop_con_corr_ls else 0)

'''
to_new_excel(filePath = file_path,
             fileName = '变量相关性分析',
             sheetName = '连续变量相关性',
             dataFrm = continue_cor_df)
to_exist_excel(filePath = file_path,
               fileName = '变量相关性分析',
               sheetName = '分类变量相关性',
               dataFrm = class_chisq_df)
'''

del drop_con_corr_ls, drop_class_corr_ls

#==========================================变量预测能力IV及变量初步选择========================================= 

iv_var_ls = ins_corred_class_df[~ins_corred_class_df['Dclass'].isin(['Key','Target'])]['index'].tolist()
iv_df = iv_df_auto_calculate(
                            inDf = ins_corred_df,
                            xVarList = iv_var_ls,
                            yVar = var_target)


# IV值输出至excel
to_new_excel(filePath = file_path,
             fileName = 'IV',
             sheetName = 'IV',
             dataFrm = iv_df)



## 剔除预测额能力低的变量
var_high_predict_list = iv_df[iv_df['IV'] > 0.05]['VarName'].tolist()

## 手工删除不符合业务理解字段
manual_drop_list = []

## 生成最终入模变量数据集
keep_var_list = list(filter(lambda x: x not in manual_drop_list, var_high_predict_list))
ins_ived_class_df = ins_corred_class_df[ins_corred_class_df['Dclass'].isin(['Key','Target']) 
                                     | ins_corred_class_df['index'].isin(keep_var_list)]

print('===================  IV计算完成！ =====================')
times = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('=================== ', times)

del iv_var_ls, var_high_predict_list, manual_drop_list, keep_var_list




#==========================================变量逐步分箱及预测能力========================================= 

bin_rst = step_bin(inDf=ins_corred_df, inTypeDf=ins_ived_class_df, varKey=var_key, varTarget=var_target)
bin_step_dist_df = bin_rst['bin_step_dist_df']
bin_step_power_df = bin_rst['bin_step_power_df']
bin_step_iv_cal_df = bin_rst['bin_iv_cal_df']


#==========================================强预测能力变量选择========================================= 

####### 通过卡方值选择预测能力强的变量

## 二值变量选择
binary_var_power_df = nominal_power_select(inChisqDf = bin_step_power_df[bin_step_power_df['Dclass']=='Binary'], 
                                           pCutOff = 0.2,
                                           decileCutOff = 0.25)

## 名义变量强预测力变量选择
nomial_var_power_df = nominal_power_select(inChisqDf = bin_step_power_df[bin_step_power_df['Dclass']=='Nominal'], 
                                           pCutOff = 0.2,
                                           decileCutOff = 0.25)

## 有序变量变量强预测力变量选择
order_var_power_df = order_power_select(inStepFreqDf = bin_step_dist_df[bin_step_dist_df['Dclass']=='Order'], 
                                        inChisqDf = bin_step_power_df[bin_step_power_df['Dclass']=='Order'], 
                                        pCutOff = 0.2,
                                        decileCutOff = 0.25)

## 连续变量强预测力变量选择
continous_var_power_df = continuous_power_select(inStepFreqDf = bin_step_dist_df[bin_step_dist_df['Dclass']=='Continuous'], 
                                                 inChisqDf = bin_step_power_df[bin_step_power_df['Dclass']=='Continuous'], 
                                                 pCutOff = 0.2,
                                                 decileCutOff = 0.25)

## 预测能力强的变量列表
power_ls = binary_var_power_df['VarName'].unique().tolist() + nomial_var_power_df['NewVarName'].unique().tolist() + \
           order_var_power_df['NewVarName'].unique().tolist() + continous_var_power_df['NewVarName'].unique().tolist()
bin_power_dist_df = bin_step_dist_df[bin_step_dist_df['NewVarName'].isin(power_ls)]

print('===================  强预测力变量选择完成！ =====================')
print("二值变量最优选择数量： ", binary_var_power_df['VarName'].nunique())
print("名义变量最优选择数量： ", nomial_var_power_df['VarName'].nunique())
print("有序变量最优选择数量： ", order_var_power_df['VarName'].nunique())
print("连续变量最优选择数量： ", continous_var_power_df['VarName'].nunique())
times = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('=================== ', times)




## 获取处理后的变量
'''
model_step_varls_df = pd.concat([model_step_varls_df,
                                 pd.DataFrame({'step_num': 6, 'step_name': '变量最优分箱选择', 
                                               'df_sample_num': model_bin_df.shape[0], 'df_var_num': model_bin_df.shape[1],
                                               'df_name': 'model_bin_df', 'df_var_ls': model_bin_df.columns.tolist()})])
'''
    
#==========================================变量预测能力一致性选择========================================= 

## 缺失值填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] == 'object']['index'].tolist():
    model_raw_df[var_name] = model_raw_df[var_name].fillna('Missing')

for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] != 'object']['index'].tolist():
    ## 对缺失值比例小于5%的变量进行填充
    if ins_clean_class_df[ins_clean_class_df['index'] == var_name]['NmissRate'].tolist()[0] <= 0.05:
        median_value = ins_clean_df[var_name].median()
        model_raw_df[var_name] = model_raw_df[var_name].fillna(median_value)
        
       
## INS和OOT样本分箱变量转换
eval_bin_df = rate_bin_transfer(inRawDf = model_raw_df, 
                                inMapDf = bin_power_dist_df, 
                                keyVar = var_key, 
                                varKeepList=[var_target, 'SampleType'])


## 变量预测能力稳定性计算 
xVarList = list(filter(lambda x: x not in ['request_id', 'TargetBad', 'SampleType'], 
                       eval_bin_df.columns.tolist()))
var_psi_df = var_df_predict_psi(inDf=eval_bin_df, 
                                xVarList = xVarList,
                                yVarName = var_target, 
                                inTypeVar = 'SampleType')

to_new_excel(filePath = file_path,
             fileName = '变量预测力一致性计算',
             sheetName = 'var_power_psi',
             dataFrm = var_psi_df)


## 选择PSI低的变量
var_low_psi_df = var_predict_psi_select(inDf = var_psi_df, psiPoint = 0.04)
psi_low_var_ls = var_low_psi_df['VarName'].unique().tolist()


print('===================  变量一致性筛选完成！ =====================')
times = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
del xVarList
print('=================== ', times)

## 获取处理后的变量
'''
model_step_varls_df = pd.concat([model_step_varls_df,
                                 pd.DataFrame({'step_num': 7, 'step_name': '变量预测力一致性选择', 
                                               'df_sample_num': model_bin_df.shape[0], 'df_var_num': model_bin_df.shape[1],
                                               'df_name': 'model_bin_df', 'df_var_ls': model_bin_df.columns.tolist()})])
'''

#==========================================WOE计算========================================= 

# 入模变量woe筛选

## 手工剔除变量
model_woe_stat = bin_step_iv_cal_df[bin_step_iv_cal_df['NewVarName'].isin(psi_low_var_ls)]
model_woe_stat = model_woe_stat.drop('IVItem', axis=1)

## WOE宽表
model_woe_df = woe_transfer(inRawDf = ins_corred_df,
                            inMapDf = model_woe_stat,
                            keyVar = var_key,
                            targetVar = var_target)


# WOE结果excel输出
to_new_excel(filePath = file_path,
             fileName = 'WOE',
             sheetName = 'WOE',
             dataFrm = model_woe_stat)


print('===================  WOE计算完成！ =====================')
times = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print('=================== ', times)

    
    
#==========================================模型训练========================================= 

## 向前法选择入模变量
aic_exclude_var_ls = ['request_id','TargetBad','woe_bin_bj_xy_big_16','woe_bin_bj_xy_small_17','woe_bin_bj_xy_small_18']
aic_exclude_var_ls = ['request_id','TargetBad']
aic_forward_df = lr_forward_select(inDf = model_woe_df,
                        xVarNameLst = list(filter(lambda x: x not in aic_exclude_var_ls, 
                                                  model_woe_df.columns.tolist())),
                        yVarName = 'TargetBad')
print('AIC结果：\n', aic_forward_df[['AIC','AIC_Fall']])


## 创建模型
aic_var_num = input("请输入入模变量数量：")
aic_var_num = int(aic_var_num)
model_build_rst = lr_sklearn_model(inDf = model_woe_df,
                                    xVarNameLs = aic_forward_df.ix[aic_var_num]['xVarNameList'].split(','),
                                    yVarName ='TargetBad')
model_pred_df = model_build_rst['pred_df']
model_coef_df = model_build_rst['coef_df']


##模型建设检验
hyp_rst = lr_hypothesis_test(inDf = model_woe_df,
                        xVarNameLs = aic_forward_df.ix[aic_var_num]['xVarNameList'].split(','),
                        yVarName ='TargetBad')

##模型效果检验
model_eval_rst = model_prob_evaluation(inDf = model_pred_df,
                               predVarName = 'y_pred',
                               yVarName = var_target,
                               filePath = file_path+'\pic',
                               namePrefix = 'INS_')


#==========================================OOT评估========================================= 

## OOT-WOE宽表
model_var_lst = list(filter(lambda x: x not in ['Intercept'], 
                            model_coef_df['VarName'].map(lambda x: x.replace('woe_','')).tolist()))
eval_woe_df = woe_transfer(inRawDf = model_raw_df,
                            inMapDf = model_woe_stat[model_woe_stat['NewVarName'].isin(model_var_lst)],
                            keyVar = var_key,
                            targetVar = var_target,
                            keepVarLst = ['SampleType'])

## 部署模型公式，生成全部样本的预测概率值
eval_pred_df = lr_formula_deployment(inWoeDf = eval_woe_df,
                                     inCoefDf = model_coef_df)

## 验证部署的预测值与建模的预测值是否完全相等
predict_compare(insDf = model_pred_df,
                 evalDf = eval_pred_df,
                 keyVarName = var_key,
                 SampleTypeVarName = 'SampleType',
                 insValue = 'INS')

## 计算PSI
psi_by_oot_df = psi_cal(inEvalDf = eval_pred_df,
                                  xVarName = 'y_pred',
                                  psiBy = 'OOT')


## OOT样本的Lift chart、 KS、 ROC(以INS分箱标准来划分)
oot_pred_df = eval_pred_df[eval_pred_df['SampleType']=='OOT']
oot_eval_rst = oot_prob_evaluation(inDf = oot_pred_df, 
                                    predVarName = 'y_pred', 
                                    yVarName = var_target, 
                                    liftChartDf = model_eval_rst['lift_chart_rst'], 
                                    liftChartBinName = 'y_pred', 
                                    ksDf = model_eval_rst['ks_rst']['KSCurveDat'], 
                                    ksBinName = 'y_pred', 
                                    filePath = file_path+'\pic', 
                                    namePrefix='OOT_')

del model_var_lst



#==========================================生成评分卡========================================= 

## 生成评分卡
score_df = score_calculate(pdo = 40,
                            odds = 0.036,
                            oddsScore = 600,
                            woeDf = model_woe_stat,
                            coefDf = model_coef_df)
## 评分转换
model_score_df = scorecards(inDf=model_woe_df,
                             inScoreDf=score_df,
                             keepVarLs=[var_key,var_target])
## 模型效果评估
model_score_evaluation(inDf = model_score_df,
                               predVarName = 'score',
                               yVarName = var_target,
                               filePath = file_path+'\pic',
                               namePrefix = 'INS_SCORE_')


#==========================================模型变量参数输出========================================= 

## 模型参数估计及检验
model_par_df = model_coef_df.merge(hyp_rst['parameter_estimation'].rename(columns={'':'VarName'}), on='VarName', how='left')
model_par_df = model_par_df[['VarName','Coefficient','std err','t','P>|t|','[0.025','0.975]','IV']]
model_par_df = model_par_df.merge(hyp_rst['vif'], on='VarName', how='left')
tmp_var_psi_df = var_psi_df[['VarName','var_psi']].drop_duplicates()
model_par_df = model_par_df.merge(tmp_var_psi_df['var_psi'], left_on='VarName', right_on = 'woe_' + tmp_var_psi_df['VarName'], how='left')
# model_par_df = model_par_df.merge(var_comment_df)

## 入模变量逾期率及WOE
model_var_df = score_df.copy()
model_var_df['NewVarName'] = model_var_df['VarName'].map(lambda x: x.replace('woe_',''))
model_var_df['WOE'] = model_var_df['WOE'].map(lambda x: np.nan if x == '--' else float(x))
model_var_df = model_var_df.merge(model_woe_stat[['NewVarName','WOE_adjust','Levels']], left_on=['NewVarName','WOE'], right_on=['NewVarName','WOE_adjust'], how='left' )
model_var_df = model_var_df.merge(bin_step_dist_df, on=['NewVarName','Levels'], how='left')
model_var_df = model_var_df[['NewVarName', 'Bins', 0, 1, 'All', 'Rate', 'WOE', 'Score', 'Dclass']]
model_var_df = model_var_df.fillna('--')

model_var_df.to_csv(r'F:\data\coef.csv', encoding='gbk')












