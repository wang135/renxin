# -*- coding: utf-8 -*-

import sys
sys.path.append(r'F:\Python\CreditScoreCard')


import numpy as np
import pandas as pd
import datetime
import math
from statsmodels.formula.api import ols

#from base import *

from base import load_from_csv
from bins import step_bin
from analysis import corr_class_variables
from analysis import iv_df_auto_calculate
from bins import continuous_df_rate_bin


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
tz_blacklist = load_from_csv(filePath = r'F:\data\三方提测样本返回\TZ_yuQi.csv', charTrans=False)
tz_blacklist = tz_blacklist[['idCard','backTime','m1逾期提醒事件累计次数','m3逾期提醒平台累计次数',
                             'm6逾期提醒平台累计次数','最近一次逾期提醒事件距当前时间天数']]
tz_blacklist = tz_blacklist.rename(columns = {'idCard':'id_no', 'backTime':'apply_time'})

tz_blacklist = tz_blacklist.drop_duplicates()
dup_ser = tz_blacklist.groupby(['id_no','apply_time'])['id_no'].count()
tz_blacklist = tz_blacklist[~tz_blacklist['id_no'].isin(dup_ser[dup_ser>1].unstack().index.tolist())]
tz_blacklist['id_no'] = tz_blacklist['id_no'].map(lambda x: x.upper())


## 百融（）
br_blacklist = pd.read_csv(r'F:\data\三方提测样本返回\BR_SpecialList_c.csv', header=0, skiprows=[1])
br_blacklist = br_blacklist.drop(['cus_num', 'name', 'cell', 'sl_user_date', 'user_time', 'custApiCode', 'swift_number',
                                  'cus_username', 'code'], axis=1)
br_blacklist = br_blacklist.rename(columns = {'id':'id_no', 'user_date':'apply_time'})
br_blacklist = br_blacklist.drop_duplicates()
dup_ser = br_blacklist.groupby(['id_no','apply_time'])['id_no'].count()
br_blacklist = br_blacklist[~br_blacklist['id_no'].isin(dup_ser[dup_ser>1].unstack().index.tolist())]
br_blacklist['id_no'] = br_blacklist['id_no'].map(lambda x: x.upper())
br_blacklist = br_blacklist[['id_no', 'apply_time', 'flag_specialList_c']]

## 银联智策
ylzc_blacklist = pd.read_csv(r'F:\data\三方初步入选数据\ylzc_special.csv')
ylzc_blacklist = ylzc_blacklist[['request_id','TSJY002','TSJY004','TSJY009','TSJY017','TSJY047']]
ylzc_blacklist = ylzc_blacklist.drop_duplicates()


## target
df_target = pd.read_csv(r'F:\data\三方提测样本返回\final_encrypt_sample.csv')
df_target = df_target[['request_id', 'date', 'encode_id_no', 'if_loan', 'if_now_m1p']]
df_target = df_target.rename(columns = {'date':'apply_time', 'encode_id_no':'id_no'})



df_table = df_target.merge(heit_blacklist, on=['id_no','apply_time'], how='left')
df_table = df_table.merge(tz_blacklist, on=['id_no','apply_time'], how='left')
df_table = df_table.merge(br_blacklist, on=['id_no','apply_time'], how='left')
df_table = df_table.merge(ylzc_blacklist, on='request_id', how='left')


## 样本选择
df_analy = df_table[(df_table['apply_time']<='2020-06-13') & (~df_table['if_now_m1p'].isnull())]



var_ls = ['ht_suspect',
         'm1逾期提醒事件累计次数',
         'm3逾期提醒平台累计次数',
         'm6逾期提醒平台累计次数',
         '最近一次逾期提醒事件距当前时间天数',
         'flag_specialList_c',
         'TSJY002',
         'TSJY004',
         'TSJY009',
         'TSJY017',
         'TSJY047']
rawvar_iv_df = iv_df_auto_calculate(inDf = df_analy,
                     xVarList = var_ls,
                     yVar = 'if_now_m1p')

var_powerful_ls = ['ht_suspect',
         'm1逾期提醒事件累计次数',
         '最近一次逾期提醒事件距当前时间天数',
         'TSJY002',
         'TSJY004',
         'TSJY009',]
for items in var_powerful_ls:
    print(items)
    print(df_analy[items].value_counts(dropna=False).sort_index(), '\n')



corr_df = corr_class_variables(inDf=df_analy, varList=var_ls)




## 连续变量等频分箱后，逐步合并分箱及计算chisq统计量
rate_con_cmb_rst = continuous_df_rate_bin(inDf = df_analy, 
                                          varList = ['m3逾期提醒平台累计次数','最近一次逾期提醒事件距当前时间天数',
                                                     'TSJY002', 'TSJY004', 'TSJY009'],
                                          yVarName = 'if_now_m1p',
                                          n = 8)
rate_con_cmb_freq_df = rate_con_cmb_rst['rate_step_bin_df']


variables = ['bin_m3逾期提醒平台累计次数','bin_最近一次逾期提醒事件距当前时间天数', 'bin_TSJY002', 'bin_TSJY004', 'bin_TSJY009']
steps = [2,5,3,3,6]
cut_select = pd.DataFrame({'VarName': variables, 'Steps':steps})

cut_select = cut_select.merge(rate_con_cmb_freq_df, on=['VarName','Steps'], how='left')



df_analy['m3_alarm_bin'] = df_analy['m3逾期提醒平台累计次数'].apply(lambda x: 1 if x<=3 else 2 if x<=5 else 3)
df_analy['last_alarm_bin'] = df_analy['最近一次逾期提醒事件距当前时间天数'].apply(lambda x: 1 if x<=16 else 2 if x<=25 else 3)
df_analy['TSJY002_bin'] = df_analy['TSJY002'].apply(lambda x: 4 if pd.isnull(x) else 1 if x<=1 else 2 if x<=3 else 3)
df_analy['TSJY004_bin'] = df_analy['TSJY004'].apply(lambda x: 3 if pd.isnull(x) else 1 if x<=1 else 2 )
df_analy['TSJY009_bin'] = df_analy['TSJY009'].apply(lambda x: 3 if pd.isnull(x) else 1 if x<=9900 else 2 )



df_analy.groupby(['ht_suspect'])['if_now_m1p'].mean()


df_analy.groupby(['ht_suspect','m3_alarm_bin'])['if_now_m1p'].mean().unstack()
df_analy.groupby(['ht_suspect','m3_alarm_bin'])['if_now_m1p'].count().unstack()


df_analy.groupby(['ht_suspect','last_alarm_bin'])['if_now_m1p'].mean().unstack()
df_analy.groupby(['ht_suspect','last_alarm_bin'])['if_now_m1p'].count().unstack()


df_analy.groupby(['ht_suspect','m3_alarm_bin','last_alarm_bin'])['if_now_m1p'].mean().unstack()
df_analy.groupby(['ht_suspect','m3_alarm_bin','last_alarm_bin'])['if_now_m1p'].count().unstack()

##风险等级
'''
high risk: ht_suspect = 1 且 m3逾期提醒平台累计次数 > 5 且 最近一次逾期提醒事件距当前时间天数 <= 25
middle risk: ht_suspect = 1 且 (m3逾期提醒平台累计次数 > 3 且 m3逾期提醒平台累计次数 <= 5 且 最近一次逾期提醒事件距当前时间天数 <= 25) 
                                or (m3逾期提醒平台累计次数<=3 且 最近一次逾期提醒事件距当前时间天数 <= 15)
low risk: 其他
'''


def risk_level(x, y, z):
    if x==1 and y>5 and z<=25:
        return 1
    elif x==1 and ((y>3 and z<=25) or (y<=3 and z<=15)):
        return 2
    else:
        return 3

df_analy['risk_level'] = df_analy.apply(lambda x: risk_level(x.ht_suspect, x.m3逾期提醒平台累计次数, x.最近一次逾期提醒事件距当前时间天数), axis=1)

df_analy.groupby('risk_level')['if_now_m1p'].mean()

df_analy.groupby('risk_level')['if_now_m1p'].sum()























