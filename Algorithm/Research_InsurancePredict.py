# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:47:07 2020

@author: finup
"""


import sys
sys.path.append(r'F:\Python\CreditScoreCard')


from base import variable_char_type
from base import cols_one_hot
from explore import variable_summary

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost 
import lightgbm

from sklearn.metrics import roc_auc_score

df = pd.read_csv(r'F:\data\ClaimPredictionChallenge\train_set.csv', low_memory=False)

sample = df.sample(50000, random_state=0)

## 参数设置
key_var = 'Row_ID'
y_var = ''


## 变量类型数据集
var_class_df = variable_char_type(inDf = sample, 
                                  keyVarList=[key_var], 
                                  TargetVarList=['y_var'], 
                                  unScaleVarList=[])


## 变量探索
explore_res = variable_summary(inDf = sample, inVarClassDf = var_class_df)


## 变量处理
sample['used_year'] = sample['Calendar_Year'] - sample['Model_Year']
sample['is_claim'] = sample['Claim_Amount'].apply(lambda x: 1 if x>0 else 0)
sample = sample.drop(['Model_Year','Claim_Amount'], axis=1)

model_df = cols_one_hot(inDf = sample, varLst = var_class_df[var_class_df['Dclass']=='Nominal']['index'].tolist())


model_var_class_df = variable_char_type(inDf = model_df, 
                                          keyVarList=[key_var], 
                                          TargetVarList=['is_claim'], 
                                          unScaleVarList=[])
model_var_class_df.loc[model_var_class_df['index'].isin(['Household_ID','Vehicle']),'Dclass'] = 'Droped'

X_matrix = model_df[model_var_class_df[model_var_class_df['Dclass'].isin(['Order','Continuous','Nominal','Binary'])]['index'].tolist()]

## 建模

## Random Forest

rf = RandomForestClassifier()
rf.fit(X_matrix.values, model_df['is_claim'].values)
pred_rf = rf.predict_proba(X_matrix.values)[:,1]

roc_auc_score(model_df['is_claim'].values, pred_rf)









































