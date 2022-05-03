# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 11:53:32 2020

@author: finup
"""

import pandas as pd
import lightgbm as lgb
from datetime import datetime as dtime

import sys
sys.path.append(r'F:\Python\CreditScoreCard')
from base import cols_one_hot


##https://www.analyticsvidhya.com/wp-content/uploads/2016/02/Dataset.rar
train = pd.read_csv(r'F:\data\Dataset2\Dataset\Train_nyOWmfK.csv')
train.info()

train_modify = train.copy()
train_modify['EMI_Loan_Submitted_Missing'] = train_modify['EMI_Loan_Submitted'].isnull().apply(lambda x: 1 if x else 0)
train_modify['Age'] = train_modify['DOB'].apply(lambda x:round((dtime.now().date() - dtime.strptime(x,'%d-%b-%y').date()).days / 365,0))
train_modify['Existing_EMI'] = train_modify['Existing_EMI'].fillna(0)
train_modify['Interest_Rate_Missing'] = train_modify['Interest_Rate'].isnull().apply(lambda x: 1 if x else 0)
train_modify['Loan_Amount_Applied'] = train_modify['Loan_Amount_Applied'].fillna(train_modify['Loan_Amount_Applied'].median())
train_modify['Loan_Amount_Submitted_Missing'] = train_modify['Loan_Amount_Submitted'].isnull().apply(lambda x: 1 if x else 0)
train_modify['Loan_Tenure_Submitted_Missing'] = train_modify['Loan_Tenure_Submitted'].isnull().apply(lambda x: 1 if x else 0)
train_modify['Processing_Fee_Missing '] = train_modify['Processing_Fee'].isnull().apply(lambda x: 1 if x else 0)
train_modify['Source'] = train_modify['Source'].apply(lambda x: x if x in ['S122','S133'] else 'other')

train_modify = cols_one_hot(train_modify, ['Gender','Mobile_Verified','Salary_Account','Var1','Filled_Form','Device_Type','Var2','Source']) 

train_modify = train_modify.drop(['City','DOB','EMI_Loan_Submitted','Employer_Name','Interest_Rate',
                                  'Lead_Creation_Date','Loan_Amount_Submitted','Loan_Tenure_Submitted',
                                  'LoggedIn','Processing_Fee'], axis=1)


target = 'Disbursed'
IDcol = 'ID'

##==================================================================================================================##

x_list = [x for x in train_modify.columns if x not in [target,IDcol]]

lgb_train = lgb.Dataset(train_modify[x_list], label=train_modify[target])

params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'max_depth': 10,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
        }



train_result0 = lgb.train(params,
                          lgb_train,
                          num_boost_round = 100,
                          valid_sets = lgb_train
                          )

lgb_train.feature_name[6]


train_result0.feature_name()

train_result0.feature_importance()

lgb.plot_importance(train_result0)


train_result0.predict(train_modify[x_list])


##========================================================================================================##


## 网格搜索调参
from itertools import product
import lightgbm as lgb

def lgb_grid_cv(params, dtrain, grid_params,
                num_boost_round=50, early_stopping_rounds = 10, nfold=5, metrics='auc'):
    '''
    Function Descriptions:
    用于网格搜索调参        
    '''    
    if isinstance(grid_params, dict):
        #获取需要更新的参数名称
        grid_key_list = list(grid_params.keys())
        cv_result_name = ['num_boosters',
                          'train {}-mean'.format(params['metric']),'train {}-stdv'.format(params['metric']),
                          'valid {}-mean'.format(params['metric']),'valid {}-stdv'.format(params['metric'])]
        cv_result = pd.DataFrame(columns=tuple(grid_key_list+cv_result_name))
        for grid_value in product(*tuple(grid_params.values())):
            key_dict = dict()
            for grid_i in range(len(grid_key_list)):
                params[grid_key_list[grid_i]] = grid_value[grid_i]
                key_dict[grid_key_list[grid_i]] = grid_value[grid_i]
            cv_best_res = lgb.cv(params, lgb_train, num_boost_round=num_boost_round, nfold=nfold, 
                                 metrics=metrics, early_stopping_rounds=early_stopping_rounds, 
                                 eval_train_metric=True)
            cv_best_res = pd.DataFrame(cv_best_res).iloc[-1,:]
            print('{} train is over!'.format(key_dict))
            key_dict.update(cv_best_res.to_dict())
            eval_row = pd.DataFrame.from_dict(key_dict, orient='index').T
            eval_row['num_boosters'] = cv_best_res.name+1
            cv_result = pd.concat([cv_result, eval_row], axis=0, ignore_index=True, sort=True)
        cv_result = cv_result[grid_key_list+cv_result_name]
        cv_result['variance_pct'] = round((cv_result['valid {}-mean'.format(params['metric'])] - 
                                           cv_result['train {}-mean'.format(params['metric'])]) / 
                                           cv_result['train {}-mean'.format(params['metric'])], 4)

    else :
        cv_result = lgb.cv(params, lgb_train, num_boost_round=num_boost_round, nfold=nfold, 
                           metrics=metrics, early_stopping_rounds=early_stopping_rounds, 
                           eval_train_metric=True)
        cv_result =pd.DataFrame(cv_result)
        cv_result['variance_pct']  = round((cv_result['valid {}-mean'.format(params['metric'])] - 
                                           cv_result['train {}-mean'.format(params['metric'])]) / 
                                           cv_result['train {}-mean'.format(params['metric'])], 4)
        print('Best Boosters Size: {}'.format(cv_result.iloc[-1,:].name+1))
    
    return cv_result




##=======================================================================================================##

## K折交叉训练，参数默认设置
params = {
        'boosting_type': 'goss',
        'objective': 'binary',
        'metric': 'auc',
        'verbose': 0
        }
grid_params = None
train_res0 = lgb_grid_cv(params, lgb_train, grid_params=grid_params, nfold=5)

##train:0.934162  test:0.838912






##=======================================================================================================##


## 学习率初步确定
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
grid_params = {'learning_rate': [0.01,0.02,0.03,0.04,0.05]}
train_res1 = lgb_grid_cv(params, lgb_train, grid_params=grid_params,
                         num_boost_round=500, early_stopping_rounds = 20, nfold=5, metrics='auc')
# learning_rate: 0.04 
# boosts: 93
## train-auc: 0.91937
## F5-auc: 0.841208


## 叶子节点 & 树深度
grid_params = {'num_leaves':[23,24,25,28,31],
               'num_depth':[6,8,10],
               'learning_rate': [0.04]}
train_res2 = lgb_grid_cv(params, lgb_train, grid_params=grid_params,
                         num_boost_round=200, early_stopping_rounds = 20, nfold=5, metrics='auc')
# num_leaves: 24
# num_depth: 6
## train-auc: 0.916968
## F5-auc: 0.841943


## 特征比例
grid_params = {'feature_fraction': [0.7,0.8,0.9,1],
               'num_leaves':[24],
               'num_depth':[6],
               'learning_rate': [0.04]}
train_res3 = lgb_grid_cv(params, lgb_train, grid_params=grid_params,
                         num_boost_round=200, early_stopping_rounds = 20, nfold=5, metrics='auc')
# feature_fraction: 0.7
## train-auc: 0.916178
## F5-auc: 0.842346



## 特征比例
grid_params = {'max_bin': [150,180,250,300],
               'feature_fraction': [0.7],
               'num_leaves':[24],
               'num_depth':[6],
               'learning_rate': [0.04]}
train_res4 = lgb_grid_cv(params, lgb_train, grid_params=grid_params,
                         num_boost_round=200, early_stopping_rounds = 20, nfold=5, metrics='auc')
# max_bin: 0.7
## train-auc: 0.916178
## F5-auc: 0.842346































