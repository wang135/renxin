# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:58:36 2020

@author: finup
"""

import numpy as np
import pandas as pd
from datetime import datetime as dtime
import xgboost as xgb  ##原生算法

from xgboost.sklearn import XGBClassifier  #sklearn封装xgboost分类算法
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

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


##================================================================================================================##
##                                                 手工调参
##================================================================================================================##

x_list = [x for x in train_modify.columns if x not in [target, IDcol]]
train_dat = xgb.DMatrix(train_modify[x_list].values, label=train_modify[target].values)

par_dict_0 = {'booster': 'gbtree',
             'eta': 0.3,
             'gamma': 0,
             'max_depth': 6,
             'min_child_weight': 1,
             'subsample': 1,
             'sampling_method': 'uniform',
             'colsample_bytree': 1,
             'lambda': 1,
             'alpha': 0,
             'growth_policy': 'depthwise',  # lossguide(leaf-wise)
             'objective': 'binary:logistic',
             'eval_metric': 'auc',
             'seed': 1000
             }

## K交叉训练，选择最优的booster数量
xgb.cv(par_dict_0, train_dat, num_boost_round=500, nfold=5, metrics='auc', early_stopping_rounds = 30)

## 网格搜索调参
from itertools import product
def xgb_grid_cv(params, dtrain, grid_params,
                num_boost_round=50, early_stopping_rounds = 10, nfold=5, metrics='auc'):
    '''
    Function Descriptions:
    用于网格搜索调参  

    Examples:
    params = {'booster': 'gbtree',
             'eta': 0.3,
             'gamma': 0,
             'max_depth': 6,
             'min_child_weight': 1,
             'subsample': 1,
             'sampling_method': 'uniform',
             'colsample_bytree': 1,
             'lambda': 1,
             'alpha': 0,
             'objective': 'binary:logistic',
             'growth_policy': 'depthwise',  # lossguide(leaf-wise)
             'eval_metric': 'auc',
             'seed': 1000
             }
    grid_params = {'eta',[0.1,0.3,0.5]}
      
    '''
    if isinstance(grid_params, dict):
        #获取需要更新的参数名称
        grid_key_list = list(grid_params.keys())
        cv_result_name = ['num_boosters',
                          'train-{}-mean'.format(params['eval_metric']),'train-{}-std'.format(params['eval_metric']),
                          'test-{}-mean'.format(params['eval_metric']),'test-{}-std'.format(params['eval_metric'])]
        cv_result = pd.DataFrame(columns=tuple(grid_key_list+cv_result_name))
        for grid_value in product(*tuple(grid_params.values())):
            key_dict = dict()
            for grid_i in range(len(grid_key_list)):
                params[grid_key_list[grid_i]] = grid_value[grid_i]
                key_dict[grid_key_list[grid_i]] = grid_value[grid_i]
            cv_best_res = xgb.cv(params, dtrain, num_boost_round, nfold, metrics='auc', 
                                early_stopping_rounds = early_stopping_rounds).iloc[-1,:]
            print('{} train is over!'.format(key_dict))
            key_dict.update(cv_best_res.to_dict())
            eval_row = pd.DataFrame.from_dict(key_dict, orient='index').T
            eval_row['num_boosters'] = cv_best_res.name+1
            cv_result = pd.concat([cv_result, eval_row], axis=0, ignore_index=True, sort=True)
        cv_result = cv_result[grid_key_list+cv_result_name]
        cv_result['variance_pct'] = round((cv_result['test-{}-mean'.format(params['eval_metric'])] - 
                                           cv_result['train-{}-mean'.format(params['eval_metric'])]) / 
                                           cv_result['train-{}-mean'.format(params['eval_metric'])], 4)

    else :
        cv_result = xgb.cv(params, dtrain, num_boost_round, nfold, metrics='auc', 
                           early_stopping_rounds = early_stopping_rounds)
        cv_result['variance_pct']  = round((cv_result['test-{}-mean'.format(params['eval_metric'])] - 
                                           cv_result['train-{}-mean'.format(params['eval_metric'])]) / 
                                           cv_result['train-{}-mean'.format(params['eval_metric'])], 4)
        print('Best Boosters Size: {}'.format(cv_result.iloc[-1,:].name+1))
    
    return cv_result



##=======================================================================================================##
    
par_dict_0 = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'growth_policy': 'depthwise',  # lossguide(leaf-wise)
              'eval_metric': 'auc',
              'seed': 1000
             }
## K交叉训练，默认参数
cv_result0 = xgb_grid_cv(par_dict_0, train_dat, grid_params=None, nfold=5, metrics='auc')
##train:0.926367  test:0.841315



##=======================================================================================================##

par_dict_0 = {'booster': 'gbtree',
             'eta': 0.3,
             'gamma': 0,
             'max_depth': 6,
             'min_child_weight': 1,
             'subsample': 1,
             'sampling_method': 'uniform',
             'colsample_bytree': 1,
             'lambda': 1,
             'alpha': 0,
             'objective': 'binary:logistic',
             'growth_policy': 'depthwise',  # lossguide(leaf-wise)
             'eval_metric': 'auc',
             'seed': 1000
             }
## K交叉训练，使用网格搜索函数选择最优的booster数量
cv_result0 = xgb_grid_cv(par_dict_0, train_dat, grid_params=None,
                         num_boost_round=1000, early_stopping_rounds = 30, nfold=5, metrics='auc')


## 确定最优步长
grid_params = {'eta':[0.15,0.2,0.25]}
cv_result1 = xgb_grid_cv(par_dict_0, train_dat, grid_params=grid_params,
                         num_boost_round=1000, early_stopping_rounds = 30, nfold=5, metrics='auc')
#eta:0.15  num_booster:91
##train-auc-mean:0.921331


## 确定树的深度和最小叶子节点
grid_params = {'max_depth': [4,6,8,10,12],
               'min_child_weight': [7,9,11,13],
               'eta': [0.15]}
cv_result2 = xgb_grid_cv(par_dict_0, train_dat, grid_params=grid_params,
                        num_boost_round=150, early_stopping_rounds = 20, nfold=5, metrics='auc')
#max_depth:6  min_child_weight:13
##train-auc-mean:0.921331
##test-auc-mean:0.844332

## 确定样本和特征的抽样比例
grid_params = {'subsample': [0.9,1],
               'colsample_bytree': [0.7,0.8,0.9],
               'max_depth': [6],
               'min_child_weight': [13],
               'eta': [0.15]}
cv_result3 = xgb_grid_cv(par_dict_0, train_dat, grid_params=grid_params,
                        num_boost_round=150, early_stopping_rounds = 20, nfold=5, metrics='auc')
#subsample:0.9  
#colsample_bytree:0.9
##train-auc-mean:0.898936
##test-auc-mean:0.847656



####--------------------------------------------------------------------
## train:无验证集

par_train_dict = {'booster': 'gbtree',
             'eta': 0.15,
             'gamma': 0,
             'max_depth': 6,
             'min_child_weight': 13,
             'subsample': 0.9,
             'sampling_method': 'uniform',
             'colsample_bytree': 0.9,
             'lambda': 1,
             'alpha': 0,
             'objective': 'binary:logistic',
             'eval_metric': 'auc',
             'seed': 1000
             }


watchlist = [(dtrain,'train')]
train_result = {}  ##模型训练结果，根据eval_metric来的指标来确定
train_model = xgb.train(par_dict_0, train_dat, 78, watchlist, evals_result=train_result, 
                        verbose_eval=0)
train_result


## 获取变量的重要性（不太直观）
xgb.plot_importance(train_model)
## 获取变量的重要性并生成柱状图
feature_importance = pd.Series(train_model.get_fscore()).sort_values(ascending=False)
feature_importance.plot(kind='bar', title='Feature Importances')

## 预测
train_model.predict(train_dat)


## 模型结果及加载
train_model.save_model('xgb_model.model')
load_model = xgb.Booster()
load_model.load_model('xgb_model.model')
load_model.predict(train_dat)






























