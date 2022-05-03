# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:41:49 2020

@author: finup
"""


import pandas as pd
import numpy as np
import catboost as cbt
from datetime import datetime as dtime
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

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
train_modify['Processing_Fee_Missing'] = train_modify['Processing_Fee'].isnull().apply(lambda x: 1 if x else 0)
train_modify['Source'] = train_modify['Source'].apply(lambda x: x if x in ['S122','S133'] else 'other')

#train_modify = cols_one_hot(train_modify, ['Gender','Mobile_Verified','Salary_Account','Var1','Filled_Form','Device_Type','Var2','Source']) 

train_modify = train_modify.drop(['City','DOB','EMI_Loan_Submitted','Employer_Name','Interest_Rate',
                                  'Lead_Creation_Date','Loan_Amount_Submitted','Loan_Tenure_Submitted',
                                  'LoggedIn','Processing_Fee'], axis=1)
train_modify = train_modify.fillna(0)

target = 'Disbursed'
IDcol = 'ID'



##==================================================================================================================##


x_list = [x for x in train_modify.columns if x not in [target,IDcol]]

pool_train = cbt.Pool(train_modify[x_list], 
                      label=train_modify[target],
                      cat_features=['Gender','Mobile_Verified','Salary_Account','Var1','Filled_Form','Device_Type',
                                    'Var2','Source','Interest_Rate_Missing','Loan_Amount_Submitted_Missing',
                                    'Loan_Tenure_Submitted_Missing','Processing_Fee_Missing'])


params = {'loss_function': 'Logloss',
          'custom_metric': 'AUC',
          'eval_metric': 'AUC',
          }
cls = CatBoostClassifier(**params)
cls.fit(train_modify[x_list], train_modify[target], 
        cat_features = ['Gender','Mobile_Verified','Salary_Account','Var1','Filled_Form','Device_Type',
                                    'Var2','Source','Interest_Rate_Missing','Loan_Amount_Submitted_Missing',
                                    'Loan_Tenure_Submitted_Missing','Processing_Fee_Missing'],
        verbose=0)

y_pred = cls.predict_proba(train_modify[x_list])[:,1]
roc_auc_score(train_modify[target].values, y_pred)

variable_importance = pd.DataFrame({'VarName': cls.feature_names_,
                                   'Importance': cls.get_feature_importance()})
variable_importance = variable_importance.sort_values('Importance', ascending=False)



##========================================================================================================##


## 网格搜索调参
from itertools import product
import catboost as cbt

def cat_grid_cv(pool, params, grid_params,
                early_stopping_rounds = 10, nfold=5):
    '''
    Function Descriptions:
    用于网格搜索调参 
    
    Examples
    --------
    pool = pool_train
    params = {'loss_function': 'Logloss',
              'learning_rate': 0.03,
              'l2_leaf_reg': 3,
              'depth': 6,
              'min_data_in_leaf': 1,
              'grow_policy': 'SymmetricTree',
              'custom_metric': 'AUC:hints=skip_train~false',
              'eval_metric': 'AUC'}
    grid_params = {'grow_policy': ['Depthwise', 'SymmetricTree']}
    #grid_params = None
    num_boost_round = 50
    early_stopping_rounds = 10
    nfold = 4
       
    '''    

    cv_result_name = ['num_boosters',
                      'train-{}-mean'.format(params['eval_metric']),'train-{}-std'.format(params['eval_metric']),
                      'test-{}-mean'.format(params['eval_metric']),'test-{}-std'.format(params['eval_metric'])]
    
    if isinstance(grid_params, dict):
        #获取需要更新的参数名称
        grid_key_list = list(grid_params.keys())
        cv_result = pd.DataFrame(columns=tuple(grid_key_list+cv_result_name))
        for grid_value in product(*tuple(grid_params.values())):
            key_dict = dict()
            for grid_i in range(len(grid_key_list)):
                params[grid_key_list[grid_i]] = grid_value[grid_i]
                key_dict[grid_key_list[grid_i]] = grid_value[grid_i]
            cv_best_res = cbt.cv(pool, params, nfold=nfold,
                                 early_stopping_rounds=early_stopping_rounds, 
                                 seed = 10, verbose = 0)
            cv_best_res = pd.DataFrame(cv_best_res).iloc[-early_stopping_rounds,:]
            cv_best_res = cv_best_res[cv_result_name]
            print('{} train is over!'.format(key_dict))
            key_dict.update(cv_best_res.to_dict())
            eval_row = pd.DataFrame.from_dict(key_dict, orient='index').T
            eval_row['num_boosters'] = cv_best_res.name+1
            cv_result = pd.concat([cv_result, eval_row], axis=0, ignore_index=True, sort=True)
        cv_result = cv_result[grid_key_list+cv_result_name]
        cv_result['{}_dif_pct'.format(params['eval_metric'])] = ((cv_result['train-{}-mean'.format(params['eval_metric'])] - 
                                           cv_result['test-{}-mean'.format(params['eval_metric'])]) / 
                                           cv_result['train-{}-mean'.format(params['eval_metric'])]).apply(lambda x: round(x,4))

    else :
        cv_result = cbt.cv(pool, params, nfold=nfold,
                           early_stopping_rounds=early_stopping_rounds, 
                           seed = 10, verbose = 0)
        cv_result['{}_dif_pct'.format(params['eval_metric'])]  = round((cv_result['test-{}-mean'.format(params['eval_metric'])] - 
                                           cv_result['train-{}-mean'.format(params['eval_metric'])]) / 
                                           cv_result['train-{}-mean'.format(params['eval_metric'])], 4)
        print('Best Boosters Size: {}'.format(cv_result.iloc[-1,:].name+1-early_stopping_rounds))
    
    return cv_result



##=======================================================================================================##

params = {'loss_function': 'Logloss',
          'custom_metric': 'AUC:hints=skip_train~false',
          'eval_metric': 'AUC'}
grid_params = None
cv_res0 = cat_grid_cv(pool_train, params, grid_params, nfold=5)
## train:0.883191  test:0.84495



##=======================================================================================================##


params = {'loss_function': 'Logloss',
          'learning_rate': 0.03,
          'iterations': 1000,
          'l2_leaf_reg': 3,
          'depth': 6,
          'min_data_in_leaf': 1,
          #'border_count': 254, #连续变量分箱数量
          'grow_policy': 'SymmetricTree',
          'od_type': 'Iter', ##过拟合识别后的处理方法
          #'od_wait': 500,
          'task_type': 'CPU',
          #'metric_period': 500, ## 在GPU下加速计算
          'custom_metric': 'AUC:hints=skip_train~false',
          'eval_metric': 'AUC'}
grid_params = None
cv_res1 = cat_grid_cv(pool_train, params, grid_params,
                      early_stopping_rounds = 20, nfold=5)
## train:0.887165  test:0.845308


grid_params = {'learning_rate': [0.03,0.04,0.05]}
cv_res2 = cat_grid_cv(pool_train, params, grid_params,
                      early_stopping_rounds = 20, nfold=5)
## train:0.890013  test:0.845704
# learning_rate:0.05 
# iterations：335


grid_params = {'learning_rate': [0.05],
               'iterations': [500],
               'l2_leaf_reg': [1,2,3,4,5]}
cv_res3 = cat_grid_cv(pool_train, params, grid_params,
                      early_stopping_rounds = 20, nfold=5)
## train:0.890013  test:0.845704
# l2_leaf_reg: 3


grid_params = {'learning_rate': [0.05],
               'iterations': [500],
               'l2_leaf_reg': [3],
               'depth': [7,8,9]}
cv_res4 = cat_grid_cv(pool_train, params, grid_params,
                      early_stopping_rounds = 20, nfold=5)
## train:0.910929  test:0.847417
# depth: 8



grid_params = {'learning_rate': [0.01,0.05,0.1],
               'iterations': [2500,500,250],
               'l2_leaf_reg': [3],
               'depth': [8]}
cv_res5 = cat_grid_cv(pool_train, params, grid_params,
                      early_stopping_rounds = 20, nfold=5)
## train:0.910929  test:0.847417
# learning_rate: 0.05
# iterations: 258




from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import time

params = {'loss_function': 'Logloss',
          'learning_rate': 0.05,
          'iterations': 258,
          'l2_leaf_reg': 3,
          'depth': 8,
          'task_type': 'GPU',
          'custom_metric': 'AUC',
          'eval_metric': 'AUC'}

begin_time = time.time()
cbt_class_model = CatBoostClassifier(**params)
cbt_class_model.fit(train_modify[x_list], 
                      train_modify[target],
                      cat_features=['Gender','Mobile_Verified','Salary_Account','Var1','Filled_Form','Device_Type',
                                    'Var2','Source','Interest_Rate_Missing','Loan_Amount_Submitted_Missing',
                                    'Loan_Tenure_Submitted_Missing','Processing_Fee_Missing'],
                      verbose = 0)
y_pred = cbt_class_model.predict_proba(train_modify[x_list])[:,1]
roc_auc_score(train_modify[target].values, y_pred)
end_time = time.time()
print('执行时间：', end_time-begin_time)








































