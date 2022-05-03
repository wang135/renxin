# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 17:35:24 2020

@author: finup
"""




## 网格搜索调参
from itertools import product
import catboost as cbt
import pandas as pd

from base import time_consume


def cat_grid_cv(pool, params, grid_params,
                early_stopping_rounds = 30, nfold=4):
    '''
    Function Descriptions:
    用于网格搜索调参 
    
    Parameter:
    ---------
    pool: 训练数据
    params: 参数初始设置
    grid_params: 参数搜索值列表数据字典
    early_stopping_rounds: 学习器迭代无法进一步提升的次数
    nfold: 训练样本训练交叉验证的分割份数
    
    Examples
    --------
    pool = cbt.Pool(train_modify[x_list], 
                    label=train_modify[target],
                    cat_features=['Gender','Mobile_Verified','Salary_Account','Var1','Filled_Form','Device_Type',
                                  'Var2','Source','Interest_Rate_Missing','Loan_Amount_Submitted_Missing',
                                  'Loan_Tenure_Submitted_Missing','Processing_Fee_Missing'])
    params = {'loss_function': 'Logloss',
              'learning_rate': 0.03,
              'iterations': 1000,
              'l2_leaf_reg': 3,
              'depth': 6,
              'min_data_in_leaf': 1,
              'grow_policy': 'SymmetricTree',
              'custom_metric': 'AUC:hints=skip_train~false',
              'eval_metric': 'AUC'}
    grid_params = {'grow_policy': ['Depthwise', 'SymmetricTree']}
    #grid_params = None
    early_stopping_rounds = 10
    nfold = 4
    cat_grid_cv(pool, params, grid_params,
                early_stopping_rounds = 30, nfold=4)   
    '''    
    t = time_consume()
    cv_result_name = ['num_boosters',
                      'train-{}-mean'.format(params['eval_metric']),'train-{}-std'.format(params['eval_metric']),
                      'test-{}-mean'.format(params['eval_metric']),'test-{}-std'.format(params['eval_metric'])]
    
    if isinstance(grid_params, dict):
        #获取需要更新的参数名称
        grid_key_list = list(grid_params.keys())
        cv_result = pd.DataFrame(columns=tuple(grid_key_list+cv_result_name))
        for grid_value in product(*tuple(grid_params.values())):
            t.start()
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
            t.stop()
        cv_result = cv_result[grid_key_list+cv_result_name]
        cv_result['{}_dif_pct'.format(params['eval_metric'])] = ((cv_result['train-{}-mean'.format(params['eval_metric'])] - 
                                           cv_result['test-{}-mean'.format(params['eval_metric'])]) / 
                                           cv_result['train-{}-mean'.format(params['eval_metric'])]).apply(lambda x: round(x,4))

    else :
        t.start()
        cv_result = cbt.cv(pool, params, nfold=nfold,
                           early_stopping_rounds=early_stopping_rounds, 
                           seed = 10, verbose = 0)
        cv_result['{}_dif_pct'.format(params['eval_metric'])]  = round((cv_result['test-{}-mean'.format(params['eval_metric'])] - 
                                           cv_result['train-{}-mean'.format(params['eval_metric'])]) / 
                                           cv_result['train-{}-mean'.format(params['eval_metric'])], 4)
        t.stop()
        print('Best Boosters Size: {}'.format(cv_result.iloc[-1,:].name+1-early_stopping_rounds))
    
    return cv_result




















