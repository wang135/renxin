# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 11:37:48 2020

@author: finup
"""

import pandas as pd
import xgboost as xgb
from itertools import product

from base import time_consume



## 网格搜索调参
def xgb_grid_cv(params, dtrain, grid_params,
                num_boost_round=50, early_stopping_rounds = 10, nfold=5, metrics='auc'):
    '''
    Function Descriptions:
    用于网格搜索调参  

    Examples
    --------
    x_list = [x for x in train_modify.columns if x not in [target, IDcol]]
    train_dat = xgb.DMatrix(train_modify[x_list].values, label=train_modify[target].values)
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
    xgb_grid_cv(params, dtrain, grid_params,
                num_boost_round=50, early_stopping_rounds = 10, nfold=5, metrics='auc')
    '''
    t = time_consume()
    if isinstance(grid_params, dict):
        #获取需要更新的参数名称
        grid_key_list = list(grid_params.keys())
        cv_result_name = ['num_boosters',
                          'train-{}-mean'.format(params['eval_metric']),'train-{}-std'.format(params['eval_metric']),
                          'test-{}-mean'.format(params['eval_metric']),'test-{}-std'.format(params['eval_metric'])]
        cv_result = pd.DataFrame(columns=tuple(grid_key_list+cv_result_name))
        for grid_value in product(*tuple(grid_params.values())):
            t.start()
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
            t.stop()
        cv_result = cv_result[grid_key_list+cv_result_name]
        cv_result['variance_pct'] = ((cv_result['test-{}-mean'.format(params['eval_metric'])] - 
                                       cv_result['train-{}-mean'.format(params['eval_metric'])]) / \
                                       cv_result['train-{}-mean'.format(params['eval_metric'])]).apply(lambda x: round(x,4))

    else :
        t.start()
        cv_result = xgb.cv(params, dtrain, num_boost_round, nfold, metrics='auc', 
                           early_stopping_rounds = early_stopping_rounds)
        cv_result['variance_pct']  = ((cv_result['test-{}-mean'.format(params['eval_metric'])] - 
                                       cv_result['train-{}-mean'.format(params['eval_metric'])]) / \
                                       cv_result['train-{}-mean'.format(params['eval_metric'])]).apply(lambda x: round(x,4))
        t.stop()
        print('Best Boosters Size: {}'.format(cv_result.iloc[-1,:].name+1))
    
    return cv_result





