# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 14:04:31 2020

@author: finup
"""

import pandas as pd
import lightgbm as lgb
from itertools import product

from base import time_consume


## 网格搜索调参
def lgb_grid_cv(params, dtrain, grid_params, categorical_feature='auto',
                num_boost_round=50, early_stopping_rounds = 10, nfold=5, metrics='auc'):
    '''
    Function Descriptions:
        用于LightGBM的网格搜索调参

    Parameters
    ----------
    params: 初始参数设置
    dtrain: 模型训练数据集
    grid_params: 网格搜索参数设置
    num_boost_round: 弱分类器数量
    early_stopping_rounds: 经过多少次分类器迭代模型效果无法提升，则停止训练
    nfold: 交叉验证的样本分割份数
    metrics: 模型效果评估指标

    Examples
    --------
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
    dtrain = lgb.Dataset(train_modify[x_list], label=train_modify[target])
    grid_params = {'learning_rate': [0.01,0.02,0.03,0.04,0.05]}
    lgb_grid_cv(params, dtrain, grid_params,
                num_boost_round=50, early_stopping_rounds = 10, nfold=5, metrics='auc')
    '''    
    t = time_consume()
    if isinstance(grid_params, dict):
        #获取需要更新的参数名称
        grid_key_list = list(grid_params.keys())
        cv_result_name = ['num_boosters',
                          'train {}-mean'.format(params['metric']),'train {}-stdv'.format(params['metric']),
                          'valid {}-mean'.format(params['metric']),'valid {}-stdv'.format(params['metric'])]
        cv_result = pd.DataFrame(columns=tuple(grid_key_list+cv_result_name))
        for grid_value in product(*tuple(grid_params.values())):
            t.start()
            key_dict = dict()
            for grid_i in range(len(grid_key_list)):
                params[grid_key_list[grid_i]] = grid_value[grid_i]
                key_dict[grid_key_list[grid_i]] = grid_value[grid_i]
            cv_best_res = lgb.cv(params, dtrain, num_boost_round=num_boost_round, nfold=nfold, 
                                 metrics=metrics, early_stopping_rounds=early_stopping_rounds, 
                                 eval_train_metric=True, categorical_feature=categorical_feature)
            cv_best_res = pd.DataFrame(cv_best_res).iloc[-1,:]
            print('{} train is over!'.format(key_dict))
            key_dict.update(cv_best_res.to_dict())
            eval_row = pd.DataFrame.from_dict(key_dict, orient='index').T
            eval_row['num_boosters'] = cv_best_res.name+1
            cv_result = pd.concat([cv_result, eval_row], axis=0, ignore_index=True, sort=True)
            t.stop()
        cv_result = cv_result[grid_key_list+cv_result_name]
        cv_result['variance_pct'] = ((cv_result['valid {}-mean'.format(params['metric'])] - 
                                           cv_result['train {}-mean'.format(params['metric'])]) / 
                                           cv_result['train {}-mean'.format(params['metric'])]).apply(lambda x: round(x,4))

    else :
        t.start()
        cv_result = lgb.cv(params, dtrain, num_boost_round=num_boost_round, nfold=nfold, 
                           metrics=metrics, early_stopping_rounds=early_stopping_rounds, 
                           eval_train_metric=True, categorical_feature=categorical_feature)
        cv_result =pd.DataFrame(cv_result)
        cv_result['variance_pct']  = ((cv_result['valid {}-mean'.format(params['metric'])] - 
                                           cv_result['train {}-mean'.format(params['metric'])]) / \
                                           cv_result['train {}-mean'.format(params['metric'])]).apply(lambda x: round(x,4))
        t.stop()
        print('Best Boosters Size: {}'.format(cv_result.iloc[-1,:].name+1))
    
    return cv_result








