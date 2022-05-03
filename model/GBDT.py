# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:47:45 2020

@author: finup
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.model_selection import GridSearchCV, cross_val_score

from base import time_consume


def gbdt_grid_cv(X, y, param_grid, scoring='roc_auc', n_jobs=4, cv=4, validation_fraction=0.1):
    '''
    Descriptions:
       通过网格搜索确定GBDT的超参组合
    
    Parameters
    ----------
    X:          自变量序列
    y:          因变量序列
    param_grid: 网格搜索需要的参数字典
    scoring:    模型效果评估统计量
    n_jobs:     并行线程数
    cv：        样本分割份数
    validation_fraction: 测试样本比例
    
    Examples
    --------
    X = ins_clean_df[x_list]
    y = ins_clean_df[y]
    param_grid = {'n_estimators': [50],
                  'learning_rate': [0.03],
                  'max_depth': [4],
                  'subsample': [0.9],
                  'max_features': [0.8], #fraction、sqrt、log2、None
                  'min_samples_split':[50],
                  'random_state':[1000]
                  }
    gbdt_grid_cv(X, y, param_grid, scoring='roc_auc', n_jobs=4, cv=5, validation_fraction=0)
    '''
    t = time_consume()
    if isinstance(param_grid, dict):
        t.start()
        grid_gbdt = GridSearchCV(estimator = GBDT(learning_rate=0.1, n_estimators=100, min_samples_split=500,
                                                  min_samples_leaf=50, max_depth=8, max_features='sqrt',
                                                  validation_fraction=0, subsample=0.8,random_state=10), 
                                 param_grid = param_grid, scoring=scoring, n_jobs=n_jobs,iid=False, cv=cv, 
                                 return_train_score=True, verbose=0)
        grid_gbdt.fit(X, y)
        grid_res = pd.DataFrame({'params':grid_gbdt.cv_results_['params'],
                                  'mean_train_auc':grid_gbdt.cv_results_['mean_train_score'],
                                  'mean_test_auc':grid_gbdt.cv_results_['mean_test_score'],
                                  'std_train_auc':grid_gbdt.cv_results_['std_train_score'],
                                  'std_test_auc':grid_gbdt.cv_results_['std_test_score'],
                                  })
        t.stop()
        return grid_res
    else :
        # 初始参数交叉训练
        t.start()
        raw_gbdt = GBDT()
        cv_score = cross_val_score(raw_gbdt, X, y, cv=cv, scoring=scoring)
        t.stop()
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    





