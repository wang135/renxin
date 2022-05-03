# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 18:08:11 2020

@author: finup
"""

import pandas as pd
from base import cols_one_hot


def alg_to_feature(alg, x_df, alg_method):
    '''
    Descriptions:
        把已训练好的模型的每个弱分类器的生成叶子节点，并把叶子节点转换为二值变量。目前支持的算法有：GBDT、XGBoost、
    LightGBM、CatBoost。
    
    Parameters
    ----------
    alg:        已训练好、带有参数的算法
    x_df:    待转换为特征的自变量数组，注意：此数组的变量数量、顺序要与训练的变量数量、顺序保持一致。
    alg_method: 带转换的算法名称
    
    Examples
    --------
    1. GBDT
    gbdt_model = sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.02)
    gbdt_model.fit(xgb_ins_x, ins_clean_df[var_target])
    alg = gbdt_model
    x_df = xgb_ins_x
    alg_method = 'gbdt'
    (leaf_df, leaf_feature_df) = alg_to_feature(alg, x_df, alg_method)
    
    2. XGBoost
    params = {'booster': 'gbtree',
              'eta': 0.3,}
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(xgb_ins_x.values, ins_clean_df[var_target].values)
    alg = xgb_model
    x_df = xgb_ins_x
    alg_method = 'xgboost'
    (leaf_df, leaf_feature_df) = alg_to_feature(alg, x_df, alg_method)
    
    3. LightGBM
    lgb_model = lgb.LGBMClassifier(**params)
    lgb_model.fit(lgb_ins_x.values, ins_clean_df[var_target].values)
    alg = lgb_model
    x_df = xgb_ins_x
    alg_method = 'lightgbm'
    (leaf_df, leaf_feature_df) = alg_to_feature(alg, x_df, alg_method)
    
    4. CatBoost
    cbt_model = cbt.CatBoostClassifier(**params)
    cbt_model.fit(ins_clean_df[x_list], 
                  ins_clean_df[var_target],
                  cat_features=catgory_list,
                  verbose = 0)
    alg = cbt_model
    x_df = xgb_ins_x
    alg_method = 'catboost'
    (leaf_df, leaf_feature_df) = alg_to_feature(alg, x_df, alg_method)
    
    Returns
    -------
    leaf_df - 叶子节点特征数据框
    leaf_feature - 叶子节点转化为二值变量的特征数据框
    '''
    
    if alg_method.lower() == 'gbdt':
        ## 生成叶子节点
        leaf = alg.apply(x_df.values)[:,:,0].astype('int')
        ## 生成叶子节点数据框
        leaf_df = pd.DataFrame(leaf, columns=['{}_boost{}'.format('gbdt',i) for i in range(leaf.shape[1])])
        ## 叶子节点转化为二值变量
        leaf_feature = cols_one_hot(inDf = leaf_df, varLst=leaf_df.columns.tolist(), method='onehot')

    elif alg_method == 'xgboost':
        ## 生成叶子节点
        leaf = alg.apply(x_df.values)
        ## 生成叶子节点数据框
        leaf_df = pd.DataFrame(leaf, columns=['{}_boost{}'.format('xgb',i) for i in range(leaf.shape[1])])
        ## 叶子节点转化为二值变量
        leaf_feature = cols_one_hot(inDf = leaf_df, varLst=leaf_df.columns.tolist(), method='onehot')

    elif alg_method == 'lightgbm':
        ## 生成叶子节点
        leaf = alg.predict_proba(x_df.values, pred_leaf=True)
        ## 生成叶子节点数据框
        leaf_df = pd.DataFrame(leaf, columns=['{}_boost{}'.format('lgb',i) for i in range(leaf.shape[1])])
        ## 叶子节点转化为二值变量
        leaf_feature = cols_one_hot(inDf = leaf_df, varLst=leaf_df.columns.tolist(), method='onehot')
        
    elif alg_method == 'catboost':
        ## 生成叶子节点
        leaf = alg.calc_leaf_indexes(x_df)
        ## 生成叶子节点数据框
        leaf_df = pd.DataFrame(leaf, columns=['{}_boost{}'.format('cbt',i) for i in range(leaf.shape[1])])
        ## 叶子节点转化为二值变量
        leaf_feature = cols_one_hot(inDf = leaf_df, varLst=leaf_df.columns.tolist(), method='onehot')

    return [leaf_df, leaf_feature]











