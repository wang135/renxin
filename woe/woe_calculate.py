# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def woe_cal(inDf, xVarName, yVarName):
    '''
    '''
    inDf[xVarName] = inDf[xVarName].astype(str)
    total = inDf.groupby(xVarName)[yVarName].count()             ## 计算xVarName每个分组的样本数量(不包括缺失值)
    total.index = total.index.tolist()
    total_miss = inDf[xVarName].isnull().sum()
    if total_miss > 0:
        total = total.append(pd.Series(total_miss, index=['Null']))  ## 增加xVarName的缺失值样本数量
    total_df = pd.DataFrame({'All': total}) 

    bad = inDf.groupby(xVarName)[yVarName].sum()                 ## 计算xVarName每个分组的坏样本数量
    bad.index = bad.index.tolist()
    if total_miss > 0:
        bad_miss = inDf[inDf[xVarName].isnull()][yVarName].sum()
        bad = bad.append(pd.Series(bad_miss, index=['Null']))        ## 增加xVarName的缺失值样本数量
    bad.name = 'Bad'
    
    woe_df = total_df.merge(bad, right_index=True, left_index=True, how='left')
    
    N = sum(woe_df['All'])  ##计算总样本数量
    B = sum(woe_df['Bad'])    ##计算坏人总样本数量
    
    woe_df['Good'] = woe_df['All']-woe_df['Bad']     ##计算xVarName每个分组的好样本数量
    G = N-B                  ##计算好人总样本数量
    
    woe_df['BadDistribution'] = woe_df['Bad'].map(lambda x: x*1.0/B)     ##计算xVarName每个分组的坏人占总坏人数量的比例分布
    woe_df['GoodDistribution'] = woe_df['Good'].map(lambda x: x*1.0/G)   ##计算xVarName每个分组的好人占总好人数量的比例分布
    
    woe_df['WOE'] = round(np.log(woe_df['BadDistribution'] / woe_df['GoodDistribution']),6)  ##计算xVarName每个分组的WOE值
    
    ## 用平均值修正正负无穷的WOE值
    adjust_df = woe_df[((woe_df['Bad']>0) & (woe_df['Good']>0))]
    adjust_value = round(adjust_df['WOE'].mean(),6)
    woe_df['WOE_Adjust'] = woe_df['WOE']
    woe_df['WOE_Adjust'][((woe_df['Bad']==0)|(woe_df['Good']==0))] = adjust_value
    
    return woe_df



## 类别性
def woe_df_cal(inDf, xVarList, yVarName):
    '''
    inDf = ins_clean_df
    xVarList = ['gender']
    yVarName = 'TargetBad'
    woe_df_cal(inDf, xVarList, yVarName)
    '''
    col_name_woe_ls = [ 'VarName', 'Levels', 'Bins', 'All', 'Bad', 'Good', 'BadDistribution', 'GoodDistribution', 'WOE', \
                       'WOE_Adjust']        
    woe_df = pd.DataFrame(columns = col_name_woe_ls)
    
    for var_item in xVarList:
        one_woe_df = woe_cal(inDf = inDf, xVarName = var_item, yVarName = yVarName)
        one_woe_df = one_woe_df.reset_index(drop = False).rename(columns = {'index': 'Bins'})
        one_woe_df = one_woe_df.reset_index(drop=False).rename(columns = {'index': 'Levels'})
        one_woe_df['VarName'] = var_item
        woe_df = pd.concat([woe_df, one_woe_df], axis=0)
    woe_df = woe_df[col_name_woe_ls]
    return woe_df























