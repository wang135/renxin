# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from base.freq_order_bin import order_freq_combine_transfer
from base.freq_nominal_bin import nominal_freq_combine_transfer
from base.equal_freq_cut import equal_freq_cut_map



def iv_class_variable(inDf, xVarName, yVarName):
    
    '''
    用于计算自变量为分类变量，目标变量为二值变量的IV值
    Args:
        inDf = ins1
        xVarName = 'education'
        yVarName = 'TargetBad'
    Return:
        列表
    '''
    
    total = inDf.groupby(xVarName)[yVarName].count()             ## 计算xVarName每个分组的样本数量(不包括缺失值)
    total.index = total.index.tolist()
    total_miss = inDf[xVarName].isnull().sum()
    if total_miss > 0:
        total = total.append(pd.Series(total_miss, index=['Null']))  ## 增加xVarName的缺失值样本数量
    total_df = pd.DataFrame({'Total': total}) 

    bad = inDf.groupby(xVarName)[yVarName].sum()                 ## 计算xVarName每个分组的坏样本数量
    bad.index = bad.index.tolist()
    if total_miss > 0:
        bad_miss = inDf[inDf[xVarName].isnull()][yVarName].sum()
        bad = bad.append(pd.Series(bad_miss, index=['Null']))        ## 增加xVarName的缺失值样本数量
    bad.name = 'Bad'
    
    iv_df = total_df.merge(bad, right_index=True, left_index=True, how='left')
    
    N = sum(iv_df['Total'])  ##计算总样本数量
    B = sum(iv_df['Bad'])    ##计算坏人总样本数量
    
    iv_df['Good'] = iv_df['Total']-iv_df['Bad']     ##计算xVarName每个分组的好样本数量
    G = N-B                  ##计算好人总样本数量
    
    iv_df['BadDistribution'] = iv_df['Bad'].map(lambda x: x*1.0/B)     ##计算xVarName每个分组的坏人占总坏人数量的比例分布
    iv_df['GoodDistribution'] = iv_df['Good'].map(lambda x: x*1.0/G)   ##计算xVarName每个分组的好人占总好人数量的比例分布
    
    iv_df['WOE'] = round(np.log(iv_df['BadDistribution'] / iv_df['GoodDistribution']),6)  ##计算xVarName每个分组的WOE值
    iv_df['WOE_adjust'] = iv_df['WOE']    
    iv_df = iv_df.replace({'WOE_adjust': {np.inf: 0, -np.inf: 0}})  ##0样本情况下，调整正负无穷值
    iv_df['IVItem'] = (iv_df['BadDistribution'] - iv_df['GoodDistribution'])*iv_df['WOE_adjust']  ##计算xVarName每个分组的IV值
    
    IV = round(sum(iv_df['IVItem']),4)
 
    
    return {'IV': IV,
            'iv_df': iv_df}


def iv_calculate(inDf, xVarName, yVarName):

    '''
    inDf = ins_corred_df
    xVarName = 'query_org_1m_cnt'
    yVarName = 'TargetBad'
    '''
    print(xVarName, yVarName)
    if inDf[xVarName].dtype != 'object':
        n_var = inDf[xVarName].nunique()
        if (n_var <= 5):
            IV = iv_class_variable(inDf, xVarName, yVarName)
                
        elif (n_var > 5 and n_var <= 10):
            tmp_var = order_freq_combine_transfer(xVar=inDf[xVarName], cutOff=0.03)
            tmp_df = inDf[[yVarName]].merge(tmp_var, left_index=True, right_index=True, how='left')
            IV = iv_class_variable(inDf = tmp_df, 
                                 xVarName = 'bin_%s' % xVarName, 
                                 yVarName = yVarName) 
                
        elif (n_var > 10 and n_var <= 20):
            tmp_var = equal_freq_cut_map(x = inDf[xVarName], nBin=8, mapType='Level')
            tmp_df = inDf[[yVarName]].merge(tmp_var['x_bin_ser'], left_index=True, right_index=True, how='left')
            IV = iv_class_variable(inDf = tmp_df, 
                                 xVarName = 'bin_%s' % xVarName, 
                                 yVarName = yVarName) 
        
        elif (n_var > 20 and n_var <= 50):
            tmp_var = equal_freq_cut_map(x = inDf[xVarName], nBin=10, mapType='Level')
            tmp_df = inDf[[yVarName]].merge(tmp_var['x_bin_ser'], left_index=True, right_index=True, how='left')
            IV = iv_class_variable(inDf = tmp_df, 
                                 xVarName = 'bin_%s' % xVarName, 
                                 yVarName = yVarName) 
            
        elif (n_var > 50 and n_var <= 100):
            tmp_var = equal_freq_cut_map(x = inDf[xVarName], nBin=15, mapType='Level')
            tmp_df = inDf[[yVarName]].merge(tmp_var['x_bin_ser'], left_index=True, right_index=True, how='left')
            IV = iv_class_variable(inDf = tmp_df, 
                                 xVarName = 'bin_%s' % xVarName, 
                                 yVarName = yVarName) 

        elif n_var > 100 :
            tmp_var = equal_freq_cut_map(x = inDf[xVarName], nBin=18, mapType='Level')
            tmp_df = inDf[[yVarName]].merge(tmp_var['x_bin_ser'], left_index=True, right_index=True, how='left')
            IV = iv_class_variable(inDf = tmp_df, 
                                   xVarName = 'bin_%s' % xVarName, 
                                   yVarName = yVarName) 
            
    else :   
        
        tmp_var = nominal_freq_combine_transfer(xVar=inDf[xVarName], cutOff=0.03)
        tmp_df = inDf[[yVarName]].merge(tmp_var, left_index=True, right_index=True, how='left')
        IV = iv_class_variable(inDf = tmp_df, 
                               xVarName = 'bin_%s' % xVarName, 
                               yVarName = yVarName) 
        
    return IV


def iv_df_auto_calculate(inDf, xVarList, yVar):
    '''
    inDf = ins_corred_df
    xVarList = iv_var_ls
    yVar = var_target
    '''
    tmp_ls = list()
    for var_item in xVarList:
        iv_var = iv_calculate(inDf, var_item, yVar)['IV']
        tmp_ls.append([var_item, iv_var])
        #print('IV: ', var_item, 'is completed!')

    IV_df = pd.DataFrame(tmp_ls, columns=['VarName','IV'])
    
    return IV_df




def iv_from_freq(inDf, var0, var1):
    '''
    Funcation Descriptions:
        基于变量频数统计的结果，计算变量的IV值和WOE值
    
    Parameters
    ----------
    inDf : 单个变量的频数统计数据框
    var0 : 好人的样本量变量名称
    var1 : 坏人的样本量变量名称
    
    Returns
    -------
    数据字典：IV - 变量的iv值;  iv_df - iv的计算过程
    
    Examples
    --------
    inDf = rate_con_cmb_freq_df[(rate_con_cmb_freq_df['VarName']=='bin_ages') & (rate_con_cmb_freq_df['Steps']==0)]
    var0 = 0
    var1 = 1
    '''
    iv_df = inDf.rename(columns={0:'Bad', 1:'Good'})
    
    B = iv_df['Bad'].sum()                   ## 好样本量
    G = iv_df['Good'].sum()                  ## 坏样本量
    
    iv_df['BadDistribution'] = iv_df['Bad'].map(lambda x: x*1.0/B)     ##计算xVarName每个分组的坏人占总坏人数量的比例分布
    iv_df['GoodDistribution'] = iv_df['Good'].map(lambda x: x*1.0/G)   ##计算xVarName每个分组的好人占总好人数量的比例分布
    
    iv_df['WOE'] = round(np.log(iv_df['BadDistribution'] / iv_df['GoodDistribution']),6)  ##计算xVarName每个分组的WOE值
    iv_df['WOE_adjust'] = iv_df['WOE']    
    iv_df = iv_df.replace({'WOE_adjust': {np.inf: 0, -np.inf: 0}})  ##0样本情况下，调整正负无穷值
    iv_df['IVItem'] = (iv_df['BadDistribution'] - iv_df['GoodDistribution'])*iv_df['WOE_adjust']  ##计算xVarName每个分组的IV值
    
    IV = round(sum(iv_df['IVItem']),4)
 
    return {'IV': IV,
            'iv_df': iv_df}




def iv_df_from_freq(inDf, varName, var0, var1):
    '''
    Funcation Descriptions:
        基于变量频数统计的结果，计算变量的IV值和WOE值
    
    Parameters
    ----------
    inDf     : 单个变量的频数统计数据框
    varName  : 存放变量名称的变量
    var0     : 好人的样本量变量
    var1     : 坏人的样本量变量
    
    Returns
    -------
    数据字典：iv_value_df - 全部变量的iv值;  iv_cal_df - iv的计算过程
    
    Examples
    --------
    inDf = rate_ord_cmb_freq_df
    varName = 'NewVarName'
    var0 = 0
    var1 = 1
    iv_df_from_freq(inDf, varName, var0, var1)
    '''
    
    name_lst = ['NewVarName', 'VarName', 'Steps', 'Levels', 'Bins', 'All', 'Good', 'Bad', 'Rate',
                'BadDistribution', 'GoodDistribution', 'WOE', 'WOE_adjust', 'IVItem']
    iv_cal_df = pd.DataFrame(columns = name_lst)
    iv_value_df = pd.DataFrame(columns = ['VarName', 'IV', 'df'])
    
    var_lst = inDf[varName].unique().tolist()
    for var_item in var_lst:        
        freq_df = inDf[inDf[varName]==var_item]        
        iv = iv_from_freq(inDf=freq_df, var0=var0, var1=var1)['IV']
        woe_df = iv_from_freq(inDf=freq_df, var0=0, var1=1)['iv_df']
        iv_value_df.loc[len(iv_value_df)] = [var_item, iv, len(freq_df)-1]
        iv_cal_df = pd.concat([iv_cal_df,woe_df], axis=0, ignore_index=True)
    iv_cal_df = iv_cal_df[name_lst]
    
    return {'iv_value_df' : iv_value_df,
            'iv_cal_df' : iv_cal_df}












