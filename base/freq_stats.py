# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def cross_table(inDf, varX, varY):
    '''
    Function Decription: 
        列联表频数计算
        
    Parameters
    -------------
    inDf : 用于计算列联表的数据框
    varX : 分类自变量名称
    varY : 目标变量名称
    
    Returns
    -------
    列联表数据框
    '''
    # 计算非缺失值样本量
    TmpTab = pd.crosstab(inDf[varX],inDf[varY], dropna=False,margins=False)  
    # 创建空值列
    TmpTab['MissCnt'] = np.nan
    # 计算列合计样本量
    TmpSeri = inDf.groupby(varX).size()
    TmpSeri.name = 'Total'
    TmpTabs = TmpTab.join(TmpSeri, how='left').reset_index(drop=False)
    # 计算分类字段空值的样本量v
    TmpDfMiss = inDf[inDf[varX].isnull()]
    MissDict = dict(TmpDfMiss[varY].value_counts())  # 非缺失值分类样本量
    MissDict['MissCnt'] = TmpDfMiss[varY].isnull().sum()  # 缺失值样本量 
    MissDict['Total'] = TmpDfMiss.shape[0]  # 总样本量
    MissDict[varX] = 'Missing'
    
    if TmpDfMiss.shape[0] > 0:
        TableResult = TmpTabs.append(MissDict, ignore_index=True)
    else:
        TableResult = TmpTabs
    # 列合计    
    TableColumnTotal = TableResult.sum()
    TableColumnTotal[varX] = 'Total'
    TableResult = TableResult.append(TableColumnTotal, ignore_index=True)
    TableResult = TableResult.fillna(0)
    
    return TableResult



def var_freq_dist(x, pctFormat=True):
    '''
    Function Decription
        单指标频数统计,包括缺失值
        
    Parameter
    ---------
    x : 自变量
    
    Returns
    -------
    频数及占比数据框
    '''
    Freq = x.value_counts(dropna=False)
    TotalFreq = pd.Series(sum(Freq),index=['Total'])
    Freqs = pd.concat([Freq,TotalFreq])
    Freqs.name = 'Freq'
    FreqRate = round(Freqs / sum(Freq),2)
    FreqRate.name = 'Rate'
    FreqDf = pd.concat([Freqs,FreqRate],axis=1)
    if pctFormat == True:
        FreqDf['Rate'] = FreqDf['Rate'].apply(lambda x: format(x, '.0%'))
    else :
        pass
    return FreqDf
        






