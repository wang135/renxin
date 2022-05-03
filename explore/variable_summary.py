# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from base.freq_stats import var_freq_dist


def variable_summary(inDf, inVarClassDf ):    
    '''
    Function Description:
        用于计算变量的统计量。
    连续变量生成记录数、缺失值数量、最小值、均值、中位数、最大值、极差、方差、标准差、方差变异系数、偏度、峰度、分位数
    分类变量生成频数分布及占比
    
    Parameters
    ----------
    inDf         : 用于分析的数据框
    inVarClassDf : 变量类型数据框
    
    Returns
    -------
    数据字典：contStatSummary-连续变量统计量数据框 classFreqSummary-分类变量数据框
    {'contStatSummary':tmpDfCon, 
     'classFreqSummary':tmpDfClass}    
    '''
    
    for varType in ['Continuous','Nominal','Binary','Order']:
        if varType == 'Continuous':
            tmpConList = inVarClassDf[inVarClassDf['Dclass'] == 'Continuous']['index'].reset_index(drop=True)
            
            loci = 0
            tmpDfCon = pd.DataFrame(columns=('VarName', 'N', 'NMiss', 'Min', 'Mean', 'Median', 'Max', 
                                          'PTP' , 'Variance', 'Std', 'CV', 'Skew', 'Kurtosis',
                                          'P0', 'P5', 'P10', 'P20', 'P30', 'P40', 'P50', 'P60', 'P70', 'P80', 'P90', 'P95', 'P100'))
            for varItem in tmpConList:
                N = len(inDf[varItem])
                NMiss = inDf[varItem].isna().sum()
                Min = inDf[varItem].min()
                Mean = round(inDf[varItem].mean(),2)
                Median = inDf[varItem].median()
                Max = inDf[varItem].max()
                #Ptp = round(inDf[varItem].ptp(),2)
                Ptp = round(np.ptp(inDf[varItem]),2)
                Variance = round(inDf[varItem].var(),2)
                Std = round(inDf[varItem].std(),2)
                if Mean==0:
                    CV = Std
                else:
                    CV = round(Std/Mean,2)
                Skew = round(inDf[varItem].skew(),2)
                Kurtosis = round(inDf[varItem].kurtosis(),2)
                P0 = np.quantile(inDf[varItem].dropna(),0)
                P5 = np.quantile(inDf[varItem].dropna(),0.05)
                P10 = np.quantile(inDf[varItem].dropna(),0.1)
                P20 = np.quantile(inDf[varItem].dropna(),0.2)
                P30 = np.quantile(inDf[varItem].dropna(),0.3)
                P40 = np.quantile(inDf[varItem].dropna(),0.4)
                P50 = np.quantile(inDf[varItem].dropna(),0.5)
                P60 = np.quantile(inDf[varItem].dropna(),0.6)
                P70 = np.quantile(inDf[varItem].dropna(),0.7)
                P80 = np.quantile(inDf[varItem].dropna(),0.8)
                P90 = np.quantile(inDf[varItem].dropna(),0.9)
                P95 = np.quantile(inDf[varItem].dropna(),0.95)
                P100 = np.quantile(inDf[varItem].dropna(),1)
                
                tmpDfCon.loc[loci] = [varItem, N, NMiss, Min, Mean, Median, Max, Ptp, Variance, Std, CV, Skew, Kurtosis, 
                                 P0, P5, P10, P20, P30, P40, P50, P60, P70, P80, P90, P95, P100]
                loci = loci + 1
        
        else :
            tmpClsList = inVarClassDf[inVarClassDf['Dclass'].isin(['Nominal','Binary','Order'])]['index'].reset_index(drop=True)   
            tmpDfClass = pd.DataFrame(columns=('VarName', 'VarValue', 'ValueFreq', 'ValueRate'))    
            for varItem in tmpClsList:
                 tmpDf = var_freq_dist(inDf[varItem]).reset_index()
                 tmpDf['VarName'] = varItem
                 tmpDf = tmpDf.rename(columns={'index':'VarValue', 'Freq':'ValueFreq', 'Rate':'ValueRate'})
                 tmpDfClass = pd.concat([tmpDfClass, tmpDf], axis=0)
                 tmpDfClass = tmpDfClass[['VarName','VarValue','ValueFreq','ValueRate']]
                 
    return {'contStatSummary':tmpDfCon, 
            'classFreqSummary':tmpDfClass}





def variable_summary_bytype(inDf, varLst, varType ):    
    '''
    Function Description:
        用于对数据框中的变量进行统计描述。
    连续变量生成记录数、缺失值数量、最小值、均值、中位数、最大值、极差、方差、标准差、方差变异系数、偏度、峰度、分位数
    分类变量生成频数分布及占比
    
    Parameters
    ----------
    inDf         : 用于分析的数据框
    varLst       : 变量列表
    varType      : 变量类型, 包括：Continuous、Nominal、Binary、Order
    
    Returns
    -------
    统计结果数据框 
    
    Exampls
    -------
    variable_summary_bytype(inDf = model_table, 
                            varLst = [x for x in model_table.columns.tolist() if x not in ['request_id']], 
                            varType = 'Order')
    '''
    
    if varType == 'Continuous':
        
        loci = 0
        tmpDfCon = pd.DataFrame(columns=('VarName', 'N', 'NMiss', 'Min', 'Mean', 'Median', 'Max', 
                                      'PTP' , 'Variance', 'Std', 'CV', 'Skew', 'Kurtosis',
                                      'P0', 'P5', 'P10', 'P20', 'P30', 'P40', 'P50', 'P60', 'P70', 'P80', 'P90', 'P95', 'P100'))
        for varItem in varLst:
            N = len(inDf[varItem])
            NMiss = inDf[varItem].isna().sum()
            Min = inDf[varItem].min()
            Mean = round(inDf[varItem].mean(),2)
            Median = inDf[varItem].median()
            Max = inDf[varItem].max()
            Ptp = round(inDf[varItem].ptp(),2)
            Variance = round(inDf[varItem].var(),2)
            Std = round(inDf[varItem].std(),2)
            if Mean==0:
                CV = Std
            else:
                CV = round(Std/Mean,2)
            Skew = round(inDf[varItem].skew(),2)
            Kurtosis = round(inDf[varItem].kurtosis(),2)
            P0 = np.quantile(inDf[varItem].dropna(),0)
            P5 = np.quantile(inDf[varItem].dropna(),0.05)
            P10 = np.quantile(inDf[varItem].dropna(),0.1)
            P20 = np.quantile(inDf[varItem].dropna(),0.2)
            P30 = np.quantile(inDf[varItem].dropna(),0.3)
            P40 = np.quantile(inDf[varItem].dropna(),0.4)
            P50 = np.quantile(inDf[varItem].dropna(),0.5)
            P60 = np.quantile(inDf[varItem].dropna(),0.6)
            P70 = np.quantile(inDf[varItem].dropna(),0.7)
            P80 = np.quantile(inDf[varItem].dropna(),0.8)
            P90 = np.quantile(inDf[varItem].dropna(),0.9)
            P95 = np.quantile(inDf[varItem].dropna(),0.95)
            P100 = np.quantile(inDf[varItem].dropna(),1)
            
            tmpDfCon.loc[loci] = [varItem, N, NMiss, Min, Mean, Median, Max, Ptp, Variance, Std, CV, Skew, Kurtosis, 
                             P0, P5, P10, P20, P30, P40, P50, P60, P70, P80, P90, P95, P100]
            loci = loci + 1
        return tmpDfCon
    
    else :
        tmpDfClass = pd.DataFrame(columns=('VarName', 'VarValue', 'ValueFreq', 'ValueRate'))    
        for varItem in varLst:
             tmpDf = var_freq_dist(inDf[varItem]).reset_index()
             tmpDf['VarName'] = varItem
             tmpDf = tmpDf.rename(columns={'index':'VarValue', 'Freq':'ValueFreq', 'Rate':'ValueRate'})
             tmpDfClass = pd.concat([tmpDfClass, tmpDf], axis=0)
             tmpDfClass = tmpDfClass[['VarName','VarValue','ValueFreq','ValueRate']]                 
        return tmpDfClass
                 
    



