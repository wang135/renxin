# -*- coding: utf-8 -*-

import pandas as pd
from base.freq_stats import cross_table
from scipy.stats import chi2_contingency as chisq


def binary_df_stat(inDf, varList, varY):
    '''
    Funcation Descriptions:
        计算每个二值变量的chisq值和频数分布、逾期率
    
    Parameters
    ----------
    inDf    : 数据框
    varList : 二值变量列表
    varY    : 目标变量
    
    Returns
    -------
    列表： vars_freq_df-变量频数及逾期率数据框  vars_chisq_df-变量的卡方值数据框
    
    Examples
    --------
    inDf = ins_clean_df
    varList = ins_var_class_df[ins_var_class_df['Dclass']=='Binary']['index'].tolist()
    varY = 'TargetBad'
    binary_df_stat(inDf, varList, varY)
    '''    
    vars_freq_df = pd.DataFrame(columns=['VarName','Levels', 0, 1, 'MissCnt', 'Total'])
    vars_chisq_df = pd.DataFrame(columns=['VarName','Sample_Size','Chisq_Stat', 'Chisq_P', 'Chisq_Df', 'Chisq_DfStat'])
    for var_item in varList:
        var_stat = cross_table(inDf, var_item, varY).rename(columns={var_item : 'Levels'})
        var_stat['VarName'] = var_item
        var_chisq = chisq(var_stat[var_stat['Levels']!='Total'][[0,1]].as_matrix())
        var_chisq_df = pd.DataFrame([[var_stat[var_stat['Levels']!='Total'][[0,1]].values.sum(),
                                      round(var_chisq[0],2), round(var_chisq[1],4),
                                      var_chisq[2], round(var_chisq[0]/var_chisq[2],2)]],
                                      columns=['Sample_Size','Chisq_Stat','Chisq_P','Chisq_Df','Chisq_DfStat']
                                   )
        var_chisq_df['VarName'] = var_item
        vars_freq_df = pd.concat([vars_freq_df, var_stat], axis=0, ignore_index=True)
        vars_chisq_df = pd.concat([vars_chisq_df, var_chisq_df], axis=0, ignore_index=True)

    return {'vars_freq_df' : vars_freq_df, 
            'vars_chisq_df': vars_chisq_df}



