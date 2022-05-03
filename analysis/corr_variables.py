# -*- coding: utf-8 -*-

from pandas import DataFrame
import pandas as pd
from scipy.stats import chi2_contingency as chisq


def corr_continuous_variables(inDf, varList,  method='spearmanr'):
    '''
    Funcation Descriptions：
        计算表中全部连续变量两两间的相关系数
    
    Parameters
    ----------
    inDf      : 数据框
    varList   : 变量列表
    method    : 相关系数方法，包括：pearsonr\spearmanr\kendalltau
    
    Returns
    -------
    
    '''
    from scipy.stats import pearsonr, spearmanr, kendalltau
       
    CorrDf = DataFrame(columns=('varname1', 'varname2', 'corr_coef', 'corr_p', 'records'))
    KeyValue = -1
    if method == 'pearsonr':
        for i in range(len(varList)):
            for j in range(i+1,len(varList),1):
                KeyValue = KeyValue+1
                TmpDf = inDf[[varList[i],varList[j]]]
                TmpDf = TmpDf[~TmpDf.isnull().any(axis=1)]
                records = TmpDf.shape[0]
                correlation, pvalue = pearsonr(TmpDf[varList[i]],TmpDf[varList[j]])
                CorrDf.loc[KeyValue] = [varList[i], varList[j], correlation, round(pvalue,4), records]
    elif method == 'spearmanr':
        for i in range(len(varList)):
            for j in range(i+1,len(varList),1):
                KeyValue = KeyValue+1
                TmpDf = inDf[[varList[i],varList[j]]]
                TmpDf = TmpDf[~TmpDf.isnull().any(axis=1)]
                records = TmpDf.shape[0]
                correlation, pvalue = spearmanr(TmpDf[varList[i]],TmpDf[varList[j]])
                CorrDf.loc[KeyValue] = [varList[i], varList[j], correlation, round(pvalue,4), records]
    elif method == 'kendalltau':
        for i in range(len(varList)):
            for j in range(i+1,len(varList),1):
                KeyValue = KeyValue+1
                TmpDf = inDf[[varList[i],varList[j]]]
                TmpDf = TmpDf[~TmpDf.isnull().any(axis=1)]
                records = TmpDf.shape[0]
                correlation, pvalue = kendalltau(TmpDf[varList[i]],TmpDf[varList[j]])
                CorrDf.loc[KeyValue] = [varList[i], varList[j], correlation, round(pvalue,4), records]

    return CorrDf





def corr_class_variables(inDf, varList):
    '''
    Funcation Descriptions：
        计算表中全部类别变量两两间的相关系数
    '''
    TmpInDf = inDf[varList].astype(str)  ## 所有分类变量转换为字符型，目的是能够对缺失值进行统计

    #循环计算非连续性变量的关联性
    ChisqResultDf= DataFrame(columns=('varname1','varname2','Chisq_Stat','Chisq_P',
                                       'Chisq_df','Chisq_DfStat','Chisq_table'))
    KeyValue = -1
    for i in range(len(varList)):
        for j in range(i+1,len(varList),1):
            KeyValue = KeyValue+1
            TmpTable = pd.crosstab(TmpInDf[varList[i]], TmpInDf[varList[j]])
            #ChisqResult = chisq(TmpTable.as_matrix())
            
            ChisqResult = chisq(TmpTable.values)
            ChisqResultDf.loc[KeyValue]=[varList[i],varList[j],round(ChisqResult[0],2),
                                          round(ChisqResult[1],4),ChisqResult[2],
                                          round(ChisqResult[0]/ChisqResult[2],2),ChisqResult[3]]
    
    return ChisqResultDf



def corr_df_cal(inDf, varTypeDf):
    '''
    
    inDf = ins_clean_df
    varTypeDf = ins_clean_class_df
    corr_df_cal(inDf, varTypeDf)
    '''
    to_con_ls = varTypeDf[varTypeDf['Dclass']=='Continuous']['index'].tolist()
    corr_continue_cor_df = corr_continuous_variables(
                                             inDf = inDf,
                                             varList = to_con_ls)
    
    to_class_ls = varTypeDf[varTypeDf['Dclass'].isin(['Binary','Order','Nominal'])]['index'].tolist()
    corr_class_chisq_df = corr_class_variables(
                                             inDf = inDf,
                                             varList = to_class_ls)
    return {'continue_cor_df': corr_continue_cor_df,
            'class_chisq_df': corr_class_chisq_df}



def corr_static_select(inCorrDf, statVar, dropSign, statPoint):
    '''
    Function Descriptions:
        根据相关性统计量逐步剔除相关性强的变量，最终保留相关性弱的变量
    
    Parameters
    ----------
    inCorrDf    : 变量相关性结果数据框
    selectVar   : 进行相关性选择的统计量名称
    selectPoint : 相关性选择的标准值，以大于该标准值认为相关性强
        
    Returns
    -------
    相关性强的变量列表
    
    Examples
    --------
    inCorrDf = corr_df
    statVar = 'Chisq_DfStat'
    dropSign = '>'
    statPoint = 
    ''' 
    keep_var = list()    
    tmp_corr_df = inCorrDf.copy().reset_index(drop=True)
    loops = 1
    while loops>0:
        tmp_ls = tmp_corr_df['varname1'].unique().tolist()
        keep_var.append(tmp_ls[0])
        if dropSign == '<=':
            be_ls = tmp_corr_df[(tmp_corr_df['varname1']==tmp_ls[0]) & (abs(tmp_corr_df[statVar]) <= statPoint)
                               ]['varname2'].unique().tolist()
        elif dropSign == '<':
            be_ls = tmp_corr_df[(tmp_corr_df['varname1']==tmp_ls[0]) & (abs(tmp_corr_df[statVar]) < statPoint)
                               ]['varname2'].unique().tolist()
        elif dropSign == '>=':
            be_ls = tmp_corr_df[(tmp_corr_df['varname1']==tmp_ls[0]) & (abs(tmp_corr_df[statVar]) >= statPoint)
                               ]['varname2'].unique().tolist()
        elif dropSign == '>':
            be_ls = tmp_corr_df[(tmp_corr_df['varname1']==tmp_ls[0]) & (abs(tmp_corr_df[statVar]) > statPoint)
                               ]['varname2'].unique().tolist()
        tmp_corr_df = tmp_corr_df[~tmp_corr_df['varname1'].isin(be_ls+[tmp_ls[0]])]
        if tmp_corr_df['varname1'].nunique() <= 1:
            loops=0
            if tmp_corr_df['varname1'].nunique() == 1:
                keep_var.append(tmp_corr_df['varname1'].unique()[0])
                last_df=inCorrDf[inCorrDf['varname2']==tmp_corr_df['varname2'].tolist()[0]]
                if last_df[last_df[statVar]>statPoint].shape[0]==0:
                    keep_var.append(tmp_corr_df['varname2'].unique()[0])
    drop_var_ls = inCorrDf[~inCorrDf['varname1'].isin(keep_var)]['varname1'].unique().tolist()            
    return drop_var_ls



def corr_p_select(inCorrDf, selectVar, selectPoint):
    '''
    Function Descriptions:
        根据相关性检验P值逐步剔除相关性强的变量，最终保留相关性强的变量
    
    Parameters
    ----------
    inCorrDf    : 变量相关性结果数据框
    selectVar   : 进行相关性选择的p值变量名称
    selectPoint : 相关性选择的标准值，以小于该标准值认为相关性强
        
    Returns
    -------
    相关性强的变量列表
    ''' 
    keep_var = list()    
    tmp_corr_df = inCorrDf.copy()
    loops = 1
    while loops>0:
        tmp_ls = tmp_corr_df['varname1'].unique().tolist()
        keep_var.append(tmp_ls[0])
        be_ls = tmp_corr_df[(tmp_corr_df['varname1']==tmp_ls[0]) & (abs(tmp_corr_df[selectVar])<selectPoint)
                           ]['varname2'].unique().tolist()
        tmp_corr_df = tmp_corr_df[~tmp_corr_df['varname1'].isin(be_ls+[tmp_ls[0]])]
        if tmp_corr_df['varname1'].nunique() <= 1:
            loops=0
            keep_var.append(tmp_corr_df['varname1'].unique()[0])
            last_df=inCorrDf[inCorrDf['varname2']==tmp_corr_df['varname2'].tolist()[0]]
            if last_df[last_df[selectVar]<selectPoint].shape[0]==0:
                keep_var.append(tmp_corr_df['varname2'].unique()[0])
    drop_var_ls = inCorrDf[~inCorrDf['varname1'].isin(keep_var)]['varname1'].unique().tolist() 
                
    return drop_var_ls




'''
def corr_continuous_variables(inDf, inVarClassDf,  method='spearman'):

    TmpConVarList = inVarClassDf[inVarClassDf['Dclass'] == 'Continuous']['index'].tolist()     
    
    TmpConCorrDf0 = inDf[TmpConVarList].corr(method=method)
    TmpConCorrDf1 = DataFrame(np.tril(TmpConCorrDf0.as_matrix(), 0), 
                              index=TmpConVarList, 
                              columns=TmpConVarList)
    TmpConCorrDf2 = TmpConCorrDf1.unstack().reset_index().rename(columns={0:'CorValue'})
    ConCorrDf = TmpConCorrDf2[TmpConCorrDf2['level_0'] != TmpConCorrDf2['level_1']]
    ConCorrDf = ConCorrDf[ConCorrDf['CorValue'] != 0]
 
    return ConCorrDf
'''


