# -*- coding: utf-8 -*-

import pandas as pd
from base.freq_stats import var_freq_dist
from base.freq_stats import cross_table
from scipy.stats import chi2_contingency as chisq


#连续变量单调性判断
def monotonicity_cal(inFreqDf):
    '''
    Funcation Descriptions:
        变量预测力的单调性计算
        
    Parameters
    ----------
    inFreqDf  : 具有逾期率统计的变量数据框
    
    Returns
    -------
    变量单调性判断结果数据框
    
    Examples
    --------
    inFreqDf = rate_con_cmb_freq_df
    monotonicity_cal(inFreqDf)
    '''
    inFreqDf = inFreqDf[~((inFreqDf['Bins'] == 'nan') | (inFreqDf['Bins'] == 'Missing') | (inFreqDf['Bins'].isnull()))]
    #inStepFreqDf = inStepFreqDf[~inStepFreqDf['Bins'].isnull()]
    var_ls = inFreqDf['VarName'].unique().tolist()
    ConMonotonicityDat=pd.DataFrame(columns=['VarName','Steps','BinCnt','PosCnt','NegCnt','MonotonicityFlag'])
    for var_item in var_ls:
        #var_item = 'bin_cur_query_days'
        n_step = inFreqDf[inFreqDf['VarName']==var_item]['Steps'].nunique()
        for Steps in range(n_step):  
            #Steps = 0
            one_step_df = inFreqDf[(inFreqDf['VarName']==var_item) & (inFreqDf['Steps']==Steps)]
            one_step_df = one_step_df.sort_values(by='Levels').reset_index(drop=True)
            
            bin_cnt = one_step_df.shape[0]-1    
            PosCnt=0
            NegCnt=0
            #单调性判断
            for i in range(bin_cnt):
                if one_step_df['Rate'][i]-one_step_df['Rate'][i+1]>0 :
                    PosCnt = PosCnt + 1
                elif one_step_df['Rate'][i]-one_step_df['Rate'][i+1]<0 :
                    NegCnt = NegCnt + 1
            if (NegCnt==bin_cnt) | (PosCnt==bin_cnt):
                MonotonicityFlag = 1
            else :
                MonotonicityFlag = 0
            TmpMonoDat=pd.DataFrame([[var_item,Steps,bin_cnt,PosCnt,NegCnt,MonotonicityFlag]],
                                    columns=['VarName','Steps','BinCnt','PosCnt','NegCnt','MonotonicityFlag'])
            ConMonotonicityDat = pd.concat([ConMonotonicityDat,TmpMonoDat]).reset_index(drop=True)
    return ConMonotonicityDat



def binary_freq_select(inDf, varList, cutOff=0.05):
    '''
    Funcation Descriptions:
        筛选样本分类无统计意义的变量
    
    Parameters
    ----------
    inDf    : 数据框
    varList : 二值变量列表
    cutOff  : 无统计意义的样本占比标准值，范围[0，1]
    
    Returns
    -------
    满足样本占比的变量列表
    
    Examples
    --------
    inDf = ins_clean_df
    varList = inVarClassDf[inVarClassDf['Dclass']=='Binary']['index'].tolist()
    binary_freq_select(inDf, inVarClassDf)
    '''
    # 筛选样本占比满足比例的变量
    out_var_ls = []
    for xVarName in varList:
        #xVarName = 'have_spouse'
        freq_df = var_freq_dist(x=inDf[xVarName], pctFormat=False)
        if freq_df['Rate'].min() > cutOff:
            out_var_ls.append(xVarName)
            
    return out_var_ls



def binary_chisq(inDf, varList, varY):
    '''
    Funcation Descriptions:
        计算每个变量的chisq值
    
    Parameters
    ----------
    inDf    : 数据框
    varList : 二值变量列表
    varY    : 目标变量
    
    Returns
    -------
    满足样本占比的变量列表
    
    Examples
    --------
    inDf = ins_clean_df
    varList = binary_freq_select(inDf, inVarClassDf)
    varY = 'TargetBad'
    '''    
    vars_freq_df = pd.DataFrame(columns=['VarName','Levels', 0, 1, 'MissCnt', 'Total'])
    vars_chisq_df = pd.DataFrame(columns=['VarName','Chisq_Stat', 'Chisq_P', 'Chisq_Df', 'Chisq_DfStat'])
    for var_item in varList:
        var_stat = cross_table(inDf, var_item, varY).rename(columns={var_item : 'Levels'})
        var_stat['VarName'] = var_item
        var_chisq = chisq(var_stat[var_stat['Levels']!='Total'][[0,1]].as_matrix())
        var_chisq_df = pd.DataFrame([[round(var_chisq[0],2), round(var_chisq[1],4),
                                     var_chisq[2], round(var_chisq[0]/var_chisq[2],2)]],
                                     columns=['Chisq_Stat','Chisq_P','Chisq_Df','Chisq_DfStat']
                                   )
        var_chisq_df['VarName'] = var_item
        vars_freq_df = pd.concat([vars_freq_df, var_stat], axis=0, ignore_index=True)
        vars_chisq_df = pd.concat([vars_chisq_df, var_chisq_df], axis=0, ignore_index=True)

    return {'vars_freq_df' : vars_freq_df, 
            'vars_chisq_df': vars_chisq_df}


def binary_power_select(inDf, varList, varY, freqCutOff=0.05, chisqCutOff=3.5):
    '''
    Funcation Descriptions:
        预测能力强及有统计意义的变量选择
    
    Parameters
    ----------
    inDf         : 数据框
    varList      : 二值变量列表
    varY         : 目标变量
    freqCutOff   : 无统计代表性的样本占比标准
    chisqCutOff  : chisq预测能力强的标准
    
    Returns
    -------
    预测能力强及有统计意义的变量的变量列表
    
    Examples
    --------
    inDf = ins_clean_df
    varList = inVarClassDf[inVarClassDf['Dclass']=='Binary']['index'].tolist()
    varY = 'TargetBad'  
    binary_power_select(inDf, varList, varY, freqCutOff=0.05, chisqCutOff=3.5)
    '''
    
    var_ls = binary_freq_select(inDf, varList=varList, cutOff = freqCutOff)
    binary_power_rst = binary_chisq(inDf, varList=var_ls, varY=varY)
    binary_chisq_df = binary_power_rst['vars_chisq_df']

    keep_lst = list()
    for var_item in var_ls:
        if binary_chisq_df[binary_chisq_df['VarName']==var_item]['Chisq_Stat'].tolist()[0] > chisqCutOff:
            keep_lst.append(var_item)
            
    return keep_lst        



def nominal_power_select(inChisqDf, pCutOff = 0.1, decileCutOff = 0.25):
    '''
    Function Descriptions:
        名义变量及其最优分箱选择：先选择最大chisq值前75%的变量，再选择每个变量前三个最优的分箱
        
    Paramters
    ---------
    inChisqDf    : 名义变量卡方统计量数据框
    pCutOff      : chisq检验p值显著性分割点，范围：[0,1]
    decileCutOff : 选取变量的分位数，范围：[0,1].即选择大于该分位数的全部变量。
    
    Returns
    -------
    入选变量及最优分箱的chisq统计量数据框
    
    Examples
    --------
    inChisqDf = bin_step_power_df[bin_step_power_df['Dclass']=='Nominal']
    pCutOff = 0.1
    decileCutOff = 0.25
    NominalBestBinSelect(inChisqDf, pCutOff = 0.1, decileCutOff = 0.25)
    '''
    #计算名义变量下25%分位数
    var_max_chisq = inChisqDf.groupby('VarName')['Chisq_DfStat'].max()
    decile_cutoff = var_max_chisq.quantile(decileCutOff)
    #选择卡方检验P值小于pCutOff或者大于下25%分位数的变量
    select_var_lst = inChisqDf[((inChisqDf['Chisq_P'] <= pCutOff) & (inChisqDf['Chisq_DfStat'] >= decile_cutoff))
                              ]['VarName'].unique().tolist()
    select_var_df = pd.DataFrame(columns=['NewVarName', 'VarName', 'Steps', 'Sample_Size', 'Chisq_Stat',
                                          'Chisq_Df', 'Chisq_P', 'Chisq_DfStat', 'IV', 'Dclass'])
    for var_item in select_var_lst:
        select_var_chisq = inChisqDf[(inChisqDf['VarName'] == var_item) & (inChisqDf['Chisq_P'] <= pCutOff)]
        select_var_chisq = select_var_chisq.sort_values(by='Chisq_DfStat',ascending=False).reset_index(drop=True)
        #选择每个变量的前2种或者3种分箱
        select_var_df = pd.concat([select_var_df,select_var_chisq.head(3)]).reset_index(drop=True)
        '''
        if select_var_chisq.shape[0] > 2:
            NomModelVarDat = pd.concat([NomModelVarDat,select_var_chisq.iloc[[0,1,2]]]).reset_index(drop=True)
        elif select_var_chisq.index.size == 2:
            NomModelVarDat = pd.concat([NomModelVarDat,select_var_chisq.iloc[[0,1]]]).reset_index(drop=True)
        elif select_var_chisq.index.size == 1:
            NomModelVarDat = pd.concat([NomModelVarDat,select_var_chisq.iloc[[0]]]).reset_index(drop=True) 
        '''
    
    return select_var_df




def continuous_power_select(inStepFreqDf, inChisqDf, pCutOff = 0.1, decileCutOff = 0.25):
    '''
    Function Descriptions:
        连续变量及其最优分箱选择：先选择最大chisq值前75%的变量，再选择每个变量前三个最优的分箱
        
    Paramters
    ---------
    inStepFreqDf  : 连续变量频数及逾期率数据框
    inChisqDf     : 连续变量卡方统计量数据框   
    pCutOff       : chisq检验p值显著性分割点，范围：[0,1]
    decileCutOff  : 选取变量的分位数，范围：[0,1].即选择大于该分位数的全部变量。
    
    Returns
    -------
    入选变量及最优分箱的chisq统计量数据框
    
    Examples
    --------
    inStepFreqDf = rate_con_cmb_freq_df
    inChisqDf = rate_con_cmb_chisq_df
    pCutOff = 0.1
    decileCutOff = 0.25
    continuous_power_select(inStepFreqDf, inChisqDf, pCutOff = 0.1, decileCutOff = 0.25)
    '''
    
    #选择单调的变量
    monot_df = monotonicity_cal(inStepFreqDf)
    monot_df = monot_df[monot_df['MonotonicityFlag']==1]
    monot_chisq_df = pd.merge(monot_df[['VarName','Steps']], inChisqDf, on=['VarName','Steps'], how='left')

    #计算连续变量下25%分位数
    var_max_chisq = monot_chisq_df.groupby('VarName')['Chisq_DfStat'].max()
    decile_cutoff = var_max_chisq.quantile(decileCutOff)
    #选择卡方均值大于5或者大于下25%分位数的变量
    select_var_lst = monot_chisq_df[(monot_chisq_df['Chisq_P'] <= pCutOff) & (monot_chisq_df['Chisq_DfStat'] >= decile_cutoff)
                                   ]['VarName'].unique().tolist()
    select_var_df = pd.DataFrame(columns=['NewVarName', 'VarName', 'Steps', 'Sample_Size', 'Chisq_Stat',
                                          'Chisq_Df', 'Chisq_P', 'Chisq_DfStat', 'IV', 'Dclass'])
    for var_item in select_var_lst:
        select_var_chisq = monot_chisq_df[(monot_chisq_df['VarName'] == var_item) & (monot_chisq_df['Chisq_P'] <= pCutOff)]
        select_var_chisq = select_var_chisq.sort_values(by='Chisq_DfStat',ascending=False).reset_index(drop=True)
        select_var_df = pd.concat([select_var_df,select_var_chisq.head(3)]).reset_index(drop=True)
    
    return select_var_df




def order_power_select(inStepFreqDf, inChisqDf, pCutOff = 0.1, decileCutOff = 0.25):
    '''
    Function Descriptions:
        有序变量及其最优分箱选择：先选择最大chisq值前75%的变量，再选择每个变量前三个最优的分箱
        
    Paramters
    ---------
    inStepFreqDf  : 连续变量频数及逾期率数据框
    inChisqDf     : 连续变量卡方统计量数据框   
    pCutOff       : chisq检验p值显著性分割点，范围：[0,1]
    decileCutOff  : 选取变量的分位数，范围：[0,1].即选择大于该分位数的全部变量。
    
    Returns
    -------
    入选变量及最优分箱的chisq统计量数据框
    
    Examples
    --------
    inStepFreqDf = bin_step_dist_df[bin_step_dist_df['Dclass']=='Order']
    inChisqDf = bin_step_power_df[bin_step_power_df['Dclass']=='Order']
    pCutOff = 0.1
    decileCutOff = 0.25
    order_power_select(inStepFreqDf, inChisqDf, pCutOff = 0.1, decileCutOff = 0.25)
    '''
    
    #选择单调的变量
    monot_df = monotonicity_cal(inStepFreqDf)
    monot_df = monot_df[monot_df['MonotonicityFlag']==1]
    monot_chisq_df = pd.merge(monot_df[['VarName','Steps']], inChisqDf, on=['VarName','Steps'], how='left')

    #计算连续变量下25%分位数
    var_max_chisq = monot_chisq_df.groupby('VarName')['Chisq_DfStat'].max()
    decile_cutoff = var_max_chisq.quantile(decileCutOff)
    #选择卡方均值大于5或者大于下25%分位数的变量
    select_var_lst = monot_chisq_df[(monot_chisq_df['Chisq_P'] <= pCutOff) & (monot_chisq_df['Chisq_DfStat'] >= decile_cutoff)
                                   ]['VarName'].unique().tolist()
    select_var_df = pd.DataFrame(columns=['NewVarName', 'VarName', 'Steps', 'Sample_Size', 'Chisq_Stat',
                                          'Chisq_Df', 'Chisq_P', 'Chisq_DfStat', 'IV', 'Dclass'])
    for var_item in select_var_lst:
        select_var_chisq = monot_chisq_df[(monot_chisq_df['VarName'] == var_item) & (monot_chisq_df['Chisq_P'] <= pCutOff)]
        select_var_chisq = select_var_chisq.sort_values(by='Chisq_DfStat',ascending=False).reset_index(drop=True)
        select_var_df = pd.concat([select_var_df,select_var_chisq.head(3)]).reset_index(drop=True)
    
    return select_var_df












       
