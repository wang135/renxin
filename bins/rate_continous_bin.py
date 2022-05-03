# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas import DataFrame

from base import equal_freq_cut_map
from scipy.stats import chi2_contingency as chisq
from intervals import FloatInterval


##单个连续变量的等频分箱，逐步分箱合并；
def continuous_equal_rate_bin(inDf, xVarName, yVarName, n=20):
    '''
    Functions Descriptions:
        对连续变量进行等频分箱后，按照逾期率相近度进行逐步合并
    
    Paramters
    ---------
    inDf     : 数据框
    xVarName : 自变量名称
    yVarName : 因变量名称
    n        : 分箱数量
    
    Returns
    -------
    列表：qcut_step_bin_df-等频分箱及每次分箱合并的统计结果  qcut_step_chisq_df-等频分箱及每次合并分箱的chisq值
    
    Examples
    --------
    inDf = ins_clean_df 
    xVarName = 'cur2year_total_m1_cnt'
    yVarName = 'TargetBad'
    n = 10
    ContinuousBestBin(inDf, xVarName, yVarName, n=10)
    '''
    #连续变量缺失值的is_bad统计
    null_1 = inDf[inDf[xVarName].isnull()][yVarName].sum()
    null_all = inDf[inDf[xVarName].isnull()][yVarName].count()
    null_rate = round(null_1/null_all,6)
    
    #等频分箱,生成数据框
    qcut_ser = equal_freq_cut_map(inDf[xVarName], n)['x_bin_ser']
    #qcut_freq = qcut_freq.reset_index()
    
    #生成分箱统计量
    qcut_df = DataFrame(qcut_ser.value_counts().sort_index()).rename(columns={'bin_{}'.format(xVarName): 'All'})
    qcut_group = inDf[yVarName].groupby(qcut_ser)
    qcut_df[1] = qcut_group.sum()
    qcut_df['Rate'] = qcut_group.mean()
    qcut_df[0] = qcut_df['All']-qcut_df[1]
    qcut_df = qcut_df.reset_index(drop=False)
    qcut_df['Levels'] = qcut_df.index + 1
    qcut_df = qcut_df.rename(columns={'index':'Bins'})
    
    # 缺失值统计量
    if null_all>0:
        miss_df = DataFrame([[len(qcut_df)+1, 'nan', null_all, null_1, null_rate, null_all-null_1]],
                              columns = ['Levels', 'Bins', 'All', 1, 'Rate', 0])
        QcutStepsDat = qcut_df.copy()
        QcutStepsDat = pd.concat([QcutStepsDat, miss_df], axis=0, ignore_index=True)
    else :
        QcutStepsDat = qcut_df.copy()
    QcutStepsDat['Steps'] = 0
    
    #Chisq统计量计算    
    ChisqResult = chisq(QcutStepsDat[[0,1]].values)
    QcutChisqDat =pd.DataFrame([[0,QcutStepsDat[[0,1]].values.sum(), round(ChisqResult[0],2),round(ChisqResult[1],4),
                                 ChisqResult[2], round(ChisqResult[0]/(ChisqResult[2]+0.0001),2)]],
                                 columns=['Steps','Sample_Size','Chisq_Stat','Chisq_P','Chisq_Df','Chisq_DfStat']
                              )
    
    #分箱逐步合并计算
    if len(qcut_df) > 2:
        for steps in range(1,len(qcut_df)-1):
            qcut_diff_df = DataFrame(columns=['PreIndex','AftIndex','PreBin','AftBin','Diff'])
            #区间差计算
            for i in range(len(qcut_df)-1):
                Diff = np.abs(qcut_df['Rate'][i]-qcut_df['Rate'][i+1])
                TmpDiff=DataFrame([[qcut_df['Levels'][i],qcut_df['Levels'][i+1],
                                    qcut_df['Bins'][i],qcut_df['Bins'][i+1],Diff]],
                                  columns=['PreIndex','AftIndex','PreBin','AftBin','Diff'])
                qcut_diff_df = pd.concat([qcut_diff_df,TmpDiff]).reset_index(drop=True)
            #选择差最小的两个Bin    
            min_diff_ser = qcut_diff_df.sort_values(by='Diff').iloc[0]
            min_diff_df = qcut_df[qcut_df['Levels'].isin([min_diff_ser['PreIndex'], min_diff_ser['AftIndex']])]
            qcut_df = qcut_df[~qcut_df['Levels'].isin([min_diff_ser['PreIndex'],min_diff_ser['AftIndex']])] 
            qcut_df = qcut_df.reset_index(drop=True)
            #合并成新的Bin
            min_comb_ser = min_diff_df.sum()
            comb_rate = min_comb_ser[1] / min_comb_ser['All']
            comb_bin = FloatInterval.open_closed(min_diff_ser['PreBin'].lower,min_diff_ser['AftBin'].upper)
            qcut_df.loc[max(qcut_df.index)+1] = [comb_bin, min_comb_ser['All'], min_comb_ser[1], 
                                         comb_rate, min_comb_ser[0], min_diff_ser['PreIndex']]
            qcut_df = qcut_df.sort_values(by='Levels').reset_index(drop=True)
            qcut_df['Levels'] = qcut_df.index+1
            
            if null_all > 0:    
                one_step_df = pd.concat([qcut_df, miss_df], axis=0, ignore_index=True)
                one_step_df['Levels'] = one_step_df.index+1
            else :
                one_step_df = qcut_df.copy()
                
            one_step_df['Steps'] = steps        
            QcutStepsDat = pd.concat([QcutStepsDat,one_step_df]).reset_index(drop=True)
        
            #Chisq统计量计算    
            ChisqResult = chisq(one_step_df[[0,1]].values)
            one_chisq_df = pd.DataFrame([[steps,one_step_df[[0,1]].values.sum(),round(ChisqResult[0],2),round(ChisqResult[1],4),
                                          ChisqResult[2], round(ChisqResult[0]/ChisqResult[2],2)]],
                                          columns=['Steps','Sample_Size','Chisq_Stat','Chisq_P','Chisq_Df','Chisq_DfStat']
                                       )
        
            QcutChisqDat=pd.concat([QcutChisqDat, one_chisq_df]).reset_index(drop=True)
    else:
        one_step_df = QcutStepsDat.copy()
        steps = 0
    
    
    if null_all>0:
        null_rate = one_step_df[one_step_df['Bins'] == 'nan']['Rate'].tolist()[0]
        null_df = qcut_df.copy()
        null_df['diff'] = np.abs(null_df['Rate']-null_rate)
        null_df = null_df.sort_values(by='diff')
        null_min_ser = null_df.iloc[0]
        
        null_comb_df = one_step_df[one_step_df['Bins'].isin([null_min_ser['Bins'], 'nan'])]
        null_comb_ser = null_comb_df.sum()
        
        qcut_df = qcut_df[~qcut_df['Bins'].isin([null_min_ser['Bins'], 'nan'])]
        qcut_df.loc[max(qcut_df.index)+1] = [[null_min_ser['Bins'], 'nan'], null_comb_ser['All'], null_comb_ser[1],
                                    round(null_comb_ser[1]/null_comb_ser['All'],6), null_comb_ser[0], null_min_ser['Levels']]
        qcut_df = qcut_df.sort_values(by='Levels').reset_index(drop=True)
        qcut_df['Levels'] = qcut_df.index+1
        
        one_step_df = qcut_df.copy()
        one_step_df['Steps'] = steps+1        
        QcutStepsDat = pd.concat([QcutStepsDat,one_step_df]).reset_index(drop=True)
    
        #Chisq统计量计算    
        ChisqResult = chisq(one_step_df[[0,1]].values)
        one_chisq_df = pd.DataFrame([[steps+1,one_step_df[[0,1]].values.sum(),round(ChisqResult[0],2),round(ChisqResult[1],4),
                                      ChisqResult[2], round(ChisqResult[0]/ChisqResult[2],2)]],
                                      columns=['Steps','Sample_Size','Chisq_Stat','Chisq_P','Chisq_Df','Chisq_DfStat']
                                   )
    
        QcutChisqDat=pd.concat([QcutChisqDat, one_chisq_df]).reset_index(drop=True)        
        QcutStepsDat = QcutStepsDat[['Steps', 'Levels', 'Bins', 'All', 1, 0, 'Rate']]

    return {'qcut_step_bin_df': QcutStepsDat,
            'qcut_step_chisq_df': QcutChisqDat }
    
    

#多个变量分箱统计
def continuous_df_rate_bin(inDf, varList, yVarName, n=20):
    '''
    Funcations Descriptions:
        对于数据框中指定的变量进行等频分箱后，逐步按照逾期率的相近度进行分箱合并。
    
    Paramters
    ---------
    inDf     : 样本数据框
    varList  : 待分箱变量列表
    yVarName : 目标变量
    n        : 初始分箱数量
    
    Returns
    -------
    数据字典： rate_step_bin_df-等频分箱及每次分箱合并的结果  rate_step_chisq_df-等频分箱及每次分箱合并的chisq值
    
    Examples
    --------
    inDf = ins_clean_df 
    varList = ins_ived_class_df[ins_ived_class_df['Dclass']=='Continuous']['index'].tolist()
    yVarName = 'TargetBad'
    continuous_df_rate_bin(inDf, varList, yVarName)
    '''
    #生成结果表结构
    BinChisqStepDat=DataFrame(columns=['Steps','Chisq_Stat','Chisq_P','Chisq_Df',
                                       'Chisq_DfStat','VarName'])
    BinStepDat=DataFrame(columns=['Steps','Levels','Bins','All',1,0,'Rate','VarName'])
    #对每个变量进行等频分箱，逐步Bin合并和计算chisq
    for TmpVar in varList:
        print('Continuous rate bin combine: ', TmpVar)
        TmpOneBest = continuous_equal_rate_bin(inDf = inDf, xVarName = TmpVar, yVarName = yVarName, n=n)
        TmpOneStepsDat = TmpOneBest['qcut_step_bin_df']
        TmpOneChisqDat = TmpOneBest['qcut_step_chisq_df']
        TmpOneStepsDat['VarName'] = TmpVar
        TmpOneChisqDat['VarName'] = TmpVar
        BinChisqStepDat=pd.concat([BinChisqStepDat,TmpOneChisqDat])
        BinStepDat=pd.concat([BinStepDat,TmpOneStepsDat])
        
    BinChisqStepDat['VarName'] = BinChisqStepDat['VarName'].map(lambda x: 'bin_%s' % x)
    BinStepDat['VarName'] = BinStepDat['VarName'].map(lambda x: 'bin_%s' % x)
    
    print("***连续变量等频分箱及逾期率分箱合并完成！")
    return {'rate_step_chisq_df': BinChisqStepDat,
            'rate_step_bin_df': BinStepDat
            }  






