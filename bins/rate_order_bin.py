# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.stats import chi2_contingency as chisq

#有序变量最优分箱
def order_rate_combine(inDf, inBinDat, xVarName, yVarName):
    '''
    Function Descriptions:
        按照分箱的逾期率，对连续变量进行逐步分箱合并
        
    Parameters
    ----------
    inDf     : 用于逾期率合并的数据框
    inBinDat : 频数分箱合并后的数据框
    xVarName : 自变量x名称
    yVarName : 目标变量y名称
    
    Returns
    -------
    列表：rate_step_bin_df-逾期率每步合并频数分布的数据框   rate_step_chisq_df-逾期率每步合并chisq值的数据框
    
    Examples
    --------
    inDf = freq_order_bin_df
    inBinDat = freq_ord_cmb_rst['orders_bin_freq_df']
    xVarName = 'ae_d3_id_nbank_cons_allnum'
    yVarName = 'TargetBad'
    order_rate_combine(inDf, inBinDat, xVarName, yVarName)
    '''
    
    inDfXVarName = "bin_%s" % xVarName
    
    # BadRate统计
    TmpTab = pd.crosstab(inDf[inDfXVarName], inDf[yVarName],
                         dropna=False,margins=True) 
    TmpTab['Rate'] = round(TmpTab[1] / TmpTab['All'],6)
    TmpTab = TmpTab[TmpTab.index != 'All']    
    TmpTab = TmpTab.sort_values(by = inDfXVarName).reset_index(drop=False).rename(columns={inDfXVarName:'Levels'})
    if inDf[inDfXVarName].isnull().sum() > 0:
        MissDf = inDf[inDf[inDfXVarName].isnull()]
        TmpTab.loc[len(TmpTab)] = [len(TmpTab)+1, MissDf.shape[0]-MissDf[yVarName].sum(), MissDf[yVarName].sum(), 
                                   MissDf.shape[0], round(MissDf[yVarName].sum()/MissDf.shape[0],6)]
    
    # 分箱增加序列并排序
    TmpBinDat = inBinDat[inBinDat['VarName'] == xVarName]
    TmpBinDat = pd.merge(TmpBinDat[['Levels','Bins']],TmpTab,on='Levels',
                         how='left')
    TmpBinDat = TmpBinDat.sort_values(by='Levels').reset_index(drop=True)

    
    # 缺失值新增行处理    
    if sum(TmpBinDat['Bins']=='nan') > 0:
        bads = inDf[inDf[inDfXVarName].isnull()][yVarName].sum()
        totals = inDf[inDf[inDfXVarName].isnull()][yVarName].count()
        goods = totals - bads
        OrderBinsStepDat = TmpBinDat.copy()
        OrderBinsStepDat.loc[OrderBinsStepDat['Bins']=='nan', 0] = goods
        OrderBinsStepDat.loc[OrderBinsStepDat['Bins']=='nan', 1] = bads
        OrderBinsStepDat.loc[OrderBinsStepDat['Bins']=='nan', 'All'] = totals
        OrderBinsStepDat.loc[OrderBinsStepDat['Bins']=='nan', 'Rate'] = round(bads/totals,6)
        NullBinDf = OrderBinsStepDat[OrderBinsStepDat['Bins'] == 'nan']
    else :
        OrderBinsStepDat=TmpBinDat.copy()
    OrderBinsStepDat['Steps']=0
    
    # Chisq计算
    #ChisqResult = chisq(OrderBinsStepDat[[0,1]].as_matrix())
    ChisqResult = chisq(OrderBinsStepDat[[0,1]].values)
    OrderChisqStepDat =pd.DataFrame([[0,OrderBinsStepDat[[0,1]].values.sum(),round(ChisqResult[0],2),
                                      round(ChisqResult[1],4), ChisqResult[2], round(ChisqResult[0]/ChisqResult[2],2)]],
                                      columns=['Steps','Sample_Size','Chisq_Stat','Chisq_P','Chisq_Df','Chisq_DfStat']
                                    )
    if TmpBinDat.shape[0] > 2:
        NotNullBinDf = TmpBinDat[TmpBinDat['Bins'] != 'nan']
        value_cnt = len(NotNullBinDf)  ## 非缺失值值的数量
        bin_cutoff = 2 ## 最小分箱数量
        if value_cnt > bin_cutoff: 
            for step in range(value_cnt - bin_cutoff):
                DiffBinDat = DataFrame(columns=['Level1','Level2','Bins','Diff'])
                value_cnt = len(NotNullBinDf)
                for i in range(value_cnt-1):                   
                    TmpDiff = np.abs(NotNullBinDf['Rate'][i]-NotNullBinDf['Rate'][i+1])
                    TmpBins = NotNullBinDf['Bins'][i] + ', ' + NotNullBinDf['Bins'][i+1]
                    DiffBinDat.loc[i] = [NotNullBinDf['Levels'][i], NotNullBinDf['Levels'][i+1], TmpBins, TmpDiff]   
                MinDiffSer=DiffBinDat.iloc[DiffBinDat[DiffBinDat['Diff']==min(DiffBinDat['Diff'])].index[0]] 
                TmpDrop = NotNullBinDf[NotNullBinDf['Levels'].isin([MinDiffSer['Level1'],MinDiffSer['Level2']])]
                TmpComb0=TmpDrop[0].sum()
                TmpComb1=TmpDrop[1].sum()
                TmpCombAll=TmpDrop['All'].sum()
                TmpCombRate=round(TmpComb1/TmpCombAll,6)
                NotNullBinDf = NotNullBinDf[~NotNullBinDf['Levels'].isin([MinDiffSer['Level1'],MinDiffSer['Level2']])]
                NotNullBinDf.loc[max(NotNullBinDf.index)+1] = [MinDiffSer['Level1'],MinDiffSer['Bins'],
                                                    TmpComb0, TmpComb1, TmpCombAll, TmpCombRate ]
                NotNullBinDf = NotNullBinDf.sort_values(by='Levels').reset_index(drop=True)
                NotNullBinDf['Levels'] = NotNullBinDf.index
                
                if sum(TmpBinDat['Bins']=='nan') > 0:
                    TmpStep=pd.concat([NotNullBinDf,NullBinDf], axis=0).sort_values(by='Levels').reset_index(drop=True)
                    TmpStep['Levels'] = TmpStep.index
                else :
                    TmpStep = NotNullBinDf.copy()
                TmpStep['Steps']=step+1
                OrderBinsStepDat = pd.concat([OrderBinsStepDat,TmpStep])
                #Chisq计算
                #ChisqResult2 = chisq(TmpStep[[0,1]].as_matrix())
                ChisqResult2 = chisq(TmpStep[[0,1]].values)
                TmpChisqDf =pd.DataFrame([[step+1,TmpStep[[0,1]].values.sum(),round(ChisqResult2[0],2),round(ChisqResult2[1],4),
                                         ChisqResult2[2], round(ChisqResult2[0]/ChisqResult2[2],2)]],
                                         columns=['Steps','Sample_Size','Chisq_Stat','Chisq_P','Chisq_Df','Chisq_DfStat']
                                         )
                OrderChisqStepDat=pd.concat([OrderChisqStepDat,TmpChisqDf]).reset_index(drop=True)
        else :
            TmpStep = OrderBinsStepDat.copy()
        
        # 缺失值合并        
        if sum(TmpBinDat['Bins']=='nan') > 0:
            null_rate = TmpStep.loc[TmpStep[TmpStep['Bins']=='nan'].index[0],'Rate']
            TmpStep['diff'] = np.abs(TmpStep['Rate']-null_rate)
            null_step = TmpStep[TmpStep['Bins'] != 'nan']
            min_dif_ser = null_step.iloc[null_step[null_step['diff'] == min(null_step['diff'])].index[0]]
            tmp_drop = TmpStep[TmpStep['Bins'].isin([min_dif_ser['Bins'], 'nan'])]
            comb_0 = tmp_drop[0].sum()
            comb_1 = tmp_drop[1].sum()
            comb_all = tmp_drop['All'].sum()
            comb_rate = round(comb_1/comb_all, 6)
            
            null_step.loc[null_step['Levels']== min_dif_ser['Levels'], 'Bins'] = min_dif_ser['Bins'] + ', ' + 'nan'
            null_step.loc[null_step['Levels']== min_dif_ser['Levels'], 0] = comb_0
            null_step.loc[null_step['Levels']== min_dif_ser['Levels'], 1] = comb_1
            null_step.loc[null_step['Levels']== min_dif_ser['Levels'], 'All'] = comb_all
            null_step.loc[null_step['Levels']== min_dif_ser['Levels'], 'Rate'] = comb_rate
            null_step['Steps'] = min_dif_ser['Steps']+1
            null_step = null_step.drop('diff', axis=1)
    
            OrderBinsStepDat = pd.concat([OrderBinsStepDat,null_step])
            #Chisq计算
            ChisqResult2 = chisq(null_step[[0,1]].as_matrix())
            OrderChisqStepDat.loc[len(OrderChisqStepDat)] = [min_dif_ser['Steps']+1, null_step[[0,1]].values.sum(),
                                                             round(ChisqResult2[0],2),round(ChisqResult2[1],4),
                                                             ChisqResult2[2], round(ChisqResult2[0]/ChisqResult2[2],2)]
            
    return {'rate_step_bin_df': OrderBinsStepDat,
            'rate_step_chisq_df': OrderChisqStepDat}




def order_df_rate_combine(inDf, inBinMapDat, yVarName):
    '''
    Function Descriptions:
        按照分箱的逾期率，对数据框中的连续变量进行逐步分箱合并
        
    Parameters
    ----------
    inDf     : 用于逾期率合并的数据框
    inBinDat : 频数分箱合并后的数据框
    yVarName : 目标变量y名称
    
    Returns
    -------
    列表：rate_step_bin_df-逾期率每步合并频数分布的数据框   rate_step_chisq_df-逾期率每步合并chisq值的数据框
    
    Examples
    --------
    inDf = freq_order_bin_df
    inBinMapDat = freq_ord_cmb_rst['orders_bin_freq_df']
    yVarName = var_target
    order_df_rate_combine(inDf, inBinMapDat, yVarName)
    '''
    
    OrderAllVarBinsStepDat=DataFrame(columns=['Levels','Bins',0,1,'All','Rate','Steps','VarName'])
    OrderAllVarChisqStepDat=DataFrame(columns=['Steps','Chisq_Stat','Chisq_P','Chisq_Df',
                                               'Chisq_DfStat','VarName'])
    
    OrderVarList = inBinMapDat['VarName'].unique().tolist()
    for TmpVar in OrderVarList:
        print('Order rate combine: ', TmpVar)
        InBinDat=inBinMapDat.copy()  #必须放在循环里面
        OrderBestBinResult = order_rate_combine(inDf, InBinDat, TmpVar, yVarName)
        TmpVarBinDat=OrderBestBinResult['rate_step_bin_df']
        TmpVarBinDat['VarName']=TmpVar
        TmpVarChisqDat = OrderBestBinResult['rate_step_chisq_df']
        TmpVarChisqDat['VarName']=TmpVar
        
        OrderAllVarBinsStepDat = pd.concat([OrderAllVarBinsStepDat,
                                            TmpVarBinDat]).reset_index(drop=True)
        OrderAllVarChisqStepDat=pd.concat([OrderAllVarChisqStepDat,
                                           TmpVarChisqDat]).reset_index(drop=True)
    OrderAllVarBinsStepDat['VarName'] = OrderAllVarBinsStepDat['VarName'].map(lambda x: "bin_%s" % x)
    OrderAllVarChisqStepDat['VarName'] = OrderAllVarChisqStepDat['VarName'].map(lambda x: "bin_%s" % x) 
    
    print("***有序变量逾期率合并完成！")
    return {'rate_step_bin_df': OrderAllVarBinsStepDat,
            'rate_step_chisq_df': OrderAllVarChisqStepDat
            }   

