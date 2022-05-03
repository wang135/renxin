# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.stats import chi2_contingency as chisq


def nominal_rate_combine(inDf, xVar, yVar):
    '''
    Function Descriptions:
        按照分箱的逾期率，对名义变量进行逐步分箱合并
        
    Parameters
    ----------
    inDf : 用于逾期率合并的数据框
    xVar : 自变量x名称
    yVar : 目标变量y名称
    
    Returns
    -------
    列表：rate_step_bin_df-逾期率每步合并频数分布的数据框   rate_step_chisq_df-逾期率每步合并chisq值的数据框
    
    Examples
    --------
    inDf = freq_nominal_bin_df
    xVar = 'bin_email_company'
    yVar = 'TargetBad'
    nominal_rate_combine(inDf, xVar, yVar)
    '''
    #生成列联表
    TmpTab = pd.crosstab(inDf[xVar], inDf[yVar],
                         dropna=False,margins=True)    
    #定义chisq结果表
    TmpPredPowerDat = DataFrame(columns=['Steps','Chisq_Stat','Chisq_P','Chisq_Df','Chisq_DfStat'])
    TmpChisqTable = TmpTab[[0,1]][TmpTab.index!='All']
    ChisqResult = chisq(TmpChisqTable.as_matrix())
    TmpDf = pd.DataFrame([[0,TmpChisqTable.values.sum(), round(ChisqResult[0],2),round(ChisqResult[1],4),
                          ChisqResult[2], round(ChisqResult[0]/ChisqResult[2],2)]],
                         columns=['Steps','Sample_Size','Chisq_Stat','Chisq_P','Chisq_Df','Chisq_DfStat'],index = [0])
    TmpPredPowerDat = pd.concat([TmpPredPowerDat,TmpDf]).reset_index(drop=True)    
    #生成Bin合并过程频数表
    TmpTab = TmpTab[TmpTab.index!='All']
    TmpTab['Rate'] = TmpTab[1] / TmpTab['All']
    TmpBestBinStepDat = TmpTab.reset_index()
    TmpBestBinStepDat['Steps'] = 0
    #指定合并的步长
    Steps=len(TmpTab.index)-1
    
    for Step in range(1,Steps):  
        
        ###分箱合并
        TmpTab = TmpTab[[0,1,'All','Rate']]
        TmpTabDict = TmpTab.to_dict()['Rate']
        VarList = list(TmpTab.index)
        TmpTabDf = DataFrame(columns=['Var1','Var2','Diff'])
        Norder=0
        #计算各Bin间的BadRate差值
        for i in range(len(VarList)-1):
            for j in range(i+1,len(VarList),1):
                TmpTabDf.loc[Norder] = [VarList[i], VarList[j],
                                   (np.abs(TmpTabDict[VarList[i]]-TmpTabDict[VarList[j]]))]
                Norder = Norder + 1
        #选择BadRate差值最小的Bin进行合并
        MinBin = TmpTabDf.sort_values('Diff').iloc[0]
        #BadRate差值最小的两个Bin相关的指标相加
        TmpMinSer = TmpTab.ix[[MinBin['Var1'],MinBin['Var2']],:].sum()    
        TmpNewRate = TmpMinSer[1] / TmpMinSer['All']
        #剔除BadRate差值最小的两个Bin
        TmpNewTab = TmpTab[~TmpTab.index.isin([MinBin['Var1'],MinBin['Var2']])]  
        #插入新生成的Bin
        TmpNewTab.loc[','.join([str(MinBin['Var1']),str(MinBin['Var2'])])] = [TmpMinSer[0],TmpMinSer[1],TmpMinSer['All'],TmpNewRate]
        
        #插入迭代记录表中
        TmpNewTab['Steps']=Step
        TmpKeyVarName = TmpNewTab.index.name
        TmpNewTab = TmpNewTab.reset_index()    
        TmpBestBinStepDat = pd.concat([TmpBestBinStepDat,TmpNewTab]).reset_index(drop=True)
        
        ###更新迭代表
        TmpTab = TmpNewTab.set_index(TmpKeyVarName)  
    
        ###新表的Chisq统计量计算
        TmpChisqTable = TmpTab[[0,1]][TmpTab.index!='All']
        ChisqResult = chisq(TmpChisqTable.as_matrix())
        TmpDf = pd.DataFrame([[Step,TmpChisqTable.values.sum(),round(ChisqResult[0],2),round(ChisqResult[1],4),
                              ChisqResult[2], round(ChisqResult[0]/ChisqResult[2],2)]],
                             columns=['Steps','Sample_Size','Chisq_Stat','Chisq_P','Chisq_Df','Chisq_DfStat'],index = [0])
        TmpPredPowerDat = pd.concat([TmpPredPowerDat,TmpDf]).reset_index(drop=True)  
              
    return {'rate_step_bin_df':TmpBestBinStepDat,
            'rate_step_chisq_df':TmpPredPowerDat}



def nominal_df_rate_combine(inDf, keyVarName, yVarName):
    '''
    Funcation Descriptions:
        对应数据框中的名义变量进行逐步分箱合并
        
    Paramters
    ---------
    inDf       : 经过频数合并后的数据框
    keyVarName : 主键变量
    yVarName   : 目标变量
    
    Returns
    -------
    列表：rate_step_bin_df - 逐步逾期率合并的频数及逾期率数据框；  rate_step_chisq_df - 逐步逾期率合并的chisq值数据框；
    
    Examples
    --------
    ##多变量逐步分箱合并
    inDf = freq_nominal_bin_df
    keyVarName = 'request_id'
    yVarName = 'TargetBad'
    '''
    NominalPredPowerDat = DataFrame(columns=['Steps', 'Chisq_Stat', 'Chisq_P', 'Chisq_Df','Chisq_DfStat','VarName'])
    NominalBestBinStepDat = DataFrame(columns=['Bins', 0.0, 1.0, 'All', 'Rate', 'Steps', 'VarName'])
    var_ls = list(filter(lambda x: x not in [keyVarName, yVarName], 
                         inDf.columns.tolist()))
    
    for var_item in var_ls:
        print('Nominal rate combine: ', var_item)
        BestBinResult = nominal_rate_combine(inDf = inDf, 
                                       xVar = var_item, 
                                       yVar = yVarName)
        TmpBestBin = BestBinResult['rate_step_bin_df'].rename(columns={var_item:'Bins'})
        TmpBestBin['VarName'] = var_item
        TmpPredPower = BestBinResult['rate_step_chisq_df']
        TmpPredPower['VarName'] = var_item
        #各变量的预测力表和合并过程表的集合
        NominalPredPowerDat = pd.concat([NominalPredPowerDat,TmpPredPower]).reset_index(drop=True)
        NominalBestBinStepDat = pd.concat([NominalBestBinStepDat,TmpBestBin]).reset_index(drop=True)
    
    print("***名义变量逾期率分箱合并完成！")
    return {'rate_step_chisq_df': NominalPredPowerDat,
            'rate_step_bin_df': NominalBestBinStepDat
            }
    



def _code_map(x, freqBinDf, freqLevel, freqBin):
    raw_dict = dict(zip(freqBinDf['Levels'], freqBinDf['Bins']))
    values = list()
    for item in str(x).split(','):
        values.append(raw_dict[float(item)])
    return ','.join(values)


def nominal_code_map(inDf, inFreqDf, inVar, inBin, inFreqVar, inFreqLevel, inFreqBin):
    '''
    Function Descriptions:
        把名义变量的分箱码值转换为其真实值， 例如：1代表渠道A
        
    Parameters
    ----------
    inDf         : 待转换的数据框
    inFreqDf     : 具有码值含义的数据框
    inVar        : 存放变量名称的列，需要针对不同的变量值进行逐步转换
    inBin        : 存放码值的列
    inFreqVar    : 码表（inFreqDf）中存放变量名称的列
    inFreqLevel  : 码表（inFreqDf）中存放码值的列
    inFreqBin    : 码表（inFreqDf）中存放真实值的列
    
    Returns
    -------
    码值转换后的数据框
    
    Examples
    --------
    inDf = rate_nom_cmb_rst['rate_step_bin_df']
    inFreqDf = freq_nom_cmb_rst['bin_freq_df']
    inVar = 'VarName'
    inBin = 'Bins'
    inFreqVar = 'VarName'
    inFreqLevel = 'Levels'
    inFreqBin = 'Bins'
    nominal_bin_map(inDf, inFreqDf, inVar, inBin, inFreqVar, inFreqLevel, inFreqBin)
    '''
    rate_new_df = inDf.head(0)
    rate_new_df['RawBin'] = ''
    for var_item in inFreqDf[inFreqVar].unique().tolist():
        print('Nominal code map: ', var_item)
        freq_bin = inFreqDf[inFreqDf[inFreqVar] == var_item]
        rate_bin = inDf[inDf[inVar] == 'bin_{}'.format(var_item)]
        rate_bin['RawBin'] = rate_bin[inBin].map(lambda x: _code_map(x, freqBinDf=freq_bin, freqLevel=inFreqLevel, freqBin=inFreqBin))
        rate_new_df = pd.concat([rate_new_df, rate_bin], axis=0, ignore_index = True)
    
    rate_new_df[inBin] = rate_new_df['RawBin']
    rate_new_df = rate_new_df.drop('RawBin', axis=1) 
    
    return rate_new_df











