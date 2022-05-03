# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:10:12 2020

@author: finup
"""

from base.freq_stats import var_freq_dist
from pandas import DataFrame
import pandas as pd



def nominal_freq_combine(xVar, cutOff=0.05):
    
    '''
    Function Description:
        对名义变量样本占比低的类进行合并
    
    Parameters
    ----------
    xVar   : 频数合并的序列
    cutOff : 分箱样本占比小于该值被合并，取值范围为[0，1]的小数，默认为：freqCutOff=0.05
    
    Returns
    -------
    分箱合并后的序列
    
    Examples
    ---------
    xVar = ins_clean_df['email_company']
    nominal_freq_combine(xVar, cutOff=0.05)
    
    '''
    
    #原始变量的频数统计
    TmpFreqResult=var_freq_dist(xVar, pctFormat=False)
    
    #获取变量值列表
    TmpValueList = list(TmpFreqResult.index)
    TmpValueList.pop()

    if TmpFreqResult['Rate'].min() < cutOff:
        #计算样本占比大于等于CutOff的分类的累计百分比。
        TmpFreqDict = TmpFreqResult.to_dict()['Rate']
        TmpPct=0
        for TmpValue in TmpValueList: 
            if TmpFreqDict[TmpValue] >= cutOff :
                TmpPct = TmpPct + TmpFreqDict[TmpValue]
              
        #获取样本占比大于CutOff的分类。
        FreqDat = TmpFreqResult[(TmpFreqResult['Rate']>=cutOff) & (TmpFreqResult.index != 'Total')]
        #获取满足条件的样本占比最小的百分比        
        TmpFloorDat = FreqDat.sort_values(by='Freq').head(1)
        print("aaaaaaaaa",TmpFloorDat)
        TmpFloorLevel=TmpFloorDat.index[0]
        TmpFloorFreqDict = TmpFloorDat.to_dict()['Freq']
        TmpFloorRateDict = TmpFloorDat.to_dict()['Rate']
        #索引变为列值
        FreqDat=FreqDat.reset_index()
        
        #频数合并
        if (TmpPct>0.9) & (TmpPct<=0.95): #假如频数占比大于5%分类的和大于90%，则把所有频数占比小于5%的分类合并为一类
            TmpFreqDict = TmpFreqResult.to_dict()['Freq']
            TmpRateDict = TmpFreqResult.to_dict()['Rate']
            TmpPctList=list()
            TmpCulPct=0
            TmpCulFreq=0
            for TmpValue in TmpValueList:
                if TmpRateDict[TmpValue]<0.05:
                    TmpPctList.append(TmpValue)
                    TmpCulFreq=TmpCulFreq+TmpFreqDict[TmpValue]
                    TmpCulPct=TmpCulPct+TmpRateDict[TmpValue]
            FreqDat.loc[FreqDat['Rate'].count()]=[",".join(TmpPctList),TmpCulFreq,TmpCulPct]

        elif TmpPct>0.95: 
            TmpFreqDict = TmpFreqResult.to_dict()['Freq']
            TmpRateDict = TmpFreqResult.to_dict()['Rate']
            TmpPctList=list()
            TmpCulPct=0
            TmpCulFreq=0
            for TmpValue in TmpValueList:
                if TmpRateDict[TmpValue]<0.05:
                    TmpPctList.append(TmpValue)
                    TmpCulFreq=TmpCulFreq+TmpFreqDict[TmpValue]
                    TmpCulPct=TmpCulPct+TmpRateDict[TmpValue]
            ##若不满足条件的所有类之和小于5%，则向上累加
            TmpPctList.append(TmpFloorLevel)
            TmpCulFreq=TmpCulFreq+TmpFloorFreqDict[TmpFloorLevel]
            TmpCulPct=TmpCulPct+TmpFloorRateDict[TmpFloorLevel]
            FreqDat=FreqDat[FreqDat['index']!=TmpFloorLevel]
            FreqDat.loc[FreqDat['Rate'].count()]=[",".join(TmpPctList),TmpCulFreq,TmpCulPct]

        else:#假如频数占比大于5%分类的和小于等于90%，则把剩余频数占比小于5%的类，按照由小到大的顺序进行合并。
            #选择频数占比小于FreqCutOff的所有类
            TmpSmallCell = TmpFreqResult[TmpFreqResult['Rate']<cutOff].sort_values(by='Freq',ascending=False)
            TmpSmallValueList = list(TmpSmallCell.index)
            TmpFreqDict = TmpSmallCell.to_dict()['Freq']
            TmpRateDict = TmpSmallCell.to_dict()['Rate']
            TmpCulList=list()
            TmpCulPct=0
            TmpCulFreq=0
            for TmpSmall in TmpSmallValueList:
                TmpPct = TmpPct + TmpRateDict[TmpSmall]
                TmpCulFreq = TmpCulFreq+TmpFreqDict[TmpSmall]
                TmpCulPct = TmpCulPct+TmpRateDict[TmpSmall]
                TmpCulList.append(TmpSmall) 
                #累计分类的频数占比大于FreqCutOff
                if (TmpPct <= 0.9) & (TmpCulPct > 0.05):
                    FreqDat.loc[FreqDat['Rate'].count()]=[",".join(TmpCulList),TmpCulFreq,TmpCulPct]
                    TmpCulList=list()
                    TmpCulPct=0
                    TmpCulFreq=0
                    
            if TmpCulFreq > 0:
                FreqDat.loc[FreqDat['Rate'].count()]=[",".join(TmpCulList),TmpCulFreq,TmpCulPct]

    else:
         FreqDat = TmpFreqResult[TmpFreqResult.index != 'Total'] 
         FreqDat=FreqDat.reset_index() 
         
    return {'raw_freq_df': TmpFreqResult,
            'bin_freq_df': FreqDat}




def _nominal_value_map(x, valueList):
    '''
    名义变量值与分箱值对照
    '''
    for item in valueList:
        if x in str(item).split(','):
            return item

    

def nominal_freq_combine_transfer(xVar, cutOff=0.05):
    
    '''
    Function Description:
        对名义变量样本占比低的类进行合并,并生成新序列
    
    Parameters
    ----------
    xVar   : 频数合并的序列
    cutOff : 分箱样本占比小于该值被合并，取值范围为[0，1]的小数，默认为：freqCutOff=0.05
    
    Returns:
    分箱合并后的序列
    '''
    FreqDat = nominal_freq_combine(xVar, cutOff=0.05)['bin_freq_df']
    
    var_bin = xVar.map(lambda x: _nominal_value_map(x, valueList=FreqDat['index'].tolist()))
    var_bin.name = "bin_%s" % var_bin.name  ## 分箱序列重命名
         
    return var_bin



#多变量频数分箱
def nominal_ls_freq_combine(inDf, varList, cutOff=0.05):
    '''
    Function Descriptions:
        对于多个名义变量中，不满足频数占比的分箱进行频数合并
        
    Parameters
    ----------
    inDf    : 数据框
    varList : 变量列表
    cutOff  : 分箱合并的标准频数占比
    
    Returns
    -------
    列表： raw_freq_df-初始变量的频数分布数据框  bin_freq_df-分箱合并后的频数分布数据框
    
    Examples
    --------
    inDf = ins_clean_df
    varList = ins_var_class_df[ins_var_class_df['Dclass']=='Nominal']['index'].tolist()
    curOff = 0.05
    nominal_ls_freq_combine(inDf, varList, cutOff=0.05)
    '''

    RawBinDf = DataFrame(columns=['VarName', 'Levels', 'Bins', 'Freq', 'Rate'])
    NewBinDf = DataFrame(columns=['VarName', 'Levels', 'Bins', 'Freq', 'Rate'])

    for var_name in varList:
        
        CombineResult = nominal_freq_combine(xVar=inDf[var_name], cutOff=cutOff)
        RawFreqDat=CombineResult['raw_freq_df']
        BinFreqDat=CombineResult['bin_freq_df']
        
        RawFreqDat = RawFreqDat.reset_index().rename(columns={'index':'Bins'})
        RawFreqDat = RawFreqDat.reset_index().rename(columns={'index':'Levels'})
        RawFreqDat['Levels'] = RawFreqDat['Levels']+1
        RawFreqDat['VarName'] = var_name
        RawBinDf = pd.concat([RawBinDf,RawFreqDat], axis=0, ignore_index=1)
        
        BinFreqDat = BinFreqDat.rename(columns={'index':'Bins'})
        BinFreqDat = BinFreqDat.reset_index().rename(columns={'index':'Levels'})
        BinFreqDat['Levels'] = BinFreqDat['Levels']+1
        BinFreqDat['VarName'] = var_name
        NewBinDf = pd.concat([NewBinDf,BinFreqDat], axis=0, ignore_index=1)

    RawBinDf = RawBinDf[['VarName', 'Levels', 'Bins', 'Freq', 'Rate']]
    NewBinDf = NewBinDf[['VarName', 'Levels', 'Bins', 'Freq', 'Rate']]

    
    print('***名义变量频数合并完成！')
    return {'raw_freq_df':RawBinDf,
            'bin_freq_df':NewBinDf}





def _nominal_bin_value_map(x, valueDict):
    '''
    名义变量值与分箱值对照
    x = 'yidong'
    valueDict = value_df
    '''
    
    for keys in valueDict.keys():
        if x in str(valueDict[keys]).split(','):
            return valueDict[keys]



def nominal_bin_transfer(inRawDf, inMapDf, inKeyName, inTargetName):
    '''
    Function Descriptions:
        使用新分箱的频数数据集，对原始数据集进行重新分箱
        
    Parameters
    ----------
    inRawDf      : 原始数据框
    inMapDf      : 新分箱的频数分布数据框
    inKeyName    : 数据框中的主键变量
    inTargetName : 数据框中的目标变量
    
    Returns
    -------
    数据框： 重新分箱后的数据框
    '''        
    var_ls = list(inMapDf['VarName'].unique())
    nominal_df = pd.DataFrame(inRawDf[inKeyName])
    for var_item in var_ls:
        print('Nominal freq map: ', var_item)
        value_df = inMapDf[inMapDf['VarName'] == var_item]
        value_dict = value_df['Bins'].to_dict() 
        var_bin = inRawDf[var_item].map(lambda x: _nominal_bin_value_map(x, valueDict=value_dict))
        var_bin.name = "bin_%s" % var_bin.name
        nominal_df = nominal_df.merge(var_bin, left_index=True, right_index=True, how='left')
    nominal_df = nominal_df.merge(inRawDf[inTargetName], left_index=True, right_index=True, how='left')
    
    print('***名义变量频数合并map完成！')
    return nominal_df










