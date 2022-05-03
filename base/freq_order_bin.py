# -*- coding: utf-8 -*-

import pandas as pd
from pandas import DataFrame
from base.freq_stats import var_freq_dist
from intervals import FloatInterval


   
def _order_freq_combine(inFreqDf, cutOff):
    '''
    Function Description:
        一个变量单次频数合并
    '''
    RecordCnt = len(inFreqDf.index) 
    for i in range(RecordCnt):
        if (i==0) & (inFreqDf['Rate'][i] < cutOff):
            TmpCombDat=inFreqDf.iloc[[i,i+1]].sum()
            TmpComFreq=TmpCombDat['Freq']
            TmpComRate=TmpCombDat['Rate']
            TmpComIndex=i+1
            TmpComBin=[inFreqDf['Bins'][i],inFreqDf['Bins'][i+1]]            
            TmpFreqKeepDat=inFreqDf[~inFreqDf.index.isin([i,i+1])]
            TmpDat=pd.DataFrame([[TmpComIndex,TmpComFreq,TmpComRate,TmpComBin]], 
                             columns=['index','Freq','Rate','Bins'])
            inFreqDf = pd.concat([TmpFreqKeepDat,TmpDat]).sort_values(by='index').reset_index(drop=True)
            break
        elif (i<RecordCnt-1) & (inFreqDf['Rate'][i] < cutOff):
            if inFreqDf['Rate'][i-1]>inFreqDf['Rate'][i+1]:
                TmpCombDat=inFreqDf.iloc[[i,i+1]].sum()
                TmpComFreq=TmpCombDat['Freq']
                TmpComRate=TmpCombDat['Rate']
                TmpComIndex=i+1
                TmpComBin=[inFreqDf['Bins'][i],inFreqDf['Bins'][i+1]]            
                TmpFreqKeepDat=inFreqDf[~inFreqDf.index.isin([i,i+1])]
                TmpDat=pd.DataFrame([[TmpComIndex,TmpComFreq,TmpComRate,TmpComBin]], 
                                 columns=['index','Freq','Rate','Bins'])
                inFreqDf = pd.concat([TmpFreqKeepDat,TmpDat]).sort_values(by='index').reset_index(drop=True)
                break
            elif inFreqDf['Rate'][i-1]<inFreqDf['Rate'][i+1]:
                TmpCombDat=inFreqDf.iloc[[i-1,i]].sum()
                TmpComFreq=TmpCombDat['Freq']
                TmpComRate=TmpCombDat['Rate']
                TmpComIndex=i
                TmpComBin=[inFreqDf['Bins'][i-1],inFreqDf['Bins'][i]]            
                TmpFreqKeepDat=inFreqDf[~inFreqDf.index.isin([i-1,i])]
                TmpDat=pd.DataFrame([[TmpComIndex,TmpComFreq,TmpComRate,TmpComBin]], 
                                 columns=['index','Freq','Rate','Bins'])
                inFreqDf = pd.concat([TmpFreqKeepDat,TmpDat]).sort_values(by='index').reset_index(drop=True)
                break
            else :
                TmpCombDat=inFreqDf.iloc[[i,i+1]].sum()
                TmpComFreq=TmpCombDat['Freq']
                TmpComRate=TmpCombDat['Rate']
                TmpComIndex=i+1
                TmpComBin=[inFreqDf['Bins'][i],inFreqDf['Bins'][i+1]]            
                TmpFreqKeepDat=inFreqDf[~inFreqDf.index.isin([i,i+1])]
                TmpDat=pd.DataFrame([[TmpComIndex,TmpComFreq,TmpComRate,TmpComBin]], 
                                 columns=['index','Freq','Rate','Bins'])
                inFreqDf = pd.concat([TmpFreqKeepDat,TmpDat]).sort_values(by='index').reset_index(drop=True)
                break
        elif (i==RecordCnt-1) & (inFreqDf['Rate'][i] < cutOff):
            TmpCombDat=inFreqDf.iloc[[i-1,i]].sum()
            TmpComFreq=TmpCombDat['Freq']
            TmpComRate=TmpCombDat['Rate']
            TmpComIndex=i
            TmpComBin=[inFreqDf['Bins'][i-1],inFreqDf['Bins'][i]]            
            TmpFreqKeepDat=inFreqDf[~inFreqDf.index.isin([i-1,i])]
            TmpDat=pd.DataFrame([[TmpComIndex,TmpComFreq,TmpComRate,TmpComBin]], 
                             columns=['index','Freq','Rate','Bins'])
            inFreqDf = pd.concat([TmpFreqKeepDat,TmpDat]).sort_values(by='index').reset_index(drop=True)
            break 
        
    return inFreqDf




##单变量多次频数合并
def order_freq_combine(xVar, cutOff=0.05):
    '''
    Function Descriptions:
        针对有序变量样本占比低的分箱进行合并
    
    Parameters
    ----------
    xVar   : 频数合并的序列
    cutOff : 分箱样本占比小于该值被合并，取值范围为[0，1]的小数，默认为：freqCutOff=0.05
    
    Returns
    -------
    生成分箱合并后的序列
    
    Examples
    --------
    xVar = ins_clean_df['educations']
    order_freq_combine(xVar, cutOff=0.05)
    '''
    
    TmpFreqResult=var_freq_dist(xVar, pctFormat=False)
    TmpFreqResult = TmpFreqResult[TmpFreqResult.index != 'Total'].sort_index().reset_index()  
    OrderFreqRawDat = TmpFreqResult.copy() 
    OrderFreqRawDat = OrderFreqRawDat.rename(columns={'index':'Bins'})
    TmpFreqResult = TmpFreqResult.rename(columns={'index':'Bins'}) 
    TmpFreqResult = TmpFreqResult.reset_index()
    
    ## 判断是否存在缺失值
    NullValueTag = TmpFreqResult['index'].isnull().sum()
    if NullValueTag > 0:
        NullValueRate = TmpFreqResult[TmpFreqResult['index'].isnull()]['Rate'].tolist()[0]
        ## 确定缺失率是否超过cutOff
        if NullValueRate > cutOff:
            NullFreqResult = TmpFreqResult[TmpFreqResult['index'].isnull()]
            TmpFreqResult = TmpFreqResult[TmpFreqResult['index'].notnull()]
        else :
            BinsValue = TmpFreqResult[TmpFreqResult['Freq']==TmpFreqResult['Freq'].max()]['Bins'].tolist()[0]
            NullDict=dict(TmpFreqResult[(TmpFreqResult['Bins'].isnull()) | (TmpFreqResult['Bins']==BinsValue)].sum())
            TmpFreqResult.loc[TmpFreqResult['Bins']==BinsValue,'Freq'] = NullDict['Freq']
            TmpFreqResult.loc[TmpFreqResult['Bins']==BinsValue,'Rate'] = NullDict['Rate']
            TmpFreqResult.loc[TmpFreqResult['Bins']==BinsValue,'Bins'] = "[{}, NaN]".format(BinsValue)
            TmpFreqResult = TmpFreqResult[TmpFreqResult['Bins'].notnull()]
    
    
    StepTimes = TmpFreqResult[TmpFreqResult['Rate']<cutOff].index.size
    while StepTimes>0 :
        OrderOneStepDat = _order_freq_combine(inFreqDf=TmpFreqResult, cutOff=cutOff)
        TmpFreqResult=OrderOneStepDat.drop('index',axis=1)
        TmpFreqResult=TmpFreqResult.reset_index()
        StepTimes = TmpFreqResult[TmpFreqResult['Rate']<cutOff].index.size
        
    if (NullValueTag > 0):
        if (NullValueRate > cutOff):
            TmpFreqResult = pd.concat([TmpFreqResult,NullFreqResult], axis=0)
    
    TmpFreqResult['index'] = TmpFreqResult['index']+1
    TmpFreqResult = TmpFreqResult.rename(columns={'index':'Levels'})
    
    return {'raw_freq_df': OrderFreqRawDat,
            'bin_freq_df': TmpFreqResult}



def _order_value_map(xValue, valueList):
    for item in valueList:
        if str(xValue) in item.split(', '):
            return item   
        

def order_freq_combine_transfer(xVar, cutOff=0.05):
    '''
    Function Descriptions:
        针对有序变量样本占比低的分箱进行合并,并生成新序列
    
    Parameters
    ----------
    xVar   : 频数合并的序列
    cutOff : 分箱样本占比小于该值被合并，取值范围为[0，1]的小数，默认为：freqCutOff=0.05
    
    Returns
    -------
    生成分箱合并后的序列
    
    Examples
    --------
    xVar = ins_clean_df['last2year_residence_changes']
    order_freq_combine_transfer(xVar, cutOff=0.05)
    '''
    
    FreqResult = order_freq_combine(xVar, cutOff=0.05)['bin_freq_df']
    
    FreqSer = FreqResult['Bins'].astype(str).apply(lambda x: x.replace('[','').replace(']',''))
    
    var_bin = xVar.map(lambda x: _order_value_map(x, valueList=FreqSer.tolist()))
    var_bin.name = "bin_%s" % var_bin.name  ## 分箱序列重命名
    
    return var_bin         



def order_ls_freq_combine(inDf, varList, cutOff=0.05):
    '''
    Function Descriptions:
        多个连续变量低样本占比分箱合并
    
    Parameters
    ----------
    inDf    : 原始数据框
    varList : 变量列表
    cutOff  : 分箱合并的样本占比值
    
    Returns
    -------
    数据框：orders_bin_freq_df-分箱合并后的频数数据框  orders_raw_freq_df-原始的频数数据框
    
    Examples
    --------
    inDf = ins_clean_df
    varList = ['gender']
    cutOff=0.05
    '''
    #记录所有连续变量的原始频数统计
    OrderFreqBinDat = DataFrame(columns=['Levels','Freq','Rate','Bins','VarName'])
    #记录所有连续变量的频数合并后的频数统计
    OrderFreqRawDat = DataFrame(columns=['Levels','Freq','Rate','Bins','VarName'])
    for TmpVar in varList:
        OrderFreqBinResult = order_freq_combine(xVar=inDf[TmpVar], cutOff=cutOff)
        TmpOrderFreqBinDat = OrderFreqBinResult['bin_freq_df']
        TmpOrderFreqRawDat = OrderFreqBinResult['raw_freq_df']
        TmpOrderFreqBinDat['VarName'] = TmpVar
        TmpOrderFreqRawDat['VarName'] = TmpVar
        OrderFreqBinDat = pd.concat([OrderFreqBinDat,TmpOrderFreqBinDat]).reset_index(drop=True)
        OrderFreqRawDat = pd.concat([OrderFreqRawDat,TmpOrderFreqRawDat]).reset_index(drop=True)
        OrderFreqRawDat['Levels'] = OrderFreqRawDat['Bins'].map(lambda x: int(x) if pd.isnull(x)==0 else x)
    OrderFreqBinDat['Bins'] = OrderFreqBinDat['Bins'].astype(str).apply(lambda x: x.replace('[','').replace(']','')).tolist()    
    print("***有序变量频数合并完成！")
    return {'orders_bin_freq_df': OrderFreqBinDat,
            'orders_raw_freq_df': OrderFreqRawDat}




def _order_bin_value_maps(x, valueList):
    '''
    未启用
    '''
    
    if 'nan' in valueList:
        value_ls = list(filter(lambda x: x not in ['nan'], valueList))
    else :
        value_ls = valueList
    
    if 'nan' in valueList and pd.isnull(x):
        return 'Missing'
    else :
        for i in range(len(value_ls)):
            if i == 0:
                if x <= float(value_ls[0].split(', ')[-1]):
                    return value_ls[0]
            elif i == len(value_ls)-1:
                if x >= float(value_ls[-1].split(', ')[0]):
                    return value_ls[-1]
            else :
                if x >= float(value_ls[i].split(', ')[0]) and x < float(value_ls[i+1].split(', ')[0]):
                    return value_ls[i]


'''
def _order_bin_value_map(x, valueDf):
        
    for i in range(len(valueDf)):
        if i == 0:
            if x <= float(valueDf.ix[0]['Bins'].split(', ')[-1]):
                return valueDf.ix[0]['Levels']
        elif i == len(valueDf)-1:
            if x >= float(valueDf.ix[len(valueDf)-1]['Bins'].split(', ')[0]):
                return valueDf.ix[len(valueDf)-1]['Levels']
        else :
            if x >= float(valueDf.ix[i]['Bins'].split(', ')[0]) and x < float(valueDf.ix[i+1]['Bins'].split(', ')[0]):
                return valueDf.ix[i]['Levels']
'''


def _order_bin_value_map(x, valueDict):
    '''
    valueDict = value_dict
    '''
    for level in valueDict.keys():
        if level == min(valueDict.keys()):
            if x <= valueDict[level].upper:
                return level
        elif level == max(valueDict.keys()):
            if x > valueDict[level].lower:
                return level
        elif x > valueDict[level].lower and x <= valueDict[level].upper:
            return level

    
                
def order_bin_transfer(inRawDf, inMapDf, inKeyName, inTargetName):
    '''
    Function Descriptions:
        频数分箱后，生成新的分箱数据集
    
    Parameters
    ----------
    inRawDf      : 需要进行分箱的数据框
    inMapDf      : 分箱标准数据框
    inKeyName    : inRawDf的主键变量
    inTargetName : inRawDf的目标变量
        
    Returns
    -------
    有序变量重新分箱后的数据框
        
    Examples
    --------
    inRawDf = ins_clean_df
    inMapDf = freq_ord_cmb_rst['orders_bin_freq_df']
    inKeyName = 'request_id'
    inTargetName = 'TargetBad'
    '''        
    var_ls = list(inMapDf['VarName'].unique())
    order_df = pd.DataFrame(inRawDf[inKeyName])
    for var_item in var_ls:
        print('Order freq map: ', var_item)
        #var_item = 'marriage_status'
        value_df = inMapDf[inMapDf['VarName'] == var_item][['Bins','Levels']]
        # 剔除空值分箱    
        if sum(value_df['Bins'] == 'nan'):
            value_df = value_df[value_df['Bins']!='nan']
        value_df = value_df.sort_values(by = 'Levels').reset_index(drop=True)
        
        #数据框转换为字典，值列表转换为区间
        value_dict = dict()
        for level in value_df['Levels'].tolist():
            if level == min(value_df['Levels'].tolist()):
                value_dict[level] = FloatInterval.open_closed(float('-inf'),value_df[value_df['Levels']==level]['Bins'].values[0].split(',')[-1])
            elif level == max(value_df['Levels'].tolist()):
                value_dict[level] = FloatInterval.open(value_df[value_df['Levels']==level-1]['Bins'].values[0].split(',')[-1], float('inf'))
            else :
                value_dict[level] = FloatInterval.open_closed(value_df[value_df['Levels']==level-1]['Bins'].values[0].split(',')[-1],value_df[value_df['Levels']==level]['Bins'].values[0].split(',')[-1])
       
        # 针对非空值分箱进行赋值，空值分箱仍为空值    
        var_bin = inRawDf[var_item].map(lambda x: _order_bin_value_map(x, valueDict=value_dict))
        var_bin.name = "bin_%s" % var_bin.name
        order_df = order_df.merge(var_bin, left_index=True, right_index=True, how='left')
    order_df = order_df.merge(inRawDf[inTargetName], left_index=True, right_index=True, how='left')
    
    print("***有序变量频数合并map完成！")
    return order_df
    












