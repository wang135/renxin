# -*- coding: utf-8 -*-

import pandas as pd
import copy
from intervals import FloatInterval

def _nom_value_map(x, valueDict):
    '''
    名义变量值与分箱值对照
    x = '6'
    valueDict = value_dict
    '''    
    for keys in valueDict.keys():
        if x in str(valueDict[keys]).split(','):
            return valueDict[keys]

def nominal_rate_bin_transfer(inRawDf, inMapDf, inKeyName, inTargetName):
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
    
    Examples
    --------
    inRawDf = xgb_ins_feature
    inMapDf = bin_power_dist_df
    inKeyName = 'request_id'
    inTargetName = 'TargetBad'
    '''        
    var_ls = list(inMapDf['NewVarName'].unique())
    nominal_df = pd.DataFrame(inRawDf[inKeyName])
    for var_item in var_ls:
        print('Nominal rate map: ', var_item)
        value_df = inMapDf[inMapDf['NewVarName'] == var_item]
        ls = var_item.split('_')
        ls.pop(0)
        ls.pop(-1)
        var_name = '_'.join(ls)        
        value_dict = value_df['Bins'].to_dict()        
        var_bin = inRawDf[var_name].map(lambda x: _nom_value_map(x, valueDict=value_dict))
        var_bin.name = var_item
        nominal_df = nominal_df.merge(var_bin, left_index=True, right_index=True, how='left')
    if len(inTargetName) > 1:
        nominal_df = nominal_df.merge(inRawDf[inTargetName], left_index=True, right_index=True, how='left')
    
    print('***名义变量生成分箱变量完成！')
    return nominal_df



def _order_bin_value_map(x, valueDict):
    '''
    x=model_raw_df['educations'][0]
    valueDict=value_dict
    _order_bin_value_map(x, valueDict)
    '''
    del_nan_lst = list(valueDict.keys())
    if 'nan' in del_nan_lst:
        del_nan_lst.remove('nan')
    for level in valueDict.keys():
        if level == 'nan':
            if pd.isnull(x):
                return valueDict[level]
        elif level == min(del_nan_lst) :
            if x <= valueDict[level].upper:
                return level
        elif level == max(del_nan_lst) :
            if x > valueDict[level].lower:
                return level
        elif x > valueDict[level].lower and x <= valueDict[level].upper:
            return level

def order_rate_bin_transfer(inRawDf, inMapDf, inKeyName, inTargetName):
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
    inRawDf = model_raw_df
    inMapDf = bin_power_dist_df[bin_power_dist_df['Dclass']=='Order']
    inMapDf = bin_power_dist_df[bin_power_dist_df['NewVarName']=='bin_query_consumer_1m_cnt_1']
    inKeyName = 'request_id'
    inTargetName = 'TargetBad'
    '''        
    var_ls = list(inMapDf['NewVarName'].unique())
    order_df = pd.DataFrame(inRawDf[inKeyName])
    for var_item in var_ls:
        print('Order rate map:', var_item)
        # var_item = 'bin_educations_2'
        value_df = inMapDf[inMapDf['NewVarName'] == var_item][['Bins','Levels']].copy()
        # 剔除空值分箱    
        if sum(value_df['Bins'] == 'nan'):
            value_df = value_df[value_df['Bins']!='nan']
        value_df = value_df.sort_values(by = 'Levels').reset_index(drop=True)
        
        #数据框转换为字典，值列表转换为区间
        value_dict = dict()
        #确定是否有缺失值归并
        value_df['Bins_nan'] = value_df['Bins'].apply(lambda x: 1 if 'nan' in x.split(', ') else 0)
        if value_df['Bins_nan'].sum() > 0:
            value_dict['nan'] = value_df[value_df['Bins_nan']==1]['Levels'].values[0]
            #剔除缺失值进一步进行处理
            value_df['Bins'] = value_df['Bins'].apply(lambda x: x.replace(', nan',''))
            for level in value_df['Levels'].tolist():
                if level == min(value_df['Levels'].tolist()):
                    value_dict[level] = FloatInterval.open_closed(float('-inf'),value_df[value_df['Levels']==level]['Bins'].values[0].split(',')[-1])
                elif level == max(value_df['Levels'].tolist()):
                    value_dict[level] = FloatInterval.open(value_df[value_df['Levels']==level-1]['Bins'].values[0].split(',')[-1], float('inf'))
                else :
                    value_dict[level] = FloatInterval.open_closed(value_df[value_df['Levels']==level-1]['Bins'].values[0].split(',')[-1], value_df[value_df['Levels']==level]['Bins'].values[0].split(',')[-1])
        else:
            for level in value_df['Levels'].tolist():
                if level == min(value_df['Levels'].tolist()):
                    value_dict[level] = FloatInterval.open_closed(float('-inf'),value_df[value_df['Levels']==level]['Bins'].values[0].split(',')[-1])
                elif level == max(value_df['Levels'].tolist()):
                    value_dict[level] = FloatInterval.open(value_df[value_df['Levels']==level-1]['Bins'].values[0].split(',')[-1], float('inf'))
                else :
                    value_dict[level] = FloatInterval.open_closed(value_df[value_df['Levels']==level-1]['Bins'].values[0].split(',')[-1], value_df[value_df['Levels']==level]['Bins'].values[0].split(',')[-1])
        
        # 针对非空值分箱进行赋值，空值分箱仍为空值  
        ls = var_item.split('_')
        ls.pop(0)
        ls.pop(-1)
        var_name = '_'.join(ls)  
        var_bin = inRawDf[var_name].map(lambda x: _order_bin_value_map(x, valueDict=value_dict))
        var_bin.name = var_item
        order_df = order_df.merge(var_bin, left_index=True, right_index=True, how='left')
    if len(inTargetName) > 1:
        order_df = order_df.merge(inRawDf[inTargetName], left_index=True, right_index=True, how='left')
    
    print("***有序变量生成分箱变量完成！")
    return order_df
    



def _cont_bin_value_map(x, valueDict):
    '''
    x=model_raw_df['educations'][0]
    valueDict=value_dict
    _order_bin_value_map(x, valueDict)
    '''
    del_nan_lst = list(valueDict.keys())
    if 'nan' in del_nan_lst:
        del_nan_lst.remove('nan')
    for level in valueDict.keys():
        if level == 'nan':
            if pd.isnull(x):
                return valueDict[level]
        elif level == min(del_nan_lst) :
            if x <= valueDict[level].upper:
                return level
        elif level == max(del_nan_lst) :
            if x > valueDict[level].lower:
                return level
        elif x > valueDict[level].lower and x <= valueDict[level].upper:
            return level
  
 #con_rate_bin_transfer(df1,df1[''])               
def con_rate_bin_transfer(inRawDf, inMapDf, inKeyName, inTargetName):
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
    inRawDf = model_raw_df
    inMapDf = bin_power_dist_df[bin_power_dist_df['Dclass']=='Continuous']
    inMapDf = bin_power_dist_df[bin_power_dist_df['NewVarName']=='bin_deposit_open_months_19']
    inKeyName = 'request_id'
    inTargetName = 'TargetBad'
    '''        
    var_ls = list(inMapDf['NewVarName'].unique())
    order_df = pd.DataFrame(inRawDf[inKeyName])
    for var_item in var_ls:
        print('Continuous rate map: ', var_item)
        #var_item = 'bin_self_deposit_rate_max_5'
        value_df = inMapDf[inMapDf['NewVarName'] == var_item][['Bins','Levels']]
        # 剔除空值分箱    
        if sum(value_df['Bins'] == 'nan'):
            value_df = value_df[value_df['Bins']!='nan']
        value_df = value_df.sort_values(by = 'Levels').reset_index(drop=True)
        
        #数据框转换为字典，值列表转换为区间
        value_dict = dict()
        #确定是否有缺失值归并
        value_df['Bins_nan'] = value_df['Bins'].apply(lambda x: 1 if type(x) is list else 0)
        if value_df['Bins_nan'].sum() > 0:
            value_dict['nan'] = value_df[value_df['Bins_nan']==1]['Levels'].values[0]
            #剔除缺失值进一步进行处理
            value_df['Bins'] = value_df['Bins'].apply(lambda y: [x for x in y if x!='nan'][0] if type(y) is list else y)
            for level in value_df['Levels'].tolist():
                value_dict[level] = value_df[value_df['Levels']==level]['Bins'].values[0]
        else:
            for level in value_df['Levels'].tolist():
                value_dict[level] = value_df[value_df['Levels']==level]['Bins'].values[0]
        
        
        # 针对非空值分箱进行赋值，空值分箱仍为空值  
        ls = var_item.split('_')
        ls.pop(0)
        ls.pop(-1)
        var_name = '_'.join(ls)  
        var_bin = inRawDf[var_name].map(lambda x: _cont_bin_value_map(x, valueDict=value_dict))
        var_bin.name = var_item
        order_df = order_df.merge(var_bin, left_index=True, right_index=True, how='left')
    if len(inTargetName) > 1:
        order_df = order_df.merge(inRawDf[inTargetName], left_index=True, right_index=True, how='left')
    
    print("***连续变量生成分箱变量完成！")
    return order_df
    


def rate_bin_transfer(inRawDf, inMapDf, keyVar, varKeepList):
    '''
    Function Descriptions:
        按照变量分箱的标准，对原始数据进行分箱，生成新的分箱变量。具体是按照二值变量、名义变量、有序变量和连续变量四类变量分别进行
    变量分箱转换，生成分箱变量，最后合并为一个数据框。
    
    Parameters
    ----------
    inRawDf      : 需要进行分箱的数据框
    inMapDf      : 分箱标准数据框
    keyVar       : 主键变量
    varKeepList  : 保留的变量列表
        
    Returns
    -------
    分箱变量数据框
        
    Examples
    --------
    inRawDf = model_raw_df
    inMapDf = bin_power_dist_df
    keyVar = var_key
    varKeepList = [var_target, 'SampleType']
    
    '''
    
    ## 名义变量入模变量新数据集
    nominal_bin_df = nominal_rate_bin_transfer(inRawDf, 
                                               inMapDf = inMapDf[inMapDf['Dclass']=='Nominal'],
                                               inKeyName = keyVar,
                                               inTargetName = '')        
    ## 有序变量入模变量新数据集
    order_bin_df = order_rate_bin_transfer(inRawDf, 
                                           inMapDf = inMapDf[inMapDf['Dclass']=='Order'], 
                                           inKeyName = keyVar,
                                           inTargetName = '')
    ## 连续变量入模变量新数据集
    continuous_bin_df = con_rate_bin_transfer(inRawDf,
                                              inMapDf = inMapDf[inMapDf['Dclass']=='Continuous'],
                                              inKeyName = keyVar,
                                              inTargetName = '')
    ## 二值变量入模变量新数据集
    binary_bin_df = inRawDf[inMapDf[inMapDf['Dclass']=='Binary']['NewVarName'].unique().tolist()+[keyVar]]
    
    ## 四类变量合并为入模变量数据集
    bin_df = nominal_bin_df.merge(order_bin_df, on=keyVar, how='left')
    bin_df = bin_df.merge(continuous_bin_df, on=keyVar, how='left')
    bin_df = bin_df.merge(binary_bin_df, on=keyVar, how='left')
    bin_df = bin_df.merge(inRawDf[[keyVar] + varKeepList], on=[keyVar])
    
    return bin_df
    








