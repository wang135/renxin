
# -*- coding: utf-8 -*-

import pandas as pd
import copy
from intervals import FloatInterval



def _binary_woe_map(x, valueDf):
    '''
    名义变量值与分箱值对照
    x = ins_clean_df['gender'][15055]
    valueDf = value_df    
    '''
    valueList = valueDf['Bins'].tolist()
    
    for item in valueList:
        if str(x) in str(item).split(','):
            return valueDf[valueDf['Bins']==item]['WOE_adjust'].tolist()[0]

def binary_woe_transfer(inRawDf, inMapDf, inKeyName, inTargetName):
    '''
    Function Descriptions:
        使用新分箱的频数数据集，对原始数据集进行重新分箱
        
    Parameters
    ----------
    inRawDf      : 原始数据框
    inMapDf      : WOE数据框
    inKeyName    : 数据框中的主键变量
    inTargetName : 数据框中的目标变量
    
    Returns
    -------
    数据框： 重新分箱后的数据框
    
    Examples
    --------
    inRawDf = ins_corred_df
    inMapDf = model_woe_stat[model_woe_stat['Dclass']=='Binary']
    inKeyName = 'request_id'
    inTargetName = 'TargetBad'
    binary_woe_transfer(inRawDf, inMapDf, inKeyName, inTargetName)
    '''        
    var_ls = list(inMapDf['VarName'].unique())
    binary_df = pd.DataFrame(inRawDf[inKeyName])
    for var_item in var_ls:
        value_df = inMapDf[inMapDf['VarName'] == var_item]
        value_df = value_df.reset_index(drop=True)
        var_bin = inRawDf[var_item].map(lambda x: _binary_woe_map(x, valueDf=value_df))
        var_bin.name = 'woe_{}'.format(var_item)
        print(var_bin.head(5))
        binary_df = binary_df.merge(var_bin, left_index=True, right_index=True, how='left')
    if len(inTargetName) > 1:
        binary_df = binary_df.merge(inRawDf[inTargetName], left_index=True, right_index=True, how='left')
    
    print('***二值变量生成WOE变量完成！')
    return binary_df




def _nom_woe_map(x, binDict, woeDict):
    '''
    名义变量值与分箱值对照
    x = 'yidong'
    binDict = bin_dict    
    woeDict = woe_dict
    '''
    for idx in binDict.keys():
        if str(x) in str(binDict[idx]).split(','):
            return woeDict[idx]        
        
def nominal_woe_transfer(inRawDf, inMapDf, inKeyName, inTargetName):
    '''
    Function Descriptions:
        使用新分箱的频数数据集，对原始数据集进行重新分箱
        
    Parameters
    ----------
    inRawDf      : 原始数据框
    inMapDf      : WOE数据框
    inKeyName    : 数据框中的主键变量
    inTargetName : 数据框中的目标变量
    
    Returns
    -------
    数据框： 重新分箱后的数据框
    
    Examples
    --------
    inRawDf = model_table
    inMapDf = woe_stat
    inKeyName = 'request_id'
    inTargetName = 'TargetBad'
    '''        
    var_ls = list(inMapDf['NewVarName'].unique())
    nominal_df = pd.DataFrame(inRawDf[inKeyName])
    for var_item in var_ls:
        print(var_item)
        value_df = inMapDf[inMapDf['NewVarName'] == var_item].reset_index(drop=True)
        ls = var_item.split('_')
        ls.pop(0)
        ls.pop(-1)
        var_name = '_'.join(ls)
        bin_dict = value_df['Bins'].to_dict()
        woe_dict = value_df['WOE_adjust'].to_dict()
        var_bin = inRawDf[var_name].map(lambda x: _nom_woe_map(x, binDict=bin_dict, woeDict=woe_dict))
        var_bin.name = 'woe_{}'.format(var_item)
        print(var_bin.head(5))
        nominal_df = nominal_df.merge(var_bin, left_index=True, right_index=True, how='left')
    if len(inTargetName) > 1:
        nominal_df = nominal_df.merge(inRawDf[inTargetName], left_index=True, right_index=True, how='left')
    
    print('***名义变量生成WOE变量完成！')
    return nominal_df




def _order_woe_value_map(x, binDict, woeDict):
    '''
    x = 'nan'
    binDict = bin_dict
    woeDict = woe_dict
    '''

    del_nan_lst = list(binDict.keys())
    if 'nan' in del_nan_lst:
        del_nan_lst.remove('nan')
    for level in binDict.keys():
        if level == 'nan':
            if pd.isnull(x):
                return woeDict['nan']
        elif level == min(del_nan_lst) :
            if x <= binDict[level].upper:
                return woeDict[level]
        elif level == max(del_nan_lst) :
            if x > binDict[level].lower:
                return woeDict[level]
        elif x > binDict[level].lower and x <= binDict[level].upper:
            return woeDict[level]
        
                
def order_woe_transfer(inRawDf, inMapDf, inKeyName, inTargetName):
    '''
    Function Descriptions:
        频数分箱后，生成新的分箱数据集
    
    Parameters
    ----------
    inRawDf      : 需要进行分箱的数据框
    inMapDf      : WOE数据框
    inKeyName    : inRawDf的主键变量
    inTargetName : inRawDf的目标变量
        
    Returns
    -------
    有序变量重新分箱后的数据框
        
    Examples
    --------
    inRawDf = ins_corred_df
    inMapDf = model_woe_stat[model_woe_stat['Dclass']=='Order']
    inKeyName = 'request_id'
    inTargetName = 'TargetBad'
    order_woe_transfer(inRawDf, inMapDf, inKeyName, inTargetName)
    '''        
    var_ls = list(inMapDf['NewVarName'].unique())
    order_df = pd.DataFrame(inRawDf[inKeyName])
    for var_item in var_ls:
        print(var_item)
        value_df = inMapDf[inMapDf['NewVarName'] == var_item][['Bins','Levels','WOE_adjust']].copy() 
        value_df = value_df.reset_index(drop=True)
        
        #分别把woe和bin转换为字典，且bin由列表转换为区间
        bin_dict = dict()
        woe_dict = dict()
        #处理单独的缺失值区间
        if sum(value_df['Bins'] == 'nan') > 0:
            bin_dict['nan'] = value_df[value_df['Bins']=='nan']['Bins'].values[0]
            woe_dict['nan'] = value_df[value_df['Bins']=='nan']['WOE_adjust'].values[0]
            value_df = value_df[value_df['Bins']!='nan']
        
        #确定是否有缺失值归并
        value_df['Bins_nan'] = value_df['Bins'].apply(lambda x: 1 if 'nan' in x.split(', ') else 0)
        if value_df['Bins_nan'].sum() > 0:
            bin_dict['nan'] = value_df[value_df['Bins_nan']==1]['Bins'].values[0]
            woe_dict['nan'] = value_df[value_df['Bins_nan']==1]['WOE_adjust'].values[0]
            #剔除缺失值进一步进行处理
            value_df['Bins'] = value_df['Bins'].apply(lambda x: x.replace(', nan',''))
            for level in value_df['Levels'].tolist():
                if level == min(value_df['Levels'].tolist()):
                    bin_dict[level] = FloatInterval.open_closed(float('-inf'),value_df[value_df['Levels']==level]['Bins'].values[0].split(',')[-1])
                    woe_dict[level] = value_df[value_df['Levels']==level]['WOE_adjust'].values[0]
                elif level == max(value_df['Levels'].tolist()):
                    bin_dict[level] = FloatInterval.open(value_df[value_df['Levels']==level-1]['Bins'].values[0].split(',')[-1], float('inf'))
                    woe_dict[level] = value_df[value_df['Levels']==level]['WOE_adjust'].values[0]
                else :
                    bin_dict[level] = FloatInterval.open_closed(value_df[value_df['Levels']==level-1]['Bins'].values[0].split(',')[-1], value_df[value_df['Levels']==level]['Bins'].values[0].split(',')[-1])
                    woe_dict[level] = value_df[value_df['Levels']==level]['WOE_adjust'].values[0]
        else:
            for level in value_df['Levels'].tolist():
                if level == min(value_df['Levels'].tolist()):
                    bin_dict[level] = FloatInterval.open_closed(float('-inf'),value_df[value_df['Levels']==level]['Bins'].values[0].split(',')[-1])
                    woe_dict[level] = value_df[value_df['Levels']==level]['WOE_adjust'].values[0]
                elif level == max(value_df['Levels'].tolist()):
                    bin_dict[level] = FloatInterval.open(value_df[value_df['Levels']==level-1]['Bins'].values[0].split(',')[-1], float('inf'))
                    woe_dict[level] = value_df[value_df['Levels']==level]['WOE_adjust'].values[0]
                else :
                    bin_dict[level] = FloatInterval.open_closed(value_df[value_df['Levels']==level-1]['Bins'].values[0].split(',')[-1], value_df[value_df['Levels']==level]['Bins'].values[0].split(',')[-1])
                    woe_dict[level] = value_df[value_df['Levels']==level]['WOE_adjust'].values[0]
        
        ls = var_item.split('_')
        ls.pop(0)
        ls.pop(-1)
        var_name = '_'.join(ls)  
        var_bin = inRawDf[var_name].map(lambda x: _order_woe_value_map(x, binDict=bin_dict, woeDict=woe_dict))
        var_bin.name = 'woe_{}'.format(var_item)
        print(var_bin.head(5))
        order_df = order_df.merge(var_bin, left_index=True, right_index=True, how='left')
    if len(inTargetName) > 1:
        order_df = order_df.merge(inRawDf[inTargetName], left_index=True, right_index=True, how='left')
    
    print("***有序变量生成WOE变量完成！")
    return order_df
    






def _cont_woe_value_map(x, binDict, woeDict):
    '''
    x='nan'
    binDict = bin_dict
    woeDict = woe_dict
    _cont_bin_value_map(x, valueDf)
    '''
    
    del_nan_lst = list(binDict.keys())
    if 'nan' in del_nan_lst:
        del_nan_lst.remove('nan')
    for level in binDict.keys():
        if level == 'nan':
            if pd.isnull(x):
                return woeDict['nan']
        elif level == min(del_nan_lst) :
            if x <= binDict[level].upper:
                return woeDict[level]
        elif level == max(del_nan_lst) :
            if x > binDict[level].lower:
                return woeDict[level]
        elif x > binDict[level].lower and x <= binDict[level].upper:
            return woeDict[level]

                
def con_woe_transfer(inRawDf, inMapDf, inKeyName, inTargetName):
    '''
    Function Descriptions:
        频数分箱后，生成新的分箱数据集
    
    Parameters
    ----------
    inRawDf      : 需要进行分箱的数据框
    inMapDf      : WOE数据框
    inKeyName    : inRawDf的主键变量
    inTargetName : inRawDf的目标变量
        
    Returns
    -------
    有序变量重新分箱后的数据框
        
    Examples
    --------
    inRawDf = ins_corred_df
    inMapDf = model_woe_stat[model_woe_stat['Dclass']=='Continuous']
    inKeyName = 'request_id'
    inTargetName = 'TargetBad'
    con_woe_transfer(inRawDf, inMapDf, inKeyName, inTargetName)
    '''        
    var_ls = list(inMapDf['NewVarName'].unique())
    order_df = pd.DataFrame(inRawDf[inKeyName])
    for var_item in var_ls:
        print(var_item)
        #var_item = 'bin_self_deposit_rate_max_5'
        value_df = inMapDf[inMapDf['NewVarName'] == var_item][['Bins','Levels', 'WOE_adjust']].copy()
        
        
        #分别把woe和bin转换为字典，且bin由列表转换为区间
        bin_dict = dict()
        woe_dict = dict()
        #处理单独的缺失值区间
        if sum(value_df['Bins'] == 'nan') > 0:
            bin_dict['nan'] = value_df[value_df['Bins']=='nan']['Bins'].values[0]
            woe_dict['nan'] = value_df[value_df['Bins']=='nan']['WOE_adjust'].values[0]
            value_df = value_df[value_df['Bins']!='nan']
        
        #确定是否有缺失值归并        
        value_df['Bins_nan'] = value_df['Bins'].apply(lambda x: 1 if type(x) is list else 0)
        if value_df['Bins_nan'].sum() > 0:
            bin_dict['nan'] = value_df[value_df['Bins_nan']==1]['Bins'].values[0]
            woe_dict['nan'] = value_df[value_df['Bins_nan']==1]['WOE_adjust'].values[0]
            #剔除缺失值进一步进行处理
            value_df['Bins'] = value_df['Bins'].apply(lambda y: [x for x in y if x!='nan'][0] if type(y) is list else y)
            for level in value_df['Levels'].tolist():
                bin_dict[level] = value_df[value_df['Levels']==level]['Bins'].values[0]
                woe_dict[level] = value_df[value_df['Levels']==level]['WOE_adjust'].values[0]
        else:
            for level in value_df['Levels'].tolist():
                bin_dict[level] = value_df[value_df['Levels']==level]['Bins'].values[0]
                woe_dict[level] = value_df[value_df['Levels']==level]['WOE_adjust'].values[0]
        
        # 针对非空值分箱进行赋值，空值分箱仍为空值  
        ls = var_item.split('_')
        ls.pop(0)
        ls.pop(-1)
        var_name = '_'.join(ls)  
        var_bin = inRawDf[var_name].map(lambda x: _cont_woe_value_map(x, binDict=bin_dict, woeDict=woe_dict))
        var_bin.name = 'woe_{}'.format(var_item)
        print(var_bin.head(5))
        order_df = order_df.merge(var_bin, left_index=True, right_index=True, how='left')
    if len(inTargetName) > 1:
        order_df = order_df.merge(inRawDf[inTargetName], left_index=True, right_index=True, how='left')
    
    print("***连续变量生成WOE变量完成！")
    return order_df
    


def woe_transfer(inRawDf, inMapDf, keyVar, targetVar, keepVarLst=[]):
    '''
    Function Descriptions:
        根据分箱的规则，把每个样本的原始值转换为woe值。具体执行过程是分别按照二值变量、名义变量、有序变量和连续变量
    分别进行woe转换后生成woe变量，然后把全部woe转换合并为一个woe建模宽表。
    
    Parameters
    ----------
    inRawDf      : 原始变量数据框
    inMapDf      : WOE数据框
    inKeyName    : inRawDf的主键变量
    inTargetName : inRawDf的目标变量
        
    Returns
    -------
    woe数据框
        
    Examples
    --------
    inRawDf = sample_df
    inMapDf = df_desc
    keyVar = 'request_id'
    targetVar = 'TargetBad'
    keepVarLst = []
    woe_transfer(inRawDf, inMapDf, keyVar, targetVar, keepVarLst)
    '''
    woe_binary_df = binary_woe_transfer(inRawDf,
                                        inMapDf = inMapDf[inMapDf['Dclass']=='Binary'],
                                        inKeyName = keyVar,
                                        inTargetName = '')
    woe_nom_df = nominal_woe_transfer(inRawDf,
                                      inMapDf = inMapDf[inMapDf['Dclass']=='Nominal'],
                                      inKeyName = keyVar,
                                      inTargetName = '')
    woe_ord_df = order_woe_transfer(inRawDf,
                                    inMapDf = inMapDf[inMapDf['Dclass']=='Order'],
                                    inKeyName = keyVar,
                                    inTargetName = '')
    woe_con_df = con_woe_transfer(inRawDf,
                                  inMapDf = inMapDf[inMapDf['Dclass']=='Continuous'],
                                  inKeyName = keyVar,
                                  inTargetName = targetVar)
    woe_df = pd.merge(woe_binary_df, woe_nom_df, on=keyVar)
    woe_df = woe_df.merge(woe_ord_df, on=keyVar)
    woe_df = woe_df.merge(woe_con_df, on=keyVar)
    woe_df = woe_df.merge(inRawDf[[keyVar] + keepVarLst], on=keyVar)

    return woe_df


    
  
def _standard_woe_map(x, binDict, woeDict):
    '''
    名义变量值与分箱值对照
    x = 'yidong'
    binDict = bin_dict    
    woeDict = woe_dict
    '''
    for idx in binDict.keys():
        if str(x) in str(binDict[idx]).split(','):
            return woeDict[idx]        
        
def standard_woe_transfer(inRawDf, inMapDf, inKeyName, inTargetName):
    '''
    Function Descriptions:
        使用新分箱的频数数据集，对原始数据集进行重新分箱
        
    Parameters
    ----------
    inRawDf      : 原始数据框
    inMapDf      : WOE数据框
    inKeyName    : 数据框中的主键变量
    inTargetName : 数据框中的目标变量
    
    Returns
    -------
    数据框： 重新分箱后的数据框
    
    Examples
    --------
    inRawDf = model_table
    inMapDf = woe_stat
    inKeyName = 'request_id'
    inTargetName = 'TargetBad'
    '''        
    var_ls = list(inMapDf['NewVarName'].unique())
    nominal_df = pd.DataFrame(inRawDf[inKeyName])
    for var_item in var_ls:
        print(var_item)
        value_df = inMapDf[inMapDf['NewVarName'] == var_item].reset_index(drop=True)
        bin_dict = value_df['Bins'].to_dict()
        woe_dict = value_df['WOE_adjust'].to_dict()
        var_bin = inRawDf[var_item].map(lambda x: _standard_woe_map(x, binDict=bin_dict, woeDict=woe_dict))
        var_bin.name = 'woe_{}'.format(var_item)
        print(var_bin.head(5))
        nominal_df = nominal_df.merge(var_bin, left_index=True, right_index=True, how='left')
    if len(inTargetName) > 1:
        nominal_df = nominal_df.merge(inRawDf[inTargetName], left_index=True, right_index=True, how='left')
    
    print('***名义变量生成WOE变量完成！')
    return nominal_df

  
    
    
    



    
    
    
    
