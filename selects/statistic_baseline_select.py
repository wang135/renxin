# -*- coding: utf-8 -*-


from base import equal_freq_cut

def drop_variables_by_class(inVarClassDf, toDropVarClassList):
    '''
    Function Description：
        按照变量的类型确定需要剔除的变量 
    
    Examples
    --------
    inVarClassDf = var_class_df
    toDropVarClassList = ['UnScale','Date']
    '''       
    TmpDropVarDf = inVarClassDf[inVarClassDf['Dclass'].isin(toDropVarClassList)]
    ToDropVarList = TmpDropVarDf['index'].tolist()
    
    return ToDropVarList


def drop_variables_by_unique_value(inVarClassDf):
    '''
    Function Description：
        确定剔除单一值变量
    '''
    TmpDropVarDf = inVarClassDf[inVarClassDf['VarValueCnt'] == 1]
    UniqueValueVarList = TmpDropVarDf['index'].tolist()
    
    return UniqueValueVarList


def drop_variables_by_missing_value(inVarClassDf, dropMissRatePoint):
    '''
    Function Description：
        按照缺失值比例确定需要剔除的变量
    '''      
    inVarClassDf['NmissRate'] = round(inVarClassDf['Nmiss'] / inVarClassDf['N'],4)
    TmpDropMissVarDf = inVarClassDf[inVarClassDf['NmissRate'] >= dropMissRatePoint]
    HighMissValueList = TmpDropMissVarDf['index'].tolist()
    
    return HighMissValueList


def drop_cont_variables_by_overcenter_value(inDf, conVarLst, equalBinNum=10):
    '''
    Function Description：
        获取连续变量无法进行等频分箱的变量列表
        
    Examples
    --------
    inDf = ins_raw_df
    conVarLst = ins_raw_var_class_df[ins_raw_var_class_df['Dclass']=='Continuous']['index'].tolist()
    equalBinNum = 10
    '''      
    OverCenterList = list()
    for xVarName in conVarLst:
        #等频分箱,生成数据框
        qcut_ser = equal_freq_cut(inDf[xVarName], equalBinNum)
        if len(qcut_ser)<=1:
            OverCenterList.append(xVarName)
    
    return OverCenterList

def drop_class_variables_by_overcenter_value(inVarDistDf, dropOverCenterPoint=0.95):
    '''
    Function Description：
        按照分类变量值的样本占比，剔除过于集中的变量
    
    Parameters
    ----------
    inVarDistDf : 变量频数分布数据框
    dropOverCenterPoint ： 样本量占比的分割点
    ''' 
    value_center_ser = inVarDistDf.groupby('VarName')['ValueRate'].max()
    OverCenterList = value_center_ser[value_center_ser.map(lambda x: float(x.replace('%',''))/100>dropOverCenterPoint)].index.tolist()

    return OverCenterList






