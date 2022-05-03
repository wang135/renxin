# -*- coding: utf-8 -*-

from pandas import DataFrame

def df_unique_value_cnt(inDf):
    '''
    Function Description:
        计算数据框中的每个变量去重后的值数量
    
    Parameter
    ----------
    inDf : 数据框
    
    Returns
    -------
    所有变量去重值数量的数据框
    
    Examples
    --------
    df = pd.DataFrame(np.arange(0,60,2).reshape(10,3), columns=list('abc'))
    df_unique_value_cnt(inDf = df)
    '''
    NameList = list(inDf.columns)
    VarValueCntDict = dict()
    for i in range(len(NameList)): 
        VarValueCntDict.update({NameList[i]:inDf[NameList[i]].unique().size})    
    VarValueCntDat = DataFrame.from_dict(VarValueCntDict,orient='index').rename(columns={0:'VarValueCnt'})
    
    return VarValueCntDat   
    




