# -*- coding: utf-8 -*-

from pandas import DataFrame
import numpy as np
from base.df_unique_value_cnt import df_unique_value_cnt


def _var_chartype(x, y, z, keyVarList, TargetVarList, unScaleVarList):
    if z in keyVarList:
        return 'Key'
    elif z in TargetVarList:
        return 'Target'
    elif z in unScaleVarList:
        return 'UnScale'
    elif x == 2:
        return 'Date'
    elif (((x==1) and (y>100)) or y==1):
        return 'Droped' # 无效变量
    elif (x==1 and (y>2 or y<=100)):
        return 'Nominal' #名义变量
    elif (x in [0,1] and y==2):
        return 'Binary' #分类变量
    elif (x==0 and y<=20):
        return 'Order'  #有序
    elif (x==0 and y>20):
        return 'Continuous'## 连续
 
def variable_char_type(inDf, keyVarList, TargetVarList, unScaleVarList):
    '''
    inDf = sample
    keyVarList = ['lend_request_id']
    TargetVarList = ['TargetBad']
    unScaleVarList=['core_lend_request_id','SubmitMth','PassMth','id','lend_customer_id']
    '''
    
    
    # 获取存储变量类型
    type_df = DataFrame(inDf.dtypes,columns=['Dtypes'])
    type_df['Dtypes'] = type_df['Dtypes'].astype('str')
    
    # 变量分为两类：字符型和数值型
    type_df['Tclass1'] = np.where(type_df['Dtypes'].astype('str').str.contains('datetime'),2,
                                  np.where(type_df['Dtypes'] == 'object',1,0))  # 1-字符型；0-数值型；2-时间类型
    
    # 增加变量的值个数
    type_df = type_df.join(df_unique_value_cnt(inDf))
    
    # 增加表的记录数
    type_df['N'] = len(inDf.index)
    
    # 增加缺失值记录数
    type_df['Nmiss'] = inDf.isnull().sum()
    
    # 使变量做为一列
    type_df = type_df.reset_index(drop=False)
    
    # 变量进一步进行分类：连续变量、二值变量、名义变量、有序变量、无效变量（分类值过多）
    type_df['Dclass'] = type_df.apply(lambda x: _var_chartype(x['Tclass1'], x['VarValueCnt'], x['index'], 
                                                              keyVarList, TargetVarList, unScaleVarList), 
                                      axis=1)
    type_df['NmissRate'] = round(type_df['Nmiss'] / type_df['N'],4)                                                 
    return type_df   




