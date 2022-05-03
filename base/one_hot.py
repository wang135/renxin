# -*- coding: utf-8 -*-

	
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder



def cols_one_hot(inDf, varLst, missValue = 'Miss', method = 'dummy'):
    '''
    Function Description:
        多个分类变量，进行one-hot转换
    
    Parameters
    ----------
    inDf      : 待处理数据框
    varLst    : 待处理变量列表
    missValue : 缺失值填充值
    method    : 进行one-hot的方法：onehot-OneHotEncoder、dummy-get_dummies，默认为dummy
    
    examples
    --------
    inDf = leaf_df
    varLst = ['leaf_0']
    missValue = 'Miss'
    method = 'onehot'
    
    Returns
    -------
    数据框：one-hot转换后的数据框
    
    '''
    # 缺失值填充
    trans_df = inDf[varLst].fillna(missValue).copy()
    # one-hot encoding转换
    if method == 'onehot':
        code_df = pd.DataFrame(columns = ['Code','Value','VarName'])
        for col in varLst:
            lbe = LabelEncoder()
            lbe.fit_transform(trans_df[col])   
            tmp_code_df = pd.DataFrame({'Code':    range(len(lbe.classes_)),
                                        'Value':   lbe.classes_,
                                        'VarName': col})
            code_df = pd.concat([code_df, tmp_code_df], axis=0)
        onehot = OneHotEncoder()
        onehot_df = pd.DataFrame(onehot.fit_transform(trans_df).toarray(),
                                 columns = [i.replace('.0','') for i in list(onehot.get_feature_names(varLst))], 
                                 dtype = 'int')
        # 剔除待转换的原始变量
        df = inDf.drop(varLst, axis=1).reset_index(drop=True) 
        df = pd.concat([df, onehot_df], axis=1)
        print(code_df)
    elif method == 'dummy':
        ## 把分类变量转换为哑变量
        onehot_df = pd.get_dummies(trans_df)
        df = inDf.drop(varLst, axis=1)
        df = df.merge(onehot_df, left_index=True, right_index=True)
        
    
    return df




def cols_to_label(inDf, varLst, missValue = 'Miss'):
    '''
    Descriptions:
        
    Parameters
    ----------
    inDf: 数据框
    varLst: 需要标签化的字段列表
    missValue: 缺失值填充值，默认为Miss
    
    Examples
    --------
    inDf = ins_clean_df
    varLst = catgory_list
    missValue = 'Miss'
    (inDf, varLst, missValue = 'Miss')
    
    Returns
    -------
    [label_df, code_df] 
    label_df-标签转换后的数据框
    code_df-标签与值的对应关系
    
    '''
    # 缺失值填充
    label_df = inDf.copy()
    trans_df = inDf[varLst].fillna(missValue).copy()
    
    code_df = pd.DataFrame(columns = ['Code','Value','VarName'])
    for col in varLst:
        lbe = LabelEncoder()
        label_df[col] = lbe.fit_transform(trans_df[col])   
        tmp_code_df = pd.DataFrame({'Code':    range(len(lbe.classes_)),
                                    'Value':   lbe.classes_,
                                    'VarName': col})
        code_df = pd.concat([code_df, tmp_code_df], axis=0)
    
    return [label_df, code_df]



def cols_been_label(inDf, varLabelDf, missValue = 'Miss'):
    '''
    Descriptions:
        利用已有的标签与值的对应关系，对新数据框的变量进行打标签.假如新变量有新值不在已有的值中，则标签赋予-99。
    
    Parameters
    ----------
    inDf: 待标签转换数据框
    varLabelDf: 标签与值对应关系数据框，字段名称为['Code', 'Value', 'VarName']
    missValue: 缺失值填充值，需与已有的值的填充方式一致
        
    Examples
    --------
    inDf = gbdt_sample_df[gbdt_x_list]
    varLabelDf = gbdt_label[1]
    missValue = 'Miss'
    cols_been_label(inDf, varLabelDf, missValue = 'Miss')
    
    Returns
    -------
    label_df: 标签转换后的新数据框
    
    '''
    label_df = inDf.copy()
    varLst = varLabelDf['VarName'].unique().tolist()
    trans_df = inDf[varLst].fillna(missValue).copy()
    
    for col in varLst:
        var_label_Df = varLabelDf.copy()
        var_label_Df['MissTag'] = var_label_Df['Value'].map(lambda x: 1 if x.lower().find(missValue.lower())==0 else 0)
        if var_label_Df['MissTag'].sum() > 0:
            NowValue = var_label_Df[var_label_Df['MissTag']==1]['Value'].values[0]
        else :
            NowValue = var_label_Df['Value'].values[0]
        
        var_label = varLabelDf[varLabelDf['VarName']==col]
        label_dict = var_label.set_index('Value')['Code'].to_dict()        
        label_df[col] = trans_df[col].apply(lambda x: x if x in list(label_dict.keys()) else NowValue)
        
        if trans_df[col].apply(lambda x: 0 if x in list(label_dict.keys()) else 1).sum()>0:
            print('{} has new values: {}'.format(col, trans_df[col].apply(lambda x: 0 if x in list(label_dict.keys()) else 1).sum()))
    
    return label_df
            
























    
    
    
