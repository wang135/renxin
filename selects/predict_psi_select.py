# -*- coding: utf-8 -*-

import pandas as pd
import math

def var_predict_psi(inDf, xVarName, yVarName, inTypeVar):
    '''
    Function Descriptions:
        计算单指标的预测能力稳定性，即：计算单变量各分箱逾期率的群体稳定性指标PSI
    
    Parameters
    ----------
    inDf      : 数据框
    xVarName  : 自变量名称
    yVarName  : 因变量名称
    inTypeVar : 区分INS和OOT的变量名称
    
    Returns
    -------
    自变量预测能力指标稳定性（PSI）结果的数据框
    
    Examples
    --------
    inDf = eval_bin_df
    xVarName = 'bin_ages_8'
    yVarName = 'TargetBad'
    inTypeVar = 'SampleType'
    var_predict_psi(inDf, xVarName, yVarName, inTypeVar)
    '''
            
    ins_df = inDf[inDf[inTypeVar]=='INS']
    oot_df = inDf[inDf[inTypeVar]=='OOT']
    ## 计算INS分箱逾期率
    var_psi_df = pd.DataFrame(ins_df.groupby(ins_df[xVarName].astype('str'))[yVarName].mean())
    var_psi_df = var_psi_df.rename(columns={yVarName:'ins_bad_rate'}) 
    ## 计算OOT分箱逾期率
    var_psi_df = var_psi_df.merge(oot_df.groupby(oot_df[xVarName].astype('str'))[yVarName].mean(), 
                                  left_index=True, right_index=True, how='outer')
    var_psi_df = var_psi_df.rename(columns={yVarName:'oot_bad_rate'})    
    var_consistency_df = var_psi_df.reset_index(drop=False).copy()
    
    ## PSI计算
    var_psi_df['ins_dist'] = round(var_psi_df['ins_bad_rate']/var_psi_df['ins_bad_rate'].mean(),4)
    var_psi_df['oot_dist'] = round(var_psi_df['oot_bad_rate']/var_psi_df['oot_bad_rate'].mean(),4)
    
    var_psi_df['dist_dif'] = var_psi_df['oot_dist'] - var_psi_df['ins_dist']
    var_psi_df['dist_log'] = (var_psi_df['oot_dist']/var_psi_df['ins_dist']).map(lambda x: round(math.log(x),6) if x>0 else round(math.log(0.000001),6))
    
    var_psi_df['psi_item'] = round(var_psi_df['dist_dif']*var_psi_df['dist_log'],6)
    var_psi_df['var_psi'] = round(var_psi_df['psi_item'].sum(),4)
    
    ## 逾期率一致性判断
    var_consistency_df = var_consistency_df.reset_index(drop=False)
    var_consistency_df['index_lag'] = var_consistency_df['index']-1
    var_csy_df = var_consistency_df.merge(var_consistency_df, left_on='index', right_on='index_lag', how='left')
    var_csy_df = var_csy_df[var_csy_df['index_lag_y'].notnull()]
    var_csy_ser = (var_csy_df['ins_bad_rate_x'] - var_csy_df['ins_bad_rate_y']) * (var_csy_df['oot_bad_rate_x'] - var_csy_df['oot_bad_rate_y']) > 0
    consistency_tag = (var_csy_ser.sum()==len(var_csy_ser))
    ## 确保INS和OOT的单调性一致
    var_psi_df['var_consistency'] = consistency_tag
    var_pred_psi_df = var_psi_df
    
    return var_pred_psi_df



def var_df_predict_psi(inDf, xVarList, yVarName, inTypeVar):
    '''
    Function Descriptions:
        计算数据框中自变量指标的预测能力稳定性，即：计算所有变量各分箱逾期率的群体稳定性指标PSI
    
    Parameters
    ----------
    inDf      : 数据框
    xVarList  : 自变量列表
    yVarName  : 因变量名称
    inTypeVar : 区分INS和OOT的变量名称
    
    Returns
    -------
    全部自变量预测能力指标稳定性（PSI）结果的数据框
    
    Examples
    --------
    inDf = eval_bin_df
    yVarName = 'TargetBad'
    inTypeVar = 'SampleType'
    exclude_var_ls = ['request_id', 'TargetBad', 'SampleType', 'PassDate']
    '''
    
    vars_psi_df = pd.DataFrame(columns=['VarName', 'ins_bad_rate', 'oot_bad_rate', 'ins_dist', 'oot_dist', 
                                        'dist_dif', 'dist_log', 'psi_item', 'var_psi', 'var_consistency'])
    
    for var_name in xVarList: 
        var_psi_df = var_predict_psi(inDf, xVarName=var_name, yVarName=yVarName, inTypeVar=inTypeVar)
        var_psi_df['VarName'] = var_name
        vars_psi_df = pd.concat([vars_psi_df, var_psi_df], axis=0)
    
    vars_psi_df = vars_psi_df.reset_index(drop=False)
    vars_psi_df = vars_psi_df.rename(columns={'index':'Levels'})
    vars_psi_df = vars_psi_df[['VarName', 'Levels', 'ins_bad_rate', 'oot_bad_rate', 'ins_dist', 
                               'oot_dist', 'dist_dif', 'dist_log', 'psi_item', 'var_psi', 'var_consistency']]
    vars_psi_df = vars_psi_df.sort_values(by = ['var_psi','VarName','Levels'])
    
    return vars_psi_df


def var_predict_psi_select(inDf, psiPoint, consistencyTag = True):
    '''
    inDf = var_psi_df
    psiPoint = 0.06
    consistencyTag = True
    '''
    low_psi_df = inDf[(inDf['var_psi'] < psiPoint) & (inDf['var_consistency']==consistencyTag)]
    
    return low_psi_df
    
    


