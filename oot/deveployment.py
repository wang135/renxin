# -*- coding: utf-8 -*-

import math

def lr_formula_deployment(inWoeDf, inCoefDf):
    '''
    inWoeDf = oot_woe_df
    inCoefDf = model_coef_df
    '''
    prob_df = inWoeDf.copy()
    xvar_ls = inCoefDf['VarName'].tolist()
    z_tmp = 0
    for x_var in xvar_ls:
        if x_var.lower() == 'intercept':
            z_tmp = z_tmp + inCoefDf[inCoefDf['VarName']=='Intercept']['Coefficient'].reset_index(drop=True)[0]
        else :
            z_tmp = z_tmp + prob_df[x_var] * inCoefDf[inCoefDf['VarName']==x_var]['Coefficient'].reset_index(drop=True)[0]

    prob_df['y_pred'] = z_tmp.map(lambda x: round(1- 1/(1+math.exp(x)),12))
    
    return prob_df


def predict_compare(insDf, evalDf, keyVarName, SampleTypeVarName, insValue):
    '''
    insDf = model_pred_df
    evalDf = eval_pred_df
    keyVarName = 'request_id'
    SampleTypeVarName = 'SampleType'
    insValue = 'INS'
    '''
    tmp_pred = evalDf[evalDf[SampleTypeVarName] == insValue]
    tmp_pred['eval_y_pred'] = tmp_pred['y_pred']
    pred_compare_df = tmp_pred[[keyVarName,'eval_y_pred']].merge(insDf[[keyVarName,'y_pred']], 
                              on=[keyVarName])
    prob_wrong_cnt = round((pred_compare_df['eval_y_pred'] - pred_compare_df['y_pred']),5).sum()
    
    print("***************************************************************************")
    print('部署预测值与模型预测值不一致的样本量：', prob_wrong_cnt)
    if tmp_pred.shape[0] == insDf.shape[0]:
        print ("部署的INS样本量与建模的INS样本量相等，样本量为：", insDf.shape[0])
    else:
        print ("部署的INS样本量与建模的INS样本量不相等")










