# -*- coding: utf-8 -*-



import pandas as pd
from math import log


def score_calculate(pdo, odds, oddsScore, woeDf, coefDf):
    '''
    Funcation Descriptions:
        根据WOE、PDO、ODDS和OddsScore，把变量WOE转换为评分
    
    Parameters
    ----------
    pdo       : 增加一倍优势比，增加的评分分数
    odds      : 参照优势比
    oddsScore : 参照优势比对应的评分
    woeDf     : WOE汇总数据框
    coefDf    : 参数数据框
    
    Returns
    -------
    变量分箱对应评分的数据框
    
    Examples
    --------
    pdo = 40
    odds = 0.036
    oddsScore = 600
    woeDf = model_woe_freq_df
    coefDf = model_coef_df
    score_calculate(pdo, odds, oddsScore, woeDf, coefDf)
    '''
    B = round(pdo/log(2),4)
    A = oddsScore + round(B*log(odds),2)
    
    score_df = pd.DataFrame(columns=['VarName','Level','WOE','Beta','Score'])
    var_ls = coefDf['VarName'].tolist()
    
    for var_name in var_ls: 
        #var_name = 'woe_Rbin_busi_loan_cnt_7'
        var_woe_df = woeDf[woeDf['NewVarName'] == var_name.replace('woe_','')]
        Beta = coefDf[coefDf['VarName'] == var_name]['Coefficient'].tolist()[0]
        if var_name.lower() == 'intercept':
            level_score = round(A-B*Beta)
            score_df = pd.concat([score_df,
                                  pd.DataFrame([['BasePoints','--','--',Beta,level_score]], 
                                               columns=('VarName','Level','WOE','Beta','Score'))
                                  ], axis=0) 
        else :
            var_woe_df['Score'] = -round(var_woe_df['WOE_adjust']*Beta*B)
            var_score_df = var_woe_df[['NewVarName','Bins','WOE_adjust','Score']].rename(columns={'NewVarName':'VarName','Bins':'Level','WOE_adjust':'WOE'})
            var_score_df['VarName'] = 'woe_'+var_score_df['VarName']
            var_score_df['Beta'] = Beta
            score_df = pd.concat([score_df, var_score_df], axis=0)
                
    score_df['PDO'] = pdo
    score_df['Odds'] = odds
    score_df['OddsScore'] = oddsScore
    score_df['A'] = A
    score_df['B'] = B
    score_df = score_df[['VarName', 'Level', 'WOE', 'Beta', 'Score']]
    
    return score_df


def _score_map(x, scoreDict):
    '''
    x = 0.3772389229364818
    scoreDict = score_dict
    '''
    for woe_item in scoreDict.keys():
        #if round(x,6) == round(woe_item,6):
        if x == woe_item:
            return scoreDict[woe_item]


def scorecards(inDf, inScoreDf, keepVarLs):
    '''
    Funcation Descriptions:
        给每个样本的所有指标进行打分，并计算出综合评分
    
    Parameters
    ----------
    inDf      : 模型WOE宽表数据框
    inScoreDf : 评分汇总表数据框
    keepVarLs : 需要保留的变量列表
    
    Returns
    -------
    每个样本及所有指标的评分和最终评分的数据框
    
    Examples
    --------
    inDf = model_woe_df
    inScoreDf = score_df
    keepVarLs = ['request_id','TargetBad']
    '''
    
    scorecard_df = inDf[keepVarLs]
    var_ls = inScoreDf['VarName'].unique().tolist()
    for var_item in var_ls:
        print('ScoreCard: ', var_item)
        if var_item == 'BasePoints':
            scorecard_df[var_item] = inScoreDf[inScoreDf['VarName']=='BasePoints']['Score'].tolist()[0]
        else :
            var_score_df = inScoreDf[inScoreDf['VarName']==var_item]             
            score_dict = var_score_df.set_index('WOE')['Score'].to_dict()            
            scorecard_df[var_item] = inDf[var_item].map(lambda x: _score_map(x=x, scoreDict=score_dict))
    
    scorecard_df['score'] = scorecard_df[var_ls].sum(axis=1)
    
    return scorecard_df
    



