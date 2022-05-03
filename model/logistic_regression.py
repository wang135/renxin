# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import statsmodels.stats.api as sms
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif_func
from statsmodels.compat import lzip

from sklearn.linear_model import LogisticRegression

from scipy import stats

from analysis import iv_df_auto_calculate


'''
向前回归、向后回归、逐步回归三个筛选变量的方法选择变量进入模型的常用统计量有：
DJRSQ, AIC, AICC, BIC, CP, RSQUARE, SBC

逐步回归的步骤：
1、待入模变量逐个进入模型，生成模型的拟合效果统计量，选择拟合效果最好的变量；
2、计算新入选变量后的变量参数检验，针对新入选变量t检验p值(sle)和已入选变量t检验P值(sls)是否满足条件,确定
   新入选变量能否进入变量、已入选变量中的哪些需要被移除模型中。
3、不断迭代1和2
4、迭代停止的标准有两个：
   (1)无法提升模型的总体拟合效果； 
   (2)所有入选变量t检验P值均满足sls的设置标准，新入选变量的t检验p值不满足sle的设置标准。

'''



#定义向前逐步回归函数
def lr_forward_select(inDf, xVarNameLst, yVarName, fallStopPoint=0.0002):
    '''
    Funcation Descriptions:
        采用向前法选择变量
        
    Parameters
    ---------
    inDf       : 建模宽表数据框
    xVarNameLst : 自变量名称列表
    yVarName   : 因变量名称
    fallStopPoint : AIC值下降的相对值，当下降值小于该值则停止增加新变量
    
    Returns
    -------
    每步变量组合的AIC值数据框 
    
    Examples
    --------
    inDf = model_woe_df
    xVarNameLst = list(filter(lambda x: x not in ['request_id','TargetBad'], model_woe_df.columns.tolist()))
    yVarName = 'TargetBad'
    lr_forward_select(inDf, xVarNameLst, yVarName)
    '''
    variate=set(xVarNameLst)  #将字段名转换成字典类型
    selected=[]
    current_score,best_new_score=float('inf'),float('inf')  #目前的分数和最好分数初始值都为无穷大（因为AIC越小越好）
    aic_step = []
    #rsquare_step = []
    
    #循环筛选变量
    while variate:
        aic_with_variate=[]
        for candidate in variate:  #逐个遍历自变量
            formula="{}~{}".format(yVarName,"+".join(selected+[candidate]))  #将自变量名连接起来
            aic=ols(formula=formula,data=inDf).fit().aic  #利用ols训练模型得出aic值
            aic_with_variate.append((aic,candidate))  #将第每一次的aic值放进空列表
        aic_with_variate.sort(reverse=True)  #降序排序aic值
        best_new_score,best_candidate=aic_with_variate.pop()  #最好的aic值等于删除列表的最后一个值，以及最好的自变量等于列表最后一个自变量
        fall_rate = round((current_score - best_new_score) / best_new_score,5)
        #if (abs(fall_rate) >= 0.0002) & (best_new_score >= -1000):  #如果目前的aic值大于最好的aic值
        
        if (abs(fall_rate) >= fallStopPoint):  #如果目前的aic值大于最好的aic值
            variate.remove(best_candidate)  #移除加进来的变量名，即第二次循环时，不考虑此自变量了
            selected.append(best_candidate)  #将此自变量作为加进模型中的自变量
            current_score=best_new_score  #最新的分数等于最好的分数
            aic_step.append([current_score, 
                             fall_rate, 
                             ','.join(selected)])
            print("aic is {},continuing!".format(current_score))  #输出最小的aic值
        else:
            print("for selection over!")
            break
    aic_step_df = pd.DataFrame(aic_step, columns=['AIC', 'AIC_Fall', 'xVarNameList'])
    
    matplotlib.use('Qt5Agg')
    plt.title("AIC下降趋势线")
    plt.plot(aic_step_df['AIC'])
    
    return(aic_step_df)
    


def lr_stepwise_select(inDf, xVarNameLst, yVarName, sle = 0.05, sls = 0.05, fallStopPoint=0.00001):
    '''
    Funcation Descriptions:
        逐步回归法
        
    Parameters
    ---------
    inDf       : 建模宽表数据框
    xVarNameLst : 自变量名称列表
    yVarName   : 因变量名称
    fallStopPoint : AIC值下降的相对值，当下降值小于该值则停止增加新变量
    
    Returns
    -------
    每步变量组合的AIC值数据框 
    
    Examples
    --------
    inDf = model_woe_df
    xVarNameLst = list(filter(lambda x: x not in ['request_id','TargetBad'], model_woe_df.columns.tolist()))
    yVarName = 'TargetBad'
    sle = 0.05
    sls = 0.05
    fallStopPoint=0.00001
    lr_stepwise_select(inDf, xVarNameLst, yVarName)
    '''
    variate=set(xVarNameLst)  #将字段名转换成字典类型
    selected=[]
    current_score,best_new_score=float('inf'),float('inf')  #目前的分数和最好分数初始值都为无穷大（因为AIC越小越好）
    aic_step = []
    #rsquare_step = []
    
    #循环筛选变量
    while variate:
        aic_with_variate=[]
        for candidate in variate:  #逐个遍历自变量
            formula="{}~{}".format(yVarName,"+".join(selected+[candidate]))  #将自变量名连接起来
            aic=ols(formula=formula,data=inDf).fit().aic  #利用ols训练模型得出aic值
            aic_with_variate.append((aic,candidate))  #将第每一次的aic值放进空列表
        aic_with_variate.sort(reverse=True)  #降序排序aic值
        best_new_score,best_candidate=aic_with_variate.pop()  #最好的aic值等于删除列表的最后一个值，以及最好的自变量等于列表最后一个自变量
        fall_rate = round((current_score - best_new_score) / best_new_score,5)
        #if (abs(fall_rate) >= 0.0002) & (best_new_score >= -1000):  #如果目前的aic值大于最好的aic值

        #最佳入选变量的t检验
        best_formula = "{}~{}".format(yVarName,"+".join(selected+[best_candidate])) 
        ols_rst = ols(formula=best_formula,data=inDf).fit()
        t_test_df = pd.DataFrame(ols_rst.summary().tables[1].data[1:], 
                                 columns=ols_rst.summary().tables[1].data[0])
        t_test_dict = t_test_df.set_index('')['P>|t|'].to_dict()
        
        #判断逐步回归是否继续迭代
        if (abs(fall_rate) >= fallStopPoint) and (float(t_test_dict[best_candidate])<=sle): 
            #已入选变量t检验
            selected_drop_lst = t_test_df[(t_test_df['P>|t|'].map(lambda x: float(x))>sls)&(t_test_df['']!=best_candidate)][''].tolist()
            if len(selected_drop_lst) > 0:  #假如存在入选变量t检验P值不显著的情况，剔除不显著的变量
                final_formula = "{}~{}".format(yVarName,"+".join([x for x in selected+[best_candidate] if x not in selected_drop_lst])) 
                best_new_score = ols(formula=final_formula,data=inDf).fit().aic
                fall_rate = round((current_score - best_new_score) / best_new_score,5)
                if (abs(fall_rate) >= fallStopPoint):
                    for var_item in selected_drop_lst:
                        variate.add(var_item)  #不显著变量放回变量池
                        selected.remove(var_item)  #不显著变量在模型中剔除
                    variate.remove(best_candidate)  #移除加进来的变量名，即第二次循环时，不考虑此自变量了
                    selected.append(best_candidate)  #将此自变量作为加进模型中的自变量
                    current_score=best_new_score  #最新的分数等于最好的分数
                    aic_step.append([current_score, 
                                     fall_rate, 
                                     ','.join(selected)])
                    print("aic is {},continuing!".format(current_score))  #输出最小的aic值
                else:
                    print("for selection over!")
                    break
                
            else :
                variate.remove(best_candidate)  #移除加进来的变量名，即第二次循环时，不考虑此自变量了
                selected.append(best_candidate)  #将此自变量作为加进模型中的自变量
                current_score=best_new_score  #最新的分数等于最好的分数
                aic_step.append([current_score, 
                                 fall_rate, 
                                 ','.join(selected)])
                print("aic is {},continuing!".format(current_score))  #输出最小的aic值
    
        else:
            print("for selection over!")
            break
    aic_step_df = pd.DataFrame(aic_step, columns=['AIC', 'AIC_Fall', 'xVarNameList'])
    
    matplotlib.use('Qt5Agg')
    plt.title("AIC下降趋势线")
    plt.plot(aic_step_df['AIC'])
    
    return(aic_step_df)




def lr_sklearn_model(inDf, xVarNameLs, yVarName):    
    '''
    Funcation Descriptions:
        使用sklearn的LogisticRegression包建立逻辑回归模型，并进行建模样本的预测。
        
    Parmaters
    ---------
    inDf       : 建模宽表数据框
    xVarNameLs : 自变量名称列表
    yVarName   : 因变量名称
    
    Returns
    -------
    列表：pred_df-预测概率数据框  coef_df-lr模型参数 
    
    Examples
    --------
    inDf = model_woe_df
    xVarNameLs = ['woe_bin_gbdt_boost7_2', 'woe_bin_xgb_boost46_2', 'woe_bin_gbdt_boost22_1', 
                  'woe_bin_xgb_boost9_2', 'woe_bin_gbdt_boost34_0', 'woe_bin_gbdt_boost32_3', 
                  'woe_bin_query_ccard_apply_3m_cnt_2', 'woe_bin_gbdt_boost27_1', 'woe_bin_gender_0']
    yVarName ='TargetBad'
    lr_sklearn_model(inDf, xVarNameLs, yVarName)
    '''
    model_df = inDf.copy()
    XDat = model_df[xVarNameLs]
    YDat = model_df[yVarName]
    X = np.array(XDat)
    y = np.array(YDat)   
    
    lr = LogisticRegression(random_state=0)
    lr.fit(X, y)
    
    ## 变量系数
    var_coef_df = pd.DataFrame({'VarName': xVarNameLs, 'Coefficient': lr.coef_[0]})
    intercept_df = pd.DataFrame({'VarName': 'Intercept', 'Coefficient': lr.intercept_})
    coef_df = pd.concat([intercept_df, var_coef_df] )
    coef_df = coef_df.merge(iv_df_auto_calculate(inDf=inDf , xVarList=xVarNameLs, yVar=yVarName), on='VarName', how='left')
    print(coef_df)
    
    ## 预测值
    model_df['y_pred'] = lr.predict_proba(X)[:,1]
    model_df['y_pred'] = model_df['y_pred'].map(lambda x: round(x,12))

    return {'pred_df': model_df, 'coef_df': coef_df}




def lr_hypothesis_test(inDf, xVarNameLs, yVarName):
    '''
    Funcation Descriptions:
        利用线性回归对逻辑回归的假设进行检验，主要包括：参数检验、异方差检验、残差正态性检验、多重共线性检验和残差序列相关性检验
        
    Parmaters
    ---------
    inDf       : 建模宽表数据框
    xVarNameLs : 自变量名称列表
    yVarName   : 因变量名称
    
    Returns
    -------
    参数检验、多重共线性检验、残差正态性检验、异方差检验和残差序列相关性检验的结果
    
    Examples
    --------
    inDf = model_woe_df
    xVarNameLs = aic_forward_df.ix[aic_var_num]['xVarNameList'].split(',')
    yVarName ='TargetBad'
    '''
    xVarLs = ','.join(xVarNameLs).replace(',','+')
    lm = ols('{} ~ {}'.format(yVarName, xVarLs), data=inDf).fit()
    model_rst = lm.summary()
    
    ## 参数检验
    print(model_rst.tables[0])
    print(model_rst.tables[1])
    print('\n')
    nvar = len(model_rst.tables[1].data)
    parameter_estimation = pd.DataFrame(model_rst.tables[1].data[1:nvar], columns = model_rst.tables[1].data[0])
    
    ## 多重共线性检验
    VIF = [round(vif_func(inDf[xVarNameLs].values, i),2) for i in range(len(xVarNameLs))]        
    vif_df = pd.DataFrame({'VarName': xVarNameLs, 
                           'VIF': VIF})
    print('VIF Test'.center(50))
    print('=================================================')
    print(vif_df)
    print('=================================================')
    print('\n')
    
    ## 残差概率图分布
    '''
    print('\n========================================================================================================\n')
    print('残差概率图分布'.center(50))
    stats.probplot(lm.resid, plot=plt)
    plt.title("Model1 Residuals Probability Plot")
    '''
    
    ## 残差正态性检验
    resid_normal_rst = stats.kstest(lm.resid, 'norm')
    resid_normal_rst_ser = lzip(('statistic','pvalue'),
                                (resid_normal_rst.statistic, resid_normal_rst.pvalue))
    print('Model1 Residuals Normal Distribution Test'.center(50))
    print('=================================================')
    for x,y in resid_normal_rst_ser:
        print('{:<12}: {}'.format(x,y))
    print('=================================================')
    print('\n')
    
    ## 异方差检验（BP/WLS）
    ## 异方差BP检验
    name = ['Lagrange multiplier statistic', 'p-value', 
            'f-value', 'f p-value']
    BP_test = sms.het_breuschpagan(lm.resid, lm.model.exog)
    print('异方差BP检验'.center(50))
    print('=================================================')
    for x,y in lzip(name,BP_test):
        print('{:<32}: {}'.format(x,y))
    print('=================================================')
    print("H0: 存在异方差！")
    if BP_test[1] < 0.05:
        print("结论：模型存在异方差！")
    else:
        print("结论：模型不存在异方差！")
    print('\n')
    
    ## 异方差white检验
    name = ['White statistic', 'p-value', 
            'f-value', 'f p-value']
    wls_test = sms.het_white(lm.resid, lm.model.exog)
    print('异方差white检验'.center(50))
    print('===========================================================')
    for x,y in lzip(name,wls_test):
        print('{:<32}: {}'.format(x,y))
    print('===========================================================')
    print("H0: 存在异方差！")
    if wls_test[1] < 0.05:
        print("结论：模型存在异方差！")
    else:
        print("结论：模型不存在异方差！")
    print('\n')

    ## 残差序列自相关性检验
    print('残差序列自相关性DW检验'.center(50))
    print(model_rst.tables[2])
    print("备注：DW的区间范围为[0,4],判断是否存在序列相关的标准是否在2附近")
    
    return {'vif':vif_df,
            'parameter_estimation':parameter_estimation}










