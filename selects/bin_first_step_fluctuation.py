# -*- coding: utf-8 -*-

from statsmodels.formula.api import ols
import pandas as pd
import numpy as np

def var_bin_fluctuation(inDf):
    
    '''
    Descriptions:
        通过将连续变量或者有序变量的首次分箱的逾期率与分箱顺序进行线性回归分析，获得拟合效果的R方，
    判断该变量初始分箱单调的稳定性。
    
    Parameters
    ----------
    inDf : 变量第一步细分箱的结果，务必剔除缺失值组
    
    Returns
    -------
    变量首次分箱的逾期率的拟合效果调和R方
    
    Examples
    --------
    dist_df = bin_step_dist_df[bin_step_dist_df['Dclass'].isin('Continuous','Order')]
    first_step_dist = dist_df[(dist_df['Steps']==0) & (dist_df['Bins']!='nan')]
    var_bin_fluctuation(inDf=first_step_dist)
    '''

    rsquared_lst = []
    varname_lst = inDf['VarName'].unique()
    for item in varname_lst:
        #item = 'bin_self_deposit_rate_max'
        var_dist_df = inDf[inDf['VarName']==item]
        if var_dist_df.shape[0]>3:
            var_dist_df['x'] = var_dist_df.index
            formula="{}~{}".format('Rate',"+".join(['x']))  #将自变量名连接起来
            fit_rst = ols(formula=formula,data=var_dist_df).fit()  #利用ols训练模型得出aic值    
            rsquared_lst.append(fit_rst.rsquared_adj)
        else :
            rsquared_lst.append(np.nan)
    rsquared_df = pd.DataFrame({'VarName': varname_lst,
                                'rsquared_adj': rsquared_lst})
    
    return rsquared_df




def var_bin_fluctuation_select(inDf, r2Door):
    '''
    Examples
    --------
    dist_df = bin_step_dist_df[bin_step_dist_df['Dclass'].isin(['Continuous','Order'])]
    first_step_dist = dist_df[(dist_df['Steps']==0) & (dist_df['Bins']!='nan')]
    inDf=first_step_dist
    r2door = 0.25
    var_bin_fluctuation_select(inDf, r2door)
    '''

    rsquared_df = var_bin_fluctuation(inDf)    
    select_var = rsquared_df[(rsquared_df['rsquared_adj'] > r2Door) | pd.isnull(rsquared_df['rsquared_adj'])]['VarName'].tolist()
    
    return {'rsquared_df': rsquared_df, 'select_var_lst':select_var}












