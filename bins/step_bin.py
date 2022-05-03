
from base import nominal_ls_freq_combine, nominal_bin_transfer
from base import order_ls_freq_combine, order_bin_transfer
from base import binary_df_stat

from bins import nominal_df_rate_combine
#from bins import nominal_code_map
from bins import order_df_rate_combine
from bins import continuous_df_rate_bin
from analysis import iv_df_from_freq 

import pandas as pd
import math


def step_bin(inDf, inTypeDf, varKey, varTarget, conBins=20):
    
    '''
    Funcation Descriptions:
        本函数用于变量的分箱逐步合并生成新变量，然后计算新变量的chisq和IV。过程如下：
        1、对于分类变量（名义变量、有序变量）类的样本占比小于5%的变量，进行频数合并；
        2、对于分类变量，使用分箱合并的数据集，按照逾期率相近度逐步合并，生成新的分箱变量；
           对于连续变量，先进行等频分箱，然后按照逾期率相近度逐步合并，生成新的分箱变量；
        3、计算分箱变量的chisq值
        4、计算分箱变量的IV值
        5、四类变量生成的结果合并
    
    Parmaters
    ---------
    inDf      : 数据框
    inTypeDf  : 变量类型数据框
    varKey    : 主键变量
    varTarget : 目标变量
    
    Returns
    -------
    List: bin_step_dist_df - 全部分箱变量频数、逾期率等的数据框；
          bin_step_power_df - 全部分箱变量chisq、IV的数据框；
          bin_iv_cal_df - 全部分箱变量计算IV的过程，包括WOE
        
    Examples
    --------    
    inDf=ins_corred_df
    inTypeDf=ins_ived_class_df
    varKey=var_key
    varTarget=var_target    
    step_bin(inDf, inTypeDf, varKey, varTarget)
    ''' 
    ##---------------------------------------------------------------------------------------------
    ## 名义变量频数合并
    freq_nom_cmb_rst = nominal_ls_freq_combine(inDf=inDf, 
                            varList=inTypeDf[inTypeDf['Dclass']=='Nominal']['index'].tolist(), 
                            cutOff=0.05)
    ## 名义变量频数合并后生成新数据集
    freq_nominal_bin_df = nominal_bin_transfer(inRawDf = inDf, 
                                          inMapDf = freq_nom_cmb_rst['bin_freq_df'], 
                                          inKeyName = varKey,
                                          inTargetName = varTarget)
    ## 名义变量逾期率合并及计算chisq统计量
    rate_nom_cmb_rst = nominal_df_rate_combine(inDf = freq_nominal_bin_df,
                                               keyVarName = varKey,
                                               yVarName = varTarget)
    
    ## 名义变量分箱转换为原始码值
    '''
    rate_nom_cmb_freq_df = nominal_code_map(inDf = rate_nom_cmb_rst['rate_step_bin_df'],
                                            inFreqDf = freq_nom_cmb_rst['raw_freq_df'],
                                            inVar = 'VarName',
                                            inBin = 'Bins',
                                            inFreqVar = 'VarName',
                                            inFreqLevel = 'Levels',
                                            inFreqBin = 'Bins')
    '''
    rate_nom_cmb_freq_df = rate_nom_cmb_rst['rate_step_bin_df']
    rate_nom_cmb_freq_df['NewVarName'] = rate_nom_cmb_freq_df['VarName'] + '_' + rate_nom_cmb_freq_df['Steps'].astype('int').astype('str')
    rate_nom_cmb_chisq_df = rate_nom_cmb_rst['rate_step_chisq_df']
    rate_nom_cmb_chisq_df['NewVarName'] = rate_nom_cmb_chisq_df['VarName'] + '_' + rate_nom_cmb_chisq_df['Steps'].astype('int').astype('str')
    
    ## 名义分箱变量IV值计算
    iv_nom_rst = iv_df_from_freq(inDf = rate_nom_cmb_freq_df, varName = 'NewVarName', var0 = 0,var1 =1)
    iv_nom_value_df = iv_nom_rst['iv_value_df']
    iv_nom_cal_df = iv_nom_rst['iv_cal_df']
    print("*" * 3,'名义分箱变量IV值计算完成!')
    
    
    ##---------------------------------------------------------------------------------------------
    ## 有序变量频数合并
    freq_ord_cmb_rst = order_ls_freq_combine(inDf = inDf, 
                                            varList = inTypeDf[inTypeDf['Dclass']=='Order']['index'].tolist(),
                                            cutOff = 0.05 )
    
    ## 有序变量频数合并后，生成新数据集
    freq_order_bin_df = order_bin_transfer(inRawDf = inDf, 
                                            inMapDf = freq_ord_cmb_rst['orders_bin_freq_df'],
                                            inKeyName = varKey,
                                            inTargetName = varTarget)
    
    ## 有序变量逾期率合并及计算chisq统计量
    rate_ord_cmb_rst = order_df_rate_combine(inDf = freq_order_bin_df, 
                                             inBinMapDat = freq_ord_cmb_rst['orders_bin_freq_df'], 
                                             yVarName = varTarget)
    rate_ord_cmb_freq_df = rate_ord_cmb_rst['rate_step_bin_df']
    rate_ord_cmb_freq_df['NewVarName'] = rate_ord_cmb_freq_df['VarName'] + '_' + rate_ord_cmb_freq_df['Steps'].astype('int').astype('str')
    rate_ord_cmb_chisq_df = rate_ord_cmb_rst['rate_step_chisq_df']
    rate_ord_cmb_chisq_df['NewVarName'] = rate_ord_cmb_chisq_df['VarName'] + '_' + rate_ord_cmb_chisq_df['Steps'].astype('int').astype('str')
    
    ## 有序变量分箱变量IV值计算
    iv_ord_rst = iv_df_from_freq(inDf = rate_ord_cmb_freq_df, varName = 'NewVarName', var0 = 0,var1 =1)
    iv_ord_value_df = iv_ord_rst['iv_value_df']
    iv_ord_cal_df = iv_ord_rst['iv_cal_df']
    print("*" * 3,'有序分箱变量IV值计算完成!')
    
    
    ##---------------------------------------------------------------------------------------------
    ## 连续变量等频分箱后，逐步合并分箱及计算chisq统计量
    rate_con_cmb_rst = continuous_df_rate_bin(inDf = inDf, 
                                              varList = inTypeDf[inTypeDf['Dclass']=='Continuous']['index'].tolist(),
                                              yVarName = varTarget,
                                              n = conBins)
    rate_con_cmb_freq_df = rate_con_cmb_rst['rate_step_bin_df']
    rate_con_cmb_freq_df['NewVarName'] = rate_con_cmb_freq_df['VarName'] + '_' + rate_con_cmb_freq_df['Steps'].astype('int').astype('str')
    rate_con_cmb_chisq_df = rate_con_cmb_rst['rate_step_chisq_df']
    rate_con_cmb_chisq_df['NewVarName'] = rate_con_cmb_chisq_df['VarName'] + '_' + rate_con_cmb_chisq_df['Steps'].astype('int').astype('str')
    
    ## 分箱变量IV值计算
    iv_con_rst = iv_df_from_freq(inDf = rate_con_cmb_freq_df, varName = 'NewVarName', var0 = 0,var1 =1)
    iv_con_value_df = iv_con_rst['iv_value_df']
    iv_con_cal_df = iv_con_rst['iv_cal_df']
    
    
    ##---------------------------------------------------------------------------------------------
    ## 二值变量频数、逾期率及计算chisq统计量
    binary_freq_rst = binary_df_stat(inDf = inDf,
                                     varList = inTypeDf[inTypeDf['Dclass']=='Binary']['index'].tolist(),
                                     varY = varTarget)
    binary_freq_df = binary_freq_rst['vars_freq_df'].rename(columns={'VarName':'NewVarName', 'Levels':'Bins', 'Total':'All'})
    binary_freq_df = binary_freq_df[binary_freq_df['Bins']!='Total'].drop('MissCnt', axis=1)
    binary_freq_df['Rate'] = round(binary_freq_df[1]/binary_freq_df['All'],6)
    binary_freq_df['VarName'] = binary_freq_df['NewVarName']
    binary_chisq_df = binary_freq_rst['vars_chisq_df']
    binary_chisq_df['NewVarName'] = binary_chisq_df['VarName']
    
    ## 二值变量IV值计算
    iv_binary_rst = iv_df_from_freq(inDf = binary_freq_df, varName = 'NewVarName', var0 = 0,var1 =1)
    iv_binary_value_df = iv_binary_rst['iv_value_df']
    iv_binary_cal_df = iv_binary_rst['iv_cal_df']

    ##=============================================================================================##
    ## 分箱结果合并  
    #四类指标频数及逾期率合并
    rate_nom_cmb_freq_df['Dclass'] = 'Nominal'
    rate_ord_cmb_freq_df['Dclass'] = 'Order'
    rate_con_cmb_freq_df['Dclass'] = 'Continuous'
    binary_freq_df['Dclass'] = 'Binary'
    step_dist_df = pd.concat([rate_nom_cmb_freq_df, rate_ord_cmb_freq_df, rate_con_cmb_freq_df, binary_freq_df], axis=0)
    step_dist_df = step_dist_df[['NewVarName','VarName','Steps','Levels','Bins',0,1,'All','Rate','Dclass']]

    #四类指标chisq值合并
    rate_nom_cmb_chisq_df['Dclass'] = 'Nominal'
    rate_ord_cmb_chisq_df['Dclass'] = 'Order'
    rate_con_cmb_chisq_df['Dclass'] = 'Continuous'
    binary_chisq_df['Dclass'] = 'Binary'
    step_chisq_df = pd.concat([rate_nom_cmb_chisq_df, rate_ord_cmb_chisq_df, rate_con_cmb_chisq_df, binary_chisq_df], axis=0)
    #四类指标IV值合并
    iv_value_df = pd.concat([iv_nom_value_df,iv_ord_value_df,iv_con_value_df,iv_binary_value_df],axis=0)
    iv_value_df = iv_value_df.rename(columns={'VarName':'NewVarName'})
    #chisq值和IV值合并
    step_power_df = step_chisq_df.merge(iv_value_df, on='NewVarName', how='left' )   
    step_power_df = step_power_df[['NewVarName','VarName','Steps','Sample_Size','Chisq_Stat','Chisq_Df','Chisq_P',
                                   'Chisq_DfStat','IV','Dclass']]
    step_power_df['IV_dfsqrt'] = (step_power_df['IV'] / step_power_df['Chisq_Df'].map(lambda x: math.sqrt(x+1))).map(lambda x: round(x,4))

    #IV计算过程
    iv_nom_cal_df['Dclass'] = 'Nominal'
    iv_ord_cal_df['Dclass'] = 'Order'
    iv_con_cal_df['Dclass'] = 'Continuous' 
    iv_binary_cal_df['Dclass'] = 'Binary' 
    iv_cal_df = pd.concat([iv_nom_cal_df,iv_ord_cal_df,iv_con_cal_df,iv_binary_cal_df], axis=0)


    return {'bin_step_dist_df':step_dist_df, 
            'bin_step_power_df':step_power_df,
            'bin_iv_cal_df':iv_cal_df}


