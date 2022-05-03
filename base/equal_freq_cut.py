# -*- coding: utf-8 -*-

from intervals import FloatInterval
import numpy as np



def equal_freq_cut(x, nBin):
    '''
    Function Description:
        对连续变量进行等频分箱
    
    Parameters
    ----------
    x    : 变量序列
    nBin : 分箱数量
    
    Examples
    --------
    x = ins_clean_df['cur5year_total_m1_cnt']
    nBin = 10
    
    '''
    x = x[x.notnull()]
    
    bin_ls = [np.percentile(x.values, 100/nBin*i) for i in range(nBin+1)]
    bin_ls = sorted(list(set(bin_ls)))
    
    bin_range_ls = list()    
    for i in range(len(bin_ls)-1):
        bin_range_ls.append([i+1,FloatInterval.open_closed(bin_ls[i], bin_ls[i+1])])
        
    return bin_range_ls



def equal_freq_cut_map(x, nBin, mapType='Bin'):
    '''
    Function Description:
        对连续变量进行等频分箱,并生成分箱序列
    
    Parameters
    ----------
    x       : 变量序列
    nBin    : 分箱数量
    mapType : 分箱结果以何种形式展示：Bin-区间形式、Level-水平形式
    
    Examples
    --------
    x = ins_corred_df['query_org_1m_cnt']
    nBin = 10
    mapType = 'Level'
    
    '''
    def bin_map(x, binList):
        if x <= binList[0]:
            return FloatInterval.open_closed(binList[0], binList[1])
        for i in range(len(binList)+1):
            if x > binList[i] and x <= binList[i+1]:
                return FloatInterval.open_closed(binList[i], binList[i+1])

    def level_map(x, binList):
        if x <= binList[0]:
            return 1
        for i in range(len(binList)+1):
            if x > binList[i] and x <= binList[i+1]:
                return i+2    

        
    x = x[x.notnull()]
    bin_ls = [np.percentile(x.values, 100/nBin*i) for i in range(nBin+1)]
    bin_ls = sorted(list(set(bin_ls)))
    bin_range_ls = list()    
    for i in range(len(bin_ls)-1):
        bin_range_ls.append([i+1,FloatInterval.open_closed(bin_ls[i], bin_ls[i+1])])
        
    if mapType == 'Bin':
        x_bin_ser = x.map(lambda x: bin_map(x, bin_ls))
    elif mapType == 'Level':
        x_bin_ser = x.map(lambda x: level_map(x, bin_ls))
        
    x_bin_ser.name = 'bin_{}'.format(x.name)
    
    return {'bin_range_ls': bin_range_ls,
            'x_bin_ser':  x_bin_ser
            }




    



