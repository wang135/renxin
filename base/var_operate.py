# -*- coding: utf-8 -*-

import numpy as np

def var_operate(x1, x2, sign):
    '''
    Function Description:
        变量加减乘除运算
    
    Parameters
    ----------
    x1 : 变量序列1
    x2 : 变量序列2
    
    Returns
    -------
    运算结果的序列
    '''
    px1 = np.where(np.isnan(x1), 0, x1)
    px2 = np.where(np.isnan(x2), 0, x2)
    
    if sign == '-':
        return px1 - px2
    elif sign == '+':
        return px1 + px2
    elif sign == '*':
        return px2 * px2
    elif sign == '/':
        return px2 / px2
    





