
import numpy as np
import decimal


def _one_dot_fun(x):
    if x==1:
        return 1
    else :
        return 0

def _much_dot_fun(x):
    if x > 1:
        return 1
    else :
        return 0


def char_to_number(xSeries):
    '''
        把业务理解为数值变量，但是python读取为object的变量，转换为数值变量。
    xSeries = df_raw['account_open_date']
    char_to_number(xSeries)
    '''
    top_list = xSeries[~xSeries.isnull()].unique().tolist()[0:1000]
    lst_len = len(top_list)   
    ser_len_cnt = len(xSeries.map(lambda x: len(str(x)) if not x is None else np.nan).unique())
    
    digit_num = 0
    one_dot_num = 0
    much_dot_num = 0
    dash_num = 0
    percent_num = 0
    
    if type(top_list[0]) is decimal.Decimal:
        res_ser = xSeries.astype('float')
    elif ser_len_cnt == 1:
        varchar_len = xSeries.map(lambda x: len(x))[0]
        if varchar_len >= 12:
            res_ser = xSeries
    else :
        for value_item in top_list:
            digit_num = digit_num + str(value_item).replace('.','').isdigit()
            one_dot_num = one_dot_num + _one_dot_fun(str(value_item).count('.'))
            much_dot_num = much_dot_num + _much_dot_fun(str(value_item).count('.'))
            dash_num = dash_num + str(value_item).count('-')
            percent_num = percent_num + str(value_item).count('%')
        if (digit_num == lst_len) & (one_dot_num > 0) & (much_dot_num == 0) & (dash_num == 0) & (percent_num==0):
            res_ser = xSeries.map(lambda x: float(str(x)) if x is not None else np.NaN)
        elif (digit_num == lst_len) & (one_dot_num == 0) & (much_dot_num == 0)  & (dash_num == 0) & (percent_num==0):
            res_ser = xSeries.map(lambda x: int(x) if x is not None else np.NaN)
        else :
            res_ser = xSeries
    
    return res_ser



