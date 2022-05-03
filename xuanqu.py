# -*- coding: utf-8 -*-

import pandas as pd
from base import load_from_csv
from base import variable_char_type
from base import cross_table
from base import to_exist_excel, to_new_excel
from explore import variable_plot
from explore import variable_summary
from clean import drop_variables_by_class, drop_variables_by_unique_value, drop_variables_by_missing_value, \
                  drop_variables_by_overcenter_value
from analysis import corr_df_cal, corr_static_select
from analysis.iv import iv_df_auto_calculate
from analysis.iv import iv_class_variable,iv_calculate
from bins import step_bin
from bins import rate_bin_transfer
from selects import continuous_power_select, order_power_select, nominal_power_select
from selects import var_df_predict_psi, var_predict_psi_select
from woe import woe_transfer

from model import lr_forward_select, lr_sklearn_model, lr_hypothesis_test
from model import model_prob_evaluation, model_score_evaluation, score_calculate, scorecards
from oot import lr_formula_deployment, predict_compare, psi_cal, oot_prob_evaluation


dfs = pd.read_csv(r"C:\Users\finup\Desktop\解析出的指标_zhengli_1.csv")

#=========================================基础参数设置============================================
file_path = r'C:\Users\finup\Desktop\moxing'
#var_time_by = 'apply_time'
var_key = 'id_no'
var_target = 'if_flow'


#==========================================变量字符类型核实===========================================

var_type_df = variable_char_type(inDf=dfs, 
                                 keyVarList=[var_key], 
                                 TargetVarList=[var_target], 
                                 unScaleVarList=[])


ins_raw_var_class_df = variable_char_type(inDf = dfs, 
                           keyVarList = [var_key], 
                           TargetVarList = [var_target],
                           unScaleVarList = ['id_no'])



# 输出变量分布图（连续变量和分类变量两类进行输出）
variable_plot(inDf = dfs, 
              inVarClassDf = ins_raw_var_class_df, 
              savUrl = file_path)

# 输出变量的统计特征
var_explore_result = variable_summary(inDf=dfs,inVarClassDf=ins_raw_var_class_df)


# 连续变量统计特征excel输出
to_new_excel(filePath = file_path,
             fileName = 'VarExploreResult',
             sheetName = '连续变量统计特征',
             dataFrm = var_explore_result['contStatSummary'])

# 分类变量分布及占比excel输出
to_exist_excel(filePath = file_path,
               fileName = 'VarExploreResult',
               sheetName= '分类变量分布',
               dataFrm=var_explore_result['classFreqSummary'])




# #按照变量类型确定需要剔除的变量列表
drop_var_list_by_class = drop_variables_by_class(
                        inVarClassDf = ins_raw_var_class_df, 
                        toDropVarClassList = ['id_no'])
# 确定只有一个值的变量列表
drop_var_list_by_unique = drop_variables_by_unique_value(
                        inVarClassDf = ins_raw_var_class_df)
# 按照缺失值比例确定需要剔除的变量列表
drop_var_list_by_missing = drop_variables_by_missing_value(
                        inVarClassDf = ins_raw_var_class_df,
                        dropMissRatePoint = 0.9)
# 剔除分类样本过于集中的变量
drop_var_list_by_overcenter = drop_variables_by_overcenter_value(
                        inVarDistDf = var_explore_result['classFreqSummary'],
                        dropOverCenterPoint = 0.95)
wuyong = ['Unnamed: 0','id_no','PYDECIRCRVL011']

# 在建模样本中剔除需要剔除的变量
drop_var_list = drop_var_list_by_class+drop_var_list_by_unique+drop_var_list_by_missing+drop_var_list_by_overcenter+wuyong
ins_clean_df = dfs.drop(drop_var_list, axis=1)

# 剔除变量后，更新建模变量类型及样本量数据框
ins_clean_class_df = ins_raw_var_class_df[~ins_raw_var_class_df['index'].isin(drop_var_list)]
drop_by_freq_df = ins_raw_var_class_df[ins_raw_var_class_df['index'].isin(drop_var_list)]

print('===================  无效变量剔除完成！ =====================')

del drop_var_list_by_class, drop_var_list_by_unique, drop_var_list_by_missing, drop_var_list_by_overcenter
del drop_var_list



# 阶段性结果输出
to_new_excel(filePath = file_path,
             fileName = '无效变量基本条件剔除',
             sheetName = '剔除变量',
             dataFrm = drop_by_freq_df)
to_exist_excel(filePath = file_path,
               fileName = '无效变量基本条件剔除',
               sheetName = '保留变量',
               dataFrm = ins_clean_class_df)



#==========================================缺失值填充===========================================
## object变量缺失值填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] == 'object']['index'].tolist():
    ins_clean_df[var_name] = ins_clean_df[var_name].fillna('Missing')

## 数值型变量缺失值中位数填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] != 'object']['index'].tolist():
    ## 对缺失值比例小于5%的变量进行填充
    if ins_clean_class_df[ins_clean_class_df['index'] == var_name]['NmissRate'].tolist()[0] <= 0.05:
        median_value = ins_clean_df[var_name].median()
        ins_clean_df[var_name] = ins_clean_df[var_name].fillna(median_value)
        
        
        
#==========================================变量相关性分析========================================= 





corr_rst = corr_df_cal(inDf = ins_clean_df, varTypeDf = ins_clean_class_df)


drop_con_corr_ls = corr_static_select(inCorrDf=corr_rst['continue_cor_df'],
                                      statVar='corr_coef',
                                      dropSign = '>',
                                      statPoint=0.8)
drop_class_corr_ls = corr_static_select(inCorrDf=corr_rst['class_chisq_df'],
                                        statVar='Chisq_DfStat',
                                        dropSign = '>',
                                        statPoint=100)

ins_corred_class_df = ins_clean_class_df[~ins_clean_class_df['index'].isin(drop_con_corr_ls+drop_class_corr_ls)]
ins_corred_df = ins_clean_df.drop(drop_con_corr_ls+drop_class_corr_ls, axis=1)

print('===================  变量相关性分析完成！ =====================')



# 阶段性结果输出
continue_cor_df = corr_rst['continue_cor_df']
class_chisq_df = inCorrDf=corr_rst['class_chisq_df']
continue_cor_df['varname1_drop'] = continue_cor_df['varname1'].map(lambda x: 1 if x in drop_con_corr_ls else 0)
class_chisq_df['varname1_drop'] = class_chisq_df['varname1'].map(lambda x: 1 if x in drop_con_corr_ls else 0)



iv_var_ls = ins_corred_class_df[~ins_corred_class_df['Dclass'].isin(['Key','Target'])]['index'].tolist()
iv_df = iv_df_auto_calculate(
                            inDf = ins_corred_df,
                            xVarList = iv_var_ls,
                            yVar = var_target)




from base.freq_order_bin import order_freq_combine_transfer
from base.freq_nominal_bin import nominal_freq_combine_transfer
from base.equal_freq_cut import equal_freq_cut_map


dd = nominal_freq_combine_transfer(xVar=ins_corred_df['PYDECIRCRVL014'], cutOff=0.03)
print(dd)

from base.freq_stats import var_freq_dist
ee = var_freq_dist(ins_corred_df['PYDECIRCRVL014'], pctFormat=True)
print(ee)

ff = order_freq_combine_transfer(xVar=ins_corred_df['PYDECIRCRVL008'], cutOff=0.03)
print(ff)

listww = []
for uu in iv_var_ls:
    print(uu)
    try:
        iv_var = iv_calculate(ins_corred_df, uu, 'if_flow')['IV']
        print(iv_var)
    except:
        listww.append(uu)
        