# -*- coding: utf-8 -*-

import pandas as pd
from analysis.corr_variables import corr_continuous_variables,corr_class_variables,corr_static_select,corr_p_select
from explore import variable_plot
from explore import variable_summary
from clean import drop_variables_by_class, drop_variables_by_unique_value, drop_variables_by_missing_value, \
                  drop_variables_by_overcenter_value
from analysis import corr_df_cal, corr_static_select
from analysis import iv_df_auto_calculate

from bins import step_bin
from bins import rate_bin_transfer
from selects import continuous_power_select, order_power_select, nominal_power_select
from selects import var_df_predict_psi, var_predict_psi_select
from woe import woe_transfer

from model import lr_forward_select, lr_sklearn_model, lr_hypothesis_test
from model import model_prob_evaluation, model_score_evaluation, score_calculate, scorecards
from oot import lr_formula_deployment, predict_compare, psi_cal, oot_prob_evaluation
 
from base import to_exist_excel, to_new_excel

from base import load_from_csv
from base import variable_char_type

dfa = pd.read_excel(r"C:\Users\finup\Desktop\临时文件夹\电动车标签_hebing_27hong_zhongwen.xlsx")

varlist = ['征信大分','三方大分']
corr_df = corr_continuous_variables(dfa, varlist,  method='spearmanr')
print(corr_df)

varList_1 = ['征信大分','三方大分','中大信分数']
dd = corr_class_variables(dfa, varList_1)
print(dd)


df = dfa.copy()

df = df.drop(['Unnamed: 0','客户姓名', '用户id','is_reject','3+件数',
       '7+件数', '30+件数', '流入金额', '3+金额', '7+金额', '30+金额',
       '申请提现时间', '申请提现金额', '放款件数','放款金额', '期限', '放款时间',
       '申请状态', '是否最终风控拒贷', '是否征信拒贷', '拒贷原因'], axis=1)
df = df.rename(columns={'流入件数': 'TargetBad'})

#=========================================基础参数设置============================================
file_path = r'C:\Users\finup\Desktop\ceshis'
#var_time_by = 'apply_time'
var_key = 'id_no'
var_target = 'TargetBad'


#==========================================变量字符类型核实===========================================

var_type_df = variable_char_type(inDf=df, 
                                 keyVarList=[var_key], 
                                 TargetVarList=[var_target], 
                                 unScaleVarList=[])

variable_plot(inDf =df, 
              inVarClassDf =var_type_df, 
              savUrl = file_path)

# 输出变量的统计特征
var_explore_result = variable_summary(inDf=df,inVarClassDf=var_type_df )


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



#==========================================无效变量剔除========================================= 

# 按照变量类型确定需要剔除的变量列表
drop_var_list_by_class = drop_variables_by_class(
                        inVarClassDf = var_type_df, 
                        toDropVarClassList = ['UnScale', 'Date', 'Droped'])
# 确定只有一个值的变量列表
drop_var_list_by_unique = drop_variables_by_unique_value(
                        inVarClassDf = var_type_df)
# 按照缺失值比例确定需要剔除的变量列表
drop_var_list_by_missing = drop_variables_by_missing_value(
                        inVarClassDf = var_type_df,
                        dropMissRatePoint = 0.9)
# 剔除分类样本过于集中的变量
drop_var_list_by_overcenter = drop_variables_by_overcenter_value(
                        inVarDistDf = var_explore_result['classFreqSummary'],
                        dropOverCenterPoint = 0.95)

# 在建模样本中剔除需要剔除的变量
drop_var_list = drop_var_list_by_class+drop_var_list_by_unique+drop_var_list_by_missing+drop_var_list_by_overcenter
ins_clean_df = df.drop(drop_var_list, axis=1)

# 剔除变量后，更新建模变量类型及样本量数据框
ins_clean_class_df = var_type_df[~var_type_df['index'].isin(drop_var_list)]
drop_by_freq_df = var_type_df[var_type_df['index'].isin(drop_var_list)]

print('===================  无效变量剔除完成！ =====================')


# 阶段性结果输出
to_new_excel(filePath = file_path,
             fileName = '无效变量基本条件剔除',
             sheetName = '剔除变量',
             dataFrm = drop_by_freq_df)
to_exist_excel(filePath = file_path,
               fileName = '无效变量基本条件剔除',
               sheetName = '保留变量',
               dataFrm = ins_clean_class_df)




## object变量缺失值填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] == 'object']['index'].tolist():
    ins_clean_df[var_name] = ins_clean_df[var_name].fillna('Missing')

## 数值型变量缺失值中位数填充
for var_name in ins_clean_class_df[ins_clean_class_df['Dtypes'] != 'object']['index'].tolist():
    ## 对缺失值比例小于5%的变量进行填充
    if ins_clean_class_df[ins_clean_class_df['index'] == var_name]['NmissRate'].tolist()[0] <= 0.05:
        median_value = ins_clean_df[var_name].median()
        ins_clean_df[var_name] = ins_clean_df[var_name].fillna(median_value)