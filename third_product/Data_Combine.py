# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:09:04 2020

@author: finup
"""


import sys
sys.path.append(r'F:\Python\CreditScoreCard')


import numpy as np
import pandas as pd
import datetime
import math
from statsmodels.formula.api import ols

#from base import *

from base import load_from_csv
from base import variable_char_type
from base import cross_table
from base import to_exist_excel, to_new_excel
from explore import variable_plot
from explore import variable_summary
from clean import drop_variables_by_class, drop_variables_by_unique_value, drop_variables_by_missing_value, \
                  drop_variables_by_overcenter_value
from bins import step_bin
from analysis import corr_df_cal, corr_static_select
from analysis import iv_df_auto_calculate

from bins import rate_bin_transfer
from selects import continuous_power_select, order_power_select, nominal_power_select
from selects import var_df_predict_psi, var_predict_psi_select
from woe import woe_transfer

from model import lr_forward_select, lr_sklearn_model, lr_hypothesis_test
from model import model_prob_evaluation, model_score_evaluation, score_calculate, scorecards
from oot import lr_formula_deployment, predict_compare, psi_cal, oot_prob_evaluation



###################################################### 百融
BR_ApplyEvaluate = pd.read_csv(r'F:\data\三方提测样本返回\BR_ApplyEvaluate.csv', header=0, skiprows=[1])
BR_ApplyEvaluate = BR_ApplyEvaluate.drop(['cus_num', 'sl_user_date', 'user_time', 'custApiCode', 'swift_number',
                                          'cus_username', 'code', 'flag_applyevaluate'], axis=1)

apy = BR_ApplyEvaluate.drop_duplicates()


##======================================================================================================================================================##
##--m1
#银行查询次数
apy['last_m1_id_bank_query_num'] = apy[['ae_m1_id_bank_national_allnum','ae_m1_id_bank_region_allnum']].sum(axis=1)
apy['last_m1_phone_bank_query_num'] = apy[['ae_m1_cell_bank_national_allnum','ae_m1_cell_bank_region_allnum']].sum(axis=1)
apy['last_m1_bank_query_num'] = apy[['last_m1_id_bank_query_num','last_m1_phone_bank_query_num']].max(axis=1)
apy = apy.drop(['last_m1_id_bank_query_num','last_m1_phone_bank_query_num'], axis=1)

#保险查询次数
apy['last_m1_ins_query_num'] = apy[['ae_m1_id_nbank_ins_allnum','ae_m1_cell_nbank_ins_allnum']].max(axis=1)

#消金查询次数
apy['last_m1_cons_query_num'] = apy[['ae_m1_id_nbank_cons_allnum','ae_m1_cell_nbank_cons_allnum']].max(axis=1)

#互金查询次数
apy['last_m1_id_fintech_query_num'] = apy[['ae_m1_id_nbank_top_allnum','ae_m1_id_nbank_sloan_allnum','ae_m1_id_nbank_trans_allnum']].sum(axis=1)
apy['last_m1_phone_fintech_query_num'] = apy[['ae_m1_cell_nbank_top_allnum','ae_m1_cell_nbank_sloan_allnum','ae_m1_cell_nbank_trans_allnum']].sum(axis=1)
apy['last_m1_fintech_query_num'] = apy[['last_m1_id_fintech_query_num','last_m1_phone_fintech_query_num']].max(axis=1)
apy = apy.drop(['last_m1_id_fintech_query_num','last_m1_phone_fintech_query_num'], axis=1)

#汽车金融查询次数
apy['last_m1_carloans_query_num'] = apy[['ae_m1_id_nbank_autofin_allnum','ae_m1_cell_nbank_autofin_allnum']].max(axis=1)

#其他金融查询次数
apy['last_m1_id_other_query_num'] = apy[['ae_m1_id_nbank_else_rel_allnum','ae_m1_id_nbank_else_cons_allnum','ae_m1_id_nbank_else_pdl_allnum']].sum(axis=1)
apy['last_m1_phone_other_query_num'] = apy[['ae_m1_cell_nbank_else_rel_allnum','ae_m1_cell_nbank_else_cons_allnum','ae_m1_cell_nbank_else_pdl_allnum']].sum(axis=1)
apy['last_m1_other_query_num'] = apy[['last_m1_id_other_query_num','last_m1_phone_other_query_num']].max(axis=1)
apy = apy.drop(['last_m1_id_other_query_num','last_m1_phone_other_query_num'], axis=1)

#银行查询机构数量
apy['last_m1_id_bank_org_num'] = apy[['ae_m1_id_bank_national_orgnum','ae_m1_id_bank_region_orgnum']].sum(axis=1)
apy['last_m1_phone_bank_org_num'] = apy[['ae_m1_cell_bank_national_orgnum','ae_m1_cell_bank_region_orgnum']].sum(axis=1)
apy['last_m1_bank_org_num'] = apy[['last_m1_id_bank_org_num','last_m1_phone_bank_org_num']].max(axis=1)
apy = apy.drop(['last_m1_id_bank_org_num','last_m1_phone_bank_org_num'], axis=1)

#保险查询机构数量
apy['last_m1_ins_org_num'] = apy[['ae_m1_id_nbank_ins_orgnum','ae_m1_cell_nbank_ins_orgnum']].max(axis=1)

#消金查询机构数量
apy['last_m1_cons_org_num'] = apy[['ae_m1_id_nbank_cons_orgnum','ae_m1_cell_nbank_cons_orgnum']].max(axis=1)

#互金查询机构数量
apy['last_m1_id_fintech_org_num'] = apy[['ae_m1_id_nbank_top_orgnum','ae_m1_id_nbank_sloan_orgnum','ae_m1_id_nbank_trans_orgnum']].sum(axis=1)
apy['last_m1_phone_fintech_org_num'] = apy[['ae_m1_cell_nbank_top_orgnum','ae_m1_cell_nbank_sloan_orgnum','ae_m1_cell_nbank_trans_orgnum']].sum(axis=1)
apy['last_m1_fintech_org_num'] = apy[['last_m1_id_fintech_org_num','last_m1_phone_fintech_org_num']].max(axis=1)
apy = apy.drop(['last_m1_id_fintech_org_num','last_m1_phone_fintech_org_num'], axis=1)

#汽车金融查询机构数量
apy['last_m1_carloans_org_num'] = apy[['ae_m1_id_nbank_autofin_orgnum','ae_m1_cell_nbank_autofin_orgnum']].max(axis=1)

#其他金融查询机构数量
apy['last_m1_id_other_org_num'] = apy[['ae_m1_id_nbank_else_rel_orgnum','ae_m1_id_nbank_else_cons_orgnum','ae_m1_id_nbank_else_pdl_orgnum']].sum(axis=1)
apy['last_m1_phone_other_org_num'] = apy[['ae_m1_cell_nbank_else_rel_orgnum','ae_m1_cell_nbank_else_cons_orgnum','ae_m1_cell_nbank_else_pdl_orgnum']].sum(axis=1)
apy['last_m1_other_org_num'] = apy[['last_m1_id_other_org_num','last_m1_phone_other_org_num']].max(axis=1)
apy = apy.drop(['last_m1_id_other_org_num','last_m1_phone_other_org_num'], axis=1)




##======================================================================================================================================================##
##--m3
#银行查询次数
apy['last_m3_id_bank_query_num'] = apy[['ae_m3_id_bank_national_allnum','ae_m3_id_bank_region_allnum']].sum(axis=1)
apy['last_m3_phone_bank_query_num'] = apy[['ae_m3_cell_bank_national_allnum','ae_m3_cell_bank_region_allnum']].sum(axis=1)
apy['last_m3_bank_query_num'] = apy[['last_m3_id_bank_query_num','last_m3_phone_bank_query_num']].max(axis=1)
apy = apy.drop(['last_m3_id_bank_query_num','last_m3_phone_bank_query_num'], axis=1)

#保险查询次数
apy['last_m3_ins_query_num'] = apy[['ae_m3_id_nbank_ins_allnum','ae_m3_cell_nbank_ins_allnum']].max(axis=1)

#消金查询次数
apy['last_m3_cons_query_num'] = apy[['ae_m3_id_nbank_cons_allnum','ae_m3_cell_nbank_cons_allnum']].max(axis=1)

#互金查询次数
apy['last_m3_id_fintech_query_num'] = apy[['ae_m3_id_nbank_top_allnum','ae_m3_id_nbank_sloan_allnum','ae_m3_id_nbank_trans_allnum']].sum(axis=1)
apy['last_m3_phone_fintech_query_num'] = apy[['ae_m3_cell_nbank_top_allnum','ae_m3_cell_nbank_sloan_allnum','ae_m3_cell_nbank_trans_allnum']].sum(axis=1)
apy['last_m3_fintech_query_num'] = apy[['last_m3_id_fintech_query_num','last_m3_phone_fintech_query_num']].max(axis=1)
apy = apy.drop(['last_m3_id_fintech_query_num','last_m3_phone_fintech_query_num'], axis=1)

#汽车金融查询次数
apy['last_m3_carloans_query_num'] = apy[['ae_m3_id_nbank_autofin_allnum','ae_m3_cell_nbank_autofin_allnum']].max(axis=1)

#其他金融查询次数
apy['last_m3_id_other_query_num'] = apy[['ae_m3_id_nbank_else_rel_allnum','ae_m3_id_nbank_else_cons_allnum','ae_m3_id_nbank_else_pdl_allnum']].sum(axis=1)
apy['last_m3_phone_other_query_num'] = apy[['ae_m3_cell_nbank_else_rel_allnum','ae_m3_cell_nbank_else_cons_allnum','ae_m3_cell_nbank_else_pdl_allnum']].sum(axis=1)
apy['last_m3_other_query_num'] = apy[['last_m3_id_other_query_num','last_m3_phone_other_query_num']].max(axis=1)
apy = apy.drop(['last_m3_id_other_query_num','last_m3_phone_other_query_num'], axis=1)

#银行查询机构数量
apy['last_m3_id_bank_org_num'] = apy[['ae_m3_id_bank_national_orgnum','ae_m3_id_bank_region_orgnum']].sum(axis=1)
apy['last_m3_phone_bank_org_num'] = apy[['ae_m3_cell_bank_national_orgnum','ae_m3_cell_bank_region_orgnum']].sum(axis=1)
apy['last_m3_bank_org_num'] = apy[['last_m3_id_bank_org_num','last_m3_phone_bank_org_num']].max(axis=1)
apy = apy.drop(['last_m3_id_bank_org_num','last_m3_phone_bank_org_num'], axis=1)

#保险查询机构数量
apy['last_m3_ins_org_num'] = apy[['ae_m3_id_nbank_ins_orgnum','ae_m3_cell_nbank_ins_orgnum']].max(axis=1)

#消金查询机构数量
apy['last_m3_cons_org_num'] = apy[['ae_m3_id_nbank_cons_orgnum','ae_m3_cell_nbank_cons_orgnum']].max(axis=1)

#互金查询机构数量
apy['last_m3_id_fintech_org_num'] = apy[['ae_m3_id_nbank_top_orgnum','ae_m3_id_nbank_sloan_orgnum','ae_m3_id_nbank_trans_orgnum']].sum(axis=1)
apy['last_m3_phone_fintech_org_num'] = apy[['ae_m3_cell_nbank_top_orgnum','ae_m3_cell_nbank_sloan_orgnum','ae_m3_cell_nbank_trans_orgnum']].sum(axis=1)
apy['last_m3_fintech_org_num'] = apy[['last_m3_id_fintech_org_num','last_m3_phone_fintech_org_num']].max(axis=1)
apy = apy.drop(['last_m3_id_fintech_org_num','last_m3_phone_fintech_org_num'], axis=1)

#汽车金融查询机构数量
apy['last_m3_carloans_org_num'] = apy[['ae_m3_id_nbank_autofin_orgnum','ae_m3_cell_nbank_autofin_orgnum']].max(axis=1)

#其他金融查询机构数量
apy['last_m3_id_other_org_num'] = apy[['ae_m3_id_nbank_else_rel_orgnum','ae_m3_id_nbank_else_cons_orgnum','ae_m3_id_nbank_else_pdl_orgnum']].sum(axis=1)
apy['last_m3_phone_other_org_num'] = apy[['ae_m3_cell_nbank_else_rel_orgnum','ae_m3_cell_nbank_else_cons_orgnum','ae_m3_cell_nbank_else_pdl_orgnum']].sum(axis=1)
apy['last_m3_other_org_num'] = apy[['last_m3_id_other_org_num','last_m3_phone_other_org_num']].max(axis=1)
apy = apy.drop(['last_m3_id_other_org_num','last_m3_phone_other_org_num'], axis=1)




##======================================================================================================================================================##
##--m6
#银行查询次数
apy['last_m6_id_bank_query_num'] = apy[['ae_m6_id_bank_national_allnum','ae_m6_id_bank_region_allnum']].sum(axis=1)
apy['last_m6_phone_bank_query_num'] = apy[['ae_m6_cell_bank_national_allnum','ae_m6_cell_bank_region_allnum']].sum(axis=1)
apy['last_m6_bank_query_num'] = apy[['last_m6_id_bank_query_num','last_m6_phone_bank_query_num']].max(axis=1)
apy = apy.drop(['last_m6_id_bank_query_num','last_m6_phone_bank_query_num'], axis=1)

#保险查询次数
apy['last_m6_ins_query_num'] = apy[['ae_m6_id_nbank_ins_allnum','ae_m6_cell_nbank_ins_allnum']].max(axis=1)

#消金查询次数
apy['last_m6_cons_query_num'] = apy[['ae_m6_id_nbank_cons_allnum','ae_m6_cell_nbank_cons_allnum']].max(axis=1)

#互金查询次数
apy['last_m6_id_fintech_query_num'] = apy[['ae_m6_id_nbank_top_allnum','ae_m6_id_nbank_sloan_allnum','ae_m6_id_nbank_trans_allnum']].sum(axis=1)
apy['last_m6_phone_fintech_query_num'] = apy[['ae_m6_cell_nbank_top_allnum','ae_m6_cell_nbank_sloan_allnum','ae_m6_cell_nbank_trans_allnum']].sum(axis=1)
apy['last_m6_fintech_query_num'] = apy[['last_m6_id_fintech_query_num','last_m6_phone_fintech_query_num']].max(axis=1)
apy = apy.drop(['last_m6_id_fintech_query_num','last_m6_phone_fintech_query_num'], axis=1)

#汽车金融查询次数
apy['last_m6_carloans_query_num'] = apy[['ae_m6_id_nbank_autofin_allnum','ae_m6_cell_nbank_autofin_allnum']].max(axis=1)

#其他金融查询次数
apy['last_m6_id_other_query_num'] = apy[['ae_m6_id_nbank_else_rel_allnum','ae_m6_id_nbank_else_cons_allnum','ae_m6_id_nbank_else_pdl_allnum']].sum(axis=1)
apy['last_m6_phone_other_query_num'] = apy[['ae_m6_cell_nbank_else_rel_allnum','ae_m6_cell_nbank_else_cons_allnum','ae_m6_cell_nbank_else_pdl_allnum']].sum(axis=1)
apy['last_m6_other_query_num'] = apy[['last_m6_id_other_query_num','last_m6_phone_other_query_num']].max(axis=1)
apy = apy.drop(['last_m6_id_other_query_num','last_m6_phone_other_query_num'], axis=1)

#银行查询机构数量
apy['last_m6_id_bank_org_num'] = apy[['ae_m6_id_bank_national_orgnum','ae_m6_id_bank_region_orgnum']].sum(axis=1)
apy['last_m6_phone_bank_org_num'] = apy[['ae_m6_cell_bank_national_orgnum','ae_m6_cell_bank_region_orgnum']].sum(axis=1)
apy['last_m6_bank_org_num'] = apy[['last_m6_id_bank_org_num','last_m6_phone_bank_org_num']].max(axis=1)
apy = apy.drop(['last_m6_id_bank_org_num','last_m6_phone_bank_org_num'], axis=1)

#保险查询机构数量
apy['last_m6_ins_org_num'] = apy[['ae_m6_id_nbank_ins_orgnum','ae_m6_cell_nbank_ins_orgnum']].max(axis=1)

#消金查询机构数量
apy['last_m6_cons_org_num'] = apy[['ae_m6_id_nbank_cons_orgnum','ae_m6_cell_nbank_cons_orgnum']].max(axis=1)

#互金查询机构数量
apy['last_m6_id_fintech_org_num'] = apy[['ae_m6_id_nbank_top_orgnum','ae_m6_id_nbank_sloan_orgnum','ae_m6_id_nbank_trans_orgnum']].sum(axis=1)
apy['last_m6_phone_fintech_org_num'] = apy[['ae_m6_cell_nbank_top_orgnum','ae_m6_cell_nbank_sloan_orgnum','ae_m6_cell_nbank_trans_orgnum']].sum(axis=1)
apy['last_m6_fintech_org_num'] = apy[['last_m6_id_fintech_org_num','last_m6_phone_fintech_org_num']].max(axis=1)
apy = apy.drop(['last_m6_id_fintech_org_num','last_m6_phone_fintech_org_num'], axis=1)

#汽车金融查询机构数量
apy['last_m6_carloans_org_num'] = apy[['ae_m6_id_nbank_autofin_orgnum','ae_m6_cell_nbank_autofin_orgnum']].max(axis=1)

#其他金融查询机构数量
apy['last_m6_id_other_org_num'] = apy[['ae_m6_id_nbank_else_rel_orgnum','ae_m6_id_nbank_else_cons_orgnum','ae_m6_id_nbank_else_pdl_orgnum']].sum(axis=1)
apy['last_m6_phone_other_org_num'] = apy[['ae_m6_cell_nbank_else_rel_orgnum','ae_m6_cell_nbank_else_cons_orgnum','ae_m6_cell_nbank_else_pdl_orgnum']].sum(axis=1)
apy['last_m6_other_org_num'] = apy[['last_m6_id_other_org_num','last_m6_phone_other_org_num']].max(axis=1)
apy = apy.drop(['last_m6_id_other_org_num','last_m6_phone_other_org_num'], axis=1)




##======================================================================================================================================================##
##--m12
#银行查询次数
apy['last_m12_id_bank_query_num'] = apy[['ae_m12_id_bank_national_allnum','ae_m12_id_bank_region_allnum']].sum(axis=1)
apy['last_m12_phone_bank_query_num'] = apy[['ae_m12_cell_bank_national_allnum','ae_m12_cell_bank_region_allnum']].sum(axis=1)
apy['last_m12_bank_query_num'] = apy[['last_m12_id_bank_query_num','last_m12_phone_bank_query_num']].max(axis=1)
apy = apy.drop(['last_m12_id_bank_query_num','last_m12_phone_bank_query_num'], axis=1)

#保险查询次数
apy['last_m12_ins_query_num'] = apy[['ae_m12_id_nbank_ins_allnum','ae_m12_cell_nbank_ins_allnum']].max(axis=1)

#消金查询次数
apy['last_m12_cons_query_num'] = apy[['ae_m12_id_nbank_cons_allnum','ae_m12_cell_nbank_cons_allnum']].max(axis=1)

#互金查询次数
apy['last_m12_id_fintech_query_num'] = apy[['ae_m12_id_nbank_top_allnum','ae_m12_id_nbank_sloan_allnum','ae_m12_id_nbank_trans_allnum']].sum(axis=1)
apy['last_m12_phone_fintech_query_num'] = apy[['ae_m12_cell_nbank_top_allnum','ae_m12_cell_nbank_sloan_allnum','ae_m12_cell_nbank_trans_allnum']].sum(axis=1)
apy['last_m12_fintech_query_num'] = apy[['last_m12_id_fintech_query_num','last_m12_phone_fintech_query_num']].max(axis=1)
apy = apy.drop(['last_m12_id_fintech_query_num','last_m12_phone_fintech_query_num'], axis=1)

#汽车金融查询次数
apy['last_m12_carloans_query_num'] = apy[['ae_m12_id_nbank_autofin_allnum','ae_m12_cell_nbank_autofin_allnum']].max(axis=1)

#其他金融查询次数
apy['last_m12_id_other_query_num'] = apy[['ae_m12_id_nbank_else_rel_allnum','ae_m12_id_nbank_else_cons_allnum','ae_m12_id_nbank_else_pdl_allnum']].sum(axis=1)
apy['last_m12_phone_other_query_num'] = apy[['ae_m12_cell_nbank_else_rel_allnum','ae_m12_cell_nbank_else_cons_allnum','ae_m12_cell_nbank_else_pdl_allnum']].sum(axis=1)
apy['last_m12_other_query_num'] = apy[['last_m12_id_other_query_num','last_m12_phone_other_query_num']].max(axis=1)
apy = apy.drop(['last_m12_id_other_query_num','last_m12_phone_other_query_num'], axis=1)

#银行查询机构数量
apy['last_m12_id_bank_org_num'] = apy[['ae_m12_id_bank_national_orgnum','ae_m12_id_bank_region_orgnum']].sum(axis=1)
apy['last_m12_phone_bank_org_num'] = apy[['ae_m12_cell_bank_national_orgnum','ae_m12_cell_bank_region_orgnum']].sum(axis=1)
apy['last_m12_bank_org_num'] = apy[['last_m12_id_bank_org_num','last_m12_phone_bank_org_num']].max(axis=1)
apy = apy.drop(['last_m12_id_bank_org_num','last_m12_phone_bank_org_num'], axis=1)

#保险查询机构数量
apy['last_m12_ins_org_num'] = apy[['ae_m12_id_nbank_ins_orgnum','ae_m12_cell_nbank_ins_orgnum']].max(axis=1)

#消金查询机构数量
apy['last_m12_cons_org_num'] = apy[['ae_m12_id_nbank_cons_orgnum','ae_m12_cell_nbank_cons_orgnum']].max(axis=1)

#互金查询机构数量
apy['last_m12_id_fintech_org_num'] = apy[['ae_m12_id_nbank_top_orgnum','ae_m12_id_nbank_sloan_orgnum','ae_m12_id_nbank_trans_orgnum']].sum(axis=1)
apy['last_m12_phone_fintech_org_num'] = apy[['ae_m12_cell_nbank_top_orgnum','ae_m12_cell_nbank_sloan_orgnum','ae_m12_cell_nbank_trans_orgnum']].sum(axis=1)
apy['last_m12_fintech_org_num'] = apy[['last_m12_id_fintech_org_num','last_m12_phone_fintech_org_num']].max(axis=1)
apy = apy.drop(['last_m12_id_fintech_org_num','last_m12_phone_fintech_org_num'], axis=1)

#汽车金融查询机构数量
apy['last_m12_carloans_org_num'] = apy[['ae_m12_id_nbank_autofin_orgnum','ae_m12_cell_nbank_autofin_orgnum']].max(axis=1)

#其他金融查询机构数量
apy['last_m12_id_other_org_num'] = apy[['ae_m12_id_nbank_else_rel_orgnum','ae_m12_id_nbank_else_cons_orgnum','ae_m12_id_nbank_else_pdl_orgnum']].sum(axis=1)
apy['last_m12_phone_other_org_num'] = apy[['ae_m12_cell_nbank_else_rel_orgnum','ae_m12_cell_nbank_else_cons_orgnum','ae_m12_cell_nbank_else_pdl_orgnum']].sum(axis=1)
apy['last_m12_other_org_num'] = apy[['last_m12_id_other_org_num','last_m12_phone_other_org_num']].max(axis=1)
apy = apy.drop(['last_m12_id_other_org_num','last_m12_phone_other_org_num'], axis=1)





keep_list = [
             'name',
             'id',
             'cell',
             'user_date',
             'last_m1_bank_query_num',
             'last_m1_ins_query_num',
             'last_m1_cons_query_num',
             'last_m1_fintech_query_num',
             'last_m1_carloans_query_num',
             'last_m1_other_query_num',
             'last_m1_bank_org_num',
             'last_m1_ins_org_num',
             'last_m1_cons_org_num',
             'last_m1_fintech_org_num',
             'last_m1_carloans_org_num',
             'last_m1_other_org_num',
             'last_m3_bank_query_num',
             'last_m3_ins_query_num',
             'last_m3_cons_query_num',
             'last_m3_fintech_query_num',
             'last_m3_carloans_query_num',
             'last_m3_other_query_num',
             'last_m3_bank_org_num',
             'last_m3_ins_org_num',
             'last_m3_cons_org_num',
             'last_m3_fintech_org_num',
             'last_m3_carloans_org_num',
             'last_m3_other_org_num',
             'last_m6_bank_query_num',
             'last_m6_ins_query_num',
             'last_m6_cons_query_num',
             'last_m6_fintech_query_num',
             'last_m6_carloans_query_num',
             'last_m6_other_query_num',
             'last_m6_bank_org_num',
             'last_m6_ins_org_num',
             'last_m6_cons_org_num',
             'last_m6_fintech_org_num',
             'last_m6_carloans_org_num',
             'last_m6_other_org_num',
             'last_m12_bank_query_num',
             'last_m12_ins_query_num',
             'last_m12_cons_query_num',
             'last_m12_fintech_query_num',
             'last_m12_carloans_query_num',
             'last_m12_other_query_num',
             'last_m12_bank_org_num',
             'last_m12_ins_org_num',
             'last_m12_cons_org_num',
             'last_m12_fintech_org_num',
             'last_m12_carloans_org_num',
             'last_m12_other_org_num']


br_querys = apy[keep_list].fillna(0)

br_querys = br_querys.rename(columns = {'id':'id_no', 'user_date':'apply_time'})
br_querys = br_querys.drop(['cell','name'], axis=1)

br_querys = br_querys.drop_duplicates()
dup_ser = br_querys.groupby(['id_no','apply_time'])['id_no'].count()
br_querys = br_querys[~br_querys['id_no'].isin(dup_ser[dup_ser>1].unstack().index.tolist())]



###################################################### 探知

df_tz_gz = load_from_csv(filePath = r'F:\data\三方提测样本返回\TZ_gongZhai.csv', charTrans=False)
df_tz_gz = df_tz_gz.drop_duplicates()
df_tz_gz = df_tz_gz.rename(columns = {'mobile':'cell_id', 'idCard':'id_no', 'backTime':'apply_time',
                                     '风险原因':'tz_risk_reason',
                                     '风险等级':'tz_risk_rank',
                                     '近一个月注册平台数':'tz_last1_reg_plat_cnt',
                                     '近一个月注册平台类型(银行)':'tz_last1_reg_plat_bank_cnt',
                                     '近一个月注册平台类型(保险)':'tz_last1_reg_plat_insure_cnt',
                                     '近一个月注册平台类型(互联网金融)':'tz_last1_reg_plat_itfin_cnt',
                                     '近一个月注册平台类型(消费金融)':'tz_last1_reg_plat_consume_cnt',
                                     '近一个月注册平台类型(其他金融)':'tz_last1_reg_plat_other_cnt',
                                     '近一个月申请平台数':'tz_last1_apl_plat_cnt',
                                     '近一个月申请次数':'tz_last1_apl_cnt',
                                     '近一个月申请平台类型(银行)':'tz_last1_apl_plat_bank_cnt',
                                     '近一个月申请平台类型(保险)':'tz_last1_apl_plat_insure_cnt',
                                     '近一个月申请平台类型(互联网金融)':'tz_last1_apl_plat_itfin_cnt',
                                     '近一个月申请平台类型(消费金融)':'tz_last1_apl_plat_consume_cnt',
                                     '近一个月申请平台类型(其他金融)':'tz_last1_apl_plat_other_cnt',
                                     '近一个月使用平台数':'tz_last1_used_plat_cnt',
                                     '近一个月使用次数':'tz_last1_used_cnt',
                                     '近一个月使用平台类型(银行)':'tz_last1_used_plat_bank_cnt',
                                     '近一个月使用平台类型(保险)':'tz_last1_used_plat_insure_cnt',
                                     '近一个月使用平台类型(互联网金融)':'tz_last1_used_plat_itfin_cnt',
                                     '近一个月使用平台类型(消费金融)':'tz_last1_used_plat_consume_cnt',
                                     '近一个月使用平台类型(其他金融)':'tz_last1_used_plat_other_cnt',
                                     '近三个月注册平台数':'tz_last3_reg_plat_cnt',
                                     '近三个月注册平台类型(银行)':'tz_last3_reg_plat_bank_cnt',
                                     '近三个月注册平台类型(保险)':'tz_last3_reg_plat_insure_cnt',
                                     '近三个月注册平台类型(互联网金融)':'tz_last3_reg_plat_itfin_cnt',
                                     '近三个月注册平台类型(消费金融)':'tz_last3_reg_plat_consume_cnt',
                                     '近三个月注册平台类型(其他金融)':'tz_last3_reg_plat_other_cnt',
                                     '近三个月申请平台数':'tz_last3_apl_plat_cnt',
                                     '近三个月申请次数':'tz_last3_apl_cnt',
                                     '近三个月申请平台类型(银行)':'tz_last3_apl_plat_bank_cnt',
                                     '近三个月申请平台类型(保险)':'tz_last3_apl_plat_insure_cnt',
                                     '近三个月申请平台类型(互联网金融)':'tz_last3_apl_plat_itfin_cnt',
                                     '近三个月申请平台类型(消费金融)':'tz_last3_apl_plat_consume_cnt',
                                     '近三个月申请平台类型(其他金融)':'tz_last3_apl_plat_other_cnt',
                                     '近三个月使用平台数':'tz_last3_used_plat_cnt',
                                     '近三个月使用次数':'tz_last3_used_cnt',
                                     '近三个月使用平台类型(银行)':'tz_last3_used_plat_bank_cnt',
                                     '近三个月使用平台类型(保险)':'tz_last3_used_plat_insure_cnt',
                                     '近三个月使用平台类型(互联网金融)':'tz_last3_used_plat_itfin_cnt',
                                     '近三个月使用平台类型(消费金融)':'tz_last3_used_plat_consume_cnt',
                                     '近三个月使用平台类型(其他金融)':'tz_last3_used_plat_other_cnt',
                                     '近六个月注册平台数':'tz_last6_reg_plat_cnt',
                                     '近六个月注册平台类型(银行)':'tz_last6_reg_plat_bank_cnt',
                                     '近六个月注册平台类型(保险)':'tz_last6_reg_plat_insure_cnt',
                                     '近六个月注册平台类型(互联网金融)':'tz_last6_reg_plat_itfin_cnt',
                                     '近六个月注册平台类型(消费金融)':'tz_last6_reg_plat_consume_cnt',
                                     '近六个月注册平台类型(其他金融)':'tz_last6_reg_plat_other_cnt',
                                     '近六个月申请平台数':'tz_last6_apl_plat_cnt',
                                     '近六个月申请次数':'tz_last6_apl_cnt',
                                     '近六个月申请平台类型(银行)':'tz_last6_apl_plat_bank_cnt',
                                     '近六个月申请平台类型(保险)':'tz_last6_apl_plat_insure_cnt',
                                     '近六个月申请平台类型(互联网金融)':'tz_last6_apl_plat_itfin_cnt',
                                     '近六个月申请平台类型(消费金融)':'tz_last6_apl_plat_consume_cnt',
                                     '近六个月申请平台类型(其他金融)':'tz_last6_apl_plat_other_cnt',
                                     '近六个月使用平台数':'tz_last6_used_plat_cnt',
                                     '近六个月使用次数':'tz_last6_used_cnt',
                                     '近六个月使用平台类型(银行)':'tz_last6_used_plat_bank_cnt',
                                     '近六个月使用平台类型(保险)':'tz_last6_used_plat_insure_cnt',
                                     '近六个月使用平台类型(互联网金融)':'tz_last6_used_plat_itfin_cnt',
                                     '近六个月使用平台类型(消费金融)':'tz_last6_used_plat_consume_cnt',
                                     '近六个月使用平台类型(其他金融)':'tz_last6_used_plat_other_cnt',
                                     '近十二个月注册平台数':'tz_last12_reg_plat_cnt',
                                     '近十二个月注册平台类型(银行)':'tz_last12_reg_plat_bank_cnt',
                                     '近十二个月注册平台类型(保险)':'tz_last12_reg_plat_insure_cnt',
                                     '近十二个月注册平台类型(互联网金融)':'tz_last12_reg_plat_itfin_cnt',
                                     '近十二个月注册平台类型(消费金融)':'tz_last12_reg_plat_consume_cnt',
                                     '近十二个月注册平台类型(其他金融)':'tz_last12_reg_plat_other_cnt',
                                     '近十二个月申请平台数':'tz_last12_apl_plat_cnt',
                                     '近十二个月申请次数':'tz_last12_apl_cnt',
                                     '近十二个月申请平台类型(银行)':'tz_last12_apl_plat_bank_cnt',
                                     '近十二个月申请平台类型(保险)':'tz_last12_apl_plat_insure_cnt',
                                     '近十二个月申请平台类型(互联网金融)':'tz_last12_apl_plat_itfin_cnt',
                                     '近十二个月申请平台类型(消费金融)':'tz_last12_apl_plat_consume_cnt',
                                     '近十二个月申请平台类型(其他金融)':'tz_last12_apl_plat_other_cnt',
                                     '近十二个月使用平台数':'tz_last12_used_plat_cnt',
                                     '近十二个月使用次数':'tz_last12_used_cnt',
                                     '近十二个月使用平台类型(银行)':'tz_last12_used_plat_bank_cnt',
                                     '近十二个月使用平台类型(保险)':'tz_last12_used_plat_insure_cnt',
                                     '近十二个月使用平台类型(互联网金融)':'tz_last12_used_plat_itfin_cnt',
                                     '近十二个月使用平台类型(消费金融)':'tz_last12_used_plat_consume_cnt',
                                     '近十二个月使用平台类型(其他金融)':'tz_last12_used_plat_other_cnt',
                                     '近二十四个月注册平台数':'tz_last24_reg_plat_cnt',
                                     '近二十四个月注册平台类型(银行)':'tz_last24_reg_plat_bank_cnt',
                                     '近二十四个月注册平台类型(保险)':'tz_last24_reg_plat_insure_cnt',
                                     '近二十四个月注册平台类型(互联网金融)':'tz_last24_reg_plat_itfin_cnt',
                                     '近二十四个月注册平台类型(消费金融)':'tz_last24_reg_plat_consume_cnt',
                                     '近二十四个月注册平台类型(其他金融)':'tz_last24_reg_plat_other_cnt',
                                     '近二十四个月申请平台数':'tz_last24_apl_plat_cnt',
                                     '近二十四个月申请次数':'tz_last24_apl_cnt',
                                     '近二十四个月申请平台类型(银行)':'tz_last24_apl_plat_bank_cnt',
                                     '近二十四个月申请平台类型(保险)':'tz_last24_apl_plat_insure_cnt',
                                     '近二十四个月申请平台类型(互联网金融)':'tz_last24_apl_plat_itfin_cnt',
                                     '近二十四个月申请平台类型(消费金融)':'tz_last24_apl_plat_consume_cnt',
                                     '近二十四个月申请平台类型(其他金融)':'tz_last24_apl_plat_other_cnt',
                                     '近二十四个月使用平台数':'tz_last24_used_plat_cnt',
                                     '近二十四个月使用次数':'tz_last24_used_cnt',
                                     '近二十四个月使用平台类型(银行)':'tz_last24_used_plat_bank_cnt',
                                     '近二十四个月使用平台类型(保险)':'tz_last24_used_plat_insure_cnt',
                                     '近二十四个月使用平台类型(互联网金融)':'tz_last24_used_plat_itfin_cnt',
                                     '近二十四个月使用平台类型(消费金融)':'tz_last24_used_plat_consume_cnt',
                                     '近二十四个月使用平台类型(其他金融)':'tz_last24_used_plat_other_cnt',
                                     '长周期共债评分':'tz_long_debt_score',
                                     '短周期共债评分':'tz_short_debt_score'
                                    })


VarList = ['id_no', 'apply_time',
            'tz_last1_reg_plat_bank_cnt', 'tz_last1_reg_plat_insure_cnt', 'tz_last1_reg_plat_itfin_cnt', 'tz_last1_reg_plat_consume_cnt', 'tz_last1_reg_plat_other_cnt',
            'tz_last1_apl_plat_bank_cnt', 'tz_last1_apl_plat_insure_cnt', 'tz_last1_apl_plat_itfin_cnt', 'tz_last1_apl_plat_consume_cnt', 'tz_last1_apl_plat_other_cnt',
            'tz_last3_reg_plat_bank_cnt', 'tz_last3_reg_plat_insure_cnt', 'tz_last3_reg_plat_itfin_cnt', 'tz_last3_reg_plat_consume_cnt', 'tz_last3_reg_plat_other_cnt',
            'tz_last3_apl_plat_bank_cnt', 'tz_last3_apl_plat_insure_cnt', 'tz_last3_apl_plat_itfin_cnt', 'tz_last3_apl_plat_consume_cnt', 'tz_last3_apl_plat_other_cnt',
            'tz_last6_reg_plat_bank_cnt', 'tz_last6_reg_plat_insure_cnt', 'tz_last6_reg_plat_itfin_cnt', 'tz_last6_reg_plat_consume_cnt', 'tz_last6_reg_plat_other_cnt',
            'tz_last6_apl_plat_bank_cnt', 'tz_last6_apl_plat_insure_cnt', 'tz_last6_apl_plat_itfin_cnt', 'tz_last6_apl_plat_consume_cnt', 'tz_last6_apl_plat_other_cnt',
            'tz_last12_reg_plat_bank_cnt', 'tz_last12_reg_plat_insure_cnt', 'tz_last12_reg_plat_itfin_cnt', 'tz_last12_reg_plat_consume_cnt', 'tz_last12_reg_plat_other_cnt',
            'tz_last12_apl_plat_bank_cnt', 'tz_last12_apl_plat_insure_cnt', 'tz_last12_apl_plat_itfin_cnt', 'tz_last12_apl_plat_consume_cnt', 'tz_last12_apl_plat_other_cnt']
tz_plat = df_tz_gz[VarList]

tz_plat = tz_plat.drop_duplicates()
dup_ser = tz_plat.groupby(['id_no','apply_time'])['id_no'].count()
tz_plat = tz_plat[~tz_plat['id_no'].isin(dup_ser[dup_ser>1].unstack().index.tolist())]
tz_plat['id_no'] = tz_plat['id_no'].map(lambda x: x.upper())


## target
df_target = pd.read_csv(r'F:\data\三方提测样本返回\final_encrypt_sample.csv')
df_target = df_target[['request_id', 'date', 'encode_id_no', 'if_loan', 'if_7p', 'if_now_m1p']]
df_target = df_target.rename(columns = {'date':'apply_time', 'encode_id_no':'id_no'})

## 百融和探知数据合并
df_table = df_target.merge(br_querys, on=['id_no','apply_time'], how='left')
df_table = df_table.merge(tz_plat, on=['id_no','apply_time'], how='left')

## 样本选择
df_analy = df_table[(df_table['apply_time']<='2020-06-13') & (~df_table['if_now_m1p'].isnull())]



##-------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------##
##计算变量IV值


var_ls = [
         'last_m1_bank_query_num',
         'last_m1_ins_query_num',
         'last_m1_cons_query_num',
         'last_m1_fintech_query_num',
         'last_m1_carloans_query_num',
         'last_m1_other_query_num',
         'last_m1_bank_org_num',
         'last_m1_ins_org_num',
         'last_m1_cons_org_num',
         'last_m1_fintech_org_num',
         'last_m1_carloans_org_num',
         'last_m1_other_org_num',
         'last_m3_bank_query_num',
         'last_m3_ins_query_num',
         'last_m3_cons_query_num',
         'last_m3_fintech_query_num',
         'last_m3_carloans_query_num',
         'last_m3_other_query_num',
         'last_m3_bank_org_num',
         'last_m3_ins_org_num',
         'last_m3_cons_org_num',
         'last_m3_fintech_org_num',
         'last_m3_carloans_org_num',
         'last_m3_other_org_num',
         'last_m6_bank_query_num',
         'last_m6_ins_query_num',
         'last_m6_cons_query_num',
         'last_m6_fintech_query_num',
         'last_m6_carloans_query_num',
         'last_m6_other_query_num',
         'last_m6_bank_org_num',
         'last_m6_ins_org_num',
         'last_m6_cons_org_num',
         'last_m6_fintech_org_num',
         'last_m6_carloans_org_num',
         'last_m6_other_org_num',
         'last_m12_bank_query_num',
         'last_m12_ins_query_num',
         'last_m12_cons_query_num',
         'last_m12_fintech_query_num',
         'last_m12_carloans_query_num',
         'last_m12_other_query_num',
         'last_m12_bank_org_num',
         'last_m12_ins_org_num',
         'last_m12_cons_org_num',
         'last_m12_fintech_org_num',
         'last_m12_carloans_org_num',
         'last_m12_other_org_num',
         'tz_last1_reg_plat_bank_cnt',
         'tz_last1_reg_plat_insure_cnt',
         'tz_last1_reg_plat_itfin_cnt',
         'tz_last1_reg_plat_consume_cnt',
         'tz_last1_reg_plat_other_cnt',
         'tz_last1_apl_plat_bank_cnt',
         'tz_last1_apl_plat_insure_cnt',
         'tz_last1_apl_plat_itfin_cnt',
         'tz_last1_apl_plat_consume_cnt',
         'tz_last1_apl_plat_other_cnt',
         'tz_last3_reg_plat_bank_cnt',
         'tz_last3_reg_plat_insure_cnt',
         'tz_last3_reg_plat_itfin_cnt',
         'tz_last3_reg_plat_consume_cnt',
         'tz_last3_reg_plat_other_cnt',
         'tz_last3_apl_plat_bank_cnt',
         'tz_last3_apl_plat_insure_cnt',
         'tz_last3_apl_plat_itfin_cnt',
         'tz_last3_apl_plat_consume_cnt',
         'tz_last3_apl_plat_other_cnt',
         'tz_last6_reg_plat_bank_cnt',
         'tz_last6_reg_plat_insure_cnt',
         'tz_last6_reg_plat_itfin_cnt',
         'tz_last6_reg_plat_consume_cnt',
         'tz_last6_reg_plat_other_cnt',
         'tz_last6_apl_plat_bank_cnt',
         'tz_last6_apl_plat_insure_cnt',
         'tz_last6_apl_plat_itfin_cnt',
         'tz_last6_apl_plat_consume_cnt',
         'tz_last6_apl_plat_other_cnt',
         'tz_last12_reg_plat_bank_cnt',
         'tz_last12_reg_plat_insure_cnt',
         'tz_last12_reg_plat_itfin_cnt',
         'tz_last12_reg_plat_consume_cnt',
         'tz_last12_reg_plat_other_cnt',
         'tz_last12_apl_plat_bank_cnt',
         'tz_last12_apl_plat_insure_cnt',
         'tz_last12_apl_plat_itfin_cnt',
         'tz_last12_apl_plat_consume_cnt',
         'tz_last12_apl_plat_other_cnt']
oldvar_iv_df = iv_df_auto_calculate(inDf = df_analy,
                     xVarList = var_ls,
                     yVar = 'if_now_m1p')





##--------------------------------------------------------------------------------------##
##变量预测

def var_br_org_repair(var_x, var_y, df_analy):
    '''
    var_x = ['last_m1_bank_org_num']
    var_y = ['last_m1_bank_query_num']
    df_analy = df_analy
    
    ## 缺失指标修复过程
    1、 if x=0 then y=0
    2、 if x>0 then
            y_pred = intercept + x * coef 
            y = floor(y_tmp / y_pred_avg)
    '''
    
    var_df = pd.DataFrame({'x_vary': var_x,
                           'y_vary': var_y})
    var_predict_coef = pd.DataFrame(columns=['x_var','y_var','intercept','coef','y_pred_avg'])
    for indexs in range(var_df.shape[0]):   
        var_predict_ls = var_df.iloc[indexs].values.tolist() 
        
        df = df_analy[df_analy[var_predict_ls[0]]>0]
        if df.shape[0]>0:
            lm = ols('{} ~ {}'.format(var_predict_ls[1], var_predict_ls[0]), data=df).fit()
            model_rst = lm.summary()
        else:
            lm = ols('{} ~ {}'.format(var_predict_ls[1], var_predict_ls[0]), data=df_analy).fit()
            model_rst = lm.summary()
            
        
        intercept = float(model_rst.tables[1].data[1][1])
        var_parm = float(model_rst.tables[1].data[2][1])
        
        print('\n')
        print('=='*30)
        print(var_predict_ls[1])
        if var_parm != 0:
            new_ser = intercept + df[var_predict_ls[0]]*var_parm
            new_ser = (new_ser/(new_ser.mean())).map(lambda x: math.floor(x))
            new_ser_avg = new_ser.mean()
        
            print('Actual:', '\n', df[var_predict_ls[1]].value_counts().sort_index())
            print('Predict:', '\n', new_ser.value_counts().sort_index())
        else :
            new_ser_avg = np.nan
        
        var_predict_coef.loc[indexs] = [var_predict_ls[0], var_predict_ls[1], intercept, var_parm, new_ser_avg]
        
    return var_predict_coef


###注册平台数修复规则

var_x = ['tz_last1_apl_plat_bank_cnt', 'tz_last1_apl_plat_insure_cnt', 'tz_last1_apl_plat_itfin_cnt', 'tz_last1_apl_plat_consume_cnt', 
        'tz_last1_apl_plat_other_cnt', 'tz_last3_apl_plat_bank_cnt', 'tz_last3_apl_plat_insure_cnt', 'tz_last3_apl_plat_itfin_cnt', 
        'tz_last3_apl_plat_consume_cnt', 'tz_last3_apl_plat_other_cnt', 'tz_last6_apl_plat_bank_cnt', 'tz_last6_apl_plat_insure_cnt', 
        'tz_last6_apl_plat_itfin_cnt', 'tz_last6_apl_plat_consume_cnt', 'tz_last6_apl_plat_other_cnt', 'tz_last12_apl_plat_bank_cnt', 
        'tz_last12_apl_plat_insure_cnt', 'tz_last12_apl_plat_itfin_cnt', 'tz_last12_apl_plat_consume_cnt', 'tz_last12_apl_plat_other_cnt' ]
var_y = ['tz_last1_reg_plat_bank_cnt', 'tz_last1_reg_plat_insure_cnt', 'tz_last1_reg_plat_itfin_cnt', 'tz_last1_reg_plat_consume_cnt', 
        'tz_last1_reg_plat_other_cnt', 'tz_last3_reg_plat_bank_cnt', 'tz_last3_reg_plat_insure_cnt', 'tz_last3_reg_plat_itfin_cnt', 
        'tz_last3_reg_plat_consume_cnt', 'tz_last3_reg_plat_other_cnt', 'tz_last6_reg_plat_bank_cnt', 'tz_last6_reg_plat_insure_cnt', 
        'tz_last6_reg_plat_itfin_cnt', 'tz_last6_reg_plat_consume_cnt', 'tz_last6_reg_plat_other_cnt', 'tz_last12_reg_plat_bank_cnt', 
        'tz_last12_reg_plat_insure_cnt', 'tz_last12_reg_plat_itfin_cnt', 'tz_last12_reg_plat_consume_cnt', 'tz_last12_reg_plat_other_cnt']
repair_reg_org = var_br_org_repair(var_x=var_x, var_y=var_y, df_analy=df_analy)


var_x = ['tz_last1_apl_plat_bank_cnt', 'tz_last1_apl_plat_insure_cnt', 'tz_last1_apl_plat_itfin_cnt', 'tz_last1_apl_plat_consume_cnt', 
        'tz_last1_apl_plat_other_cnt', 'tz_last3_apl_plat_bank_cnt', 'tz_last3_apl_plat_insure_cnt', 'tz_last3_apl_plat_itfin_cnt', 
        'tz_last3_apl_plat_consume_cnt', 'tz_last3_apl_plat_other_cnt', 'tz_last6_apl_plat_bank_cnt', 'tz_last6_apl_plat_insure_cnt', 
        'tz_last6_apl_plat_itfin_cnt', 'tz_last6_apl_plat_consume_cnt', 'tz_last6_apl_plat_other_cnt', 'tz_last12_apl_plat_bank_cnt', 
        'tz_last12_apl_plat_insure_cnt', 'tz_last12_apl_plat_itfin_cnt', 'tz_last12_apl_plat_consume_cnt', 'tz_last12_apl_plat_other_cnt' ]
var_y = ['tz_last1_reg_plat_bank_cnt', 'tz_last1_reg_plat_insure_cnt', 'tz_last1_reg_plat_itfin_cnt', 'tz_last1_reg_plat_consume_cnt', 
        'tz_last1_reg_plat_other_cnt', 'tz_last3_reg_plat_bank_cnt', 'tz_last3_reg_plat_insure_cnt', 'tz_last3_reg_plat_itfin_cnt', 
        'tz_last3_reg_plat_consume_cnt', 'tz_last3_reg_plat_other_cnt', 'tz_last6_reg_plat_bank_cnt', 'tz_last6_reg_plat_insure_cnt', 
        'tz_last6_reg_plat_itfin_cnt', 'tz_last6_reg_plat_consume_cnt', 'tz_last6_reg_plat_other_cnt', 'tz_last12_reg_plat_bank_cnt', 
        'tz_last12_reg_plat_insure_cnt', 'tz_last12_reg_plat_itfin_cnt', 'tz_last12_reg_plat_consume_cnt', 'tz_last12_reg_plat_other_cnt']
var_pred = ['last_m1_bank_org_num', 'last_m1_ins_org_num', 'last_m1_fintech_org_num', 'last_m1_cons_org_num', 'last_m1_other_org_num', 
        'last_m3_bank_org_num', 'last_m3_ins_org_num', 'last_m3_fintech_org_num', 'last_m3_cons_org_num', 'last_m3_other_org_num', 
        'last_m6_bank_org_num', 'last_m6_ins_org_num', 'last_m6_fintech_org_num', 'last_m6_cons_org_num', 'last_m6_other_org_num', 
        'last_m12_bank_org_num', 'last_m12_ins_org_num', 'last_m12_fintech_org_num', 'last_m12_cons_org_num', 'last_m12_other_org_num']
repair_pred_reg_org = pd.DataFrame({'x_var': var_x, 'y_var': var_y, 'to_pred': var_pred})
repair_pred_reg_org = repair_pred_reg_org.merge(repair_reg_org, on=['x_var','y_var'], how='left')


def var_pred_func(x, intercept, coef, y_avg):
    if pd.isnull(x):
        return np.nan
    elif x>0:
        return math.floor(intercept + x*coef)
    else:
        return 0

for rows in range(repair_pred_reg_org.shape[0]):
    row_value = repair_pred_reg_org.iloc[rows].to_dict() 
    print(row_value['to_pred'])
    if row_value['coef'] == 0:
        df_analy['{}_pred'.format(row_value['to_pred'])] = np.nan
    else:
        df_analy['{}_pred'.format(row_value['to_pred'])] = df_analy[row_value['to_pred']].apply(lambda x: var_pred_func(x, row_value['intercept'], row_value['coef'], row_value['y_pred_avg']))









##--------------------------------------------------------------------------------------------


def var_tz_apl_repair(var_x, var_y, df_analy):
    '''
    var_x = ['last_m12_other_org_num']
    var_y = ['last_m12_other_query_num']
    df_analy = df_analy
    
    ## 缺失指标修复过程
    1、 if x=0 then y=0
    2、 if x>0 then
            y_pred = intercept + x * coef 
            y = floor(y_tmp)
    '''
    
    var_df = pd.DataFrame({'x_vary': var_x,
                           'y_vary': var_y})
    var_predict_coef = pd.DataFrame(columns=['x_var','y_var','intercept','coef','y_pred_avg'])
    for indexs in range(var_df.shape[0]):   
        var_predict_ls = var_df.iloc[indexs].values.tolist() 
        
        df = df_analy[df_analy[var_predict_ls[0]]>0]
        if df.shape[0]>0:
            lm = ols('{} ~ {}'.format(var_predict_ls[1], var_predict_ls[0]), data=df).fit()
            model_rst = lm.summary()
        else:
            lm = ols('{} ~ {}'.format(var_predict_ls[1], var_predict_ls[0]), data=df_analy).fit()
            model_rst = lm.summary()
            
        
        intercept = float(model_rst.tables[1].data[1][1])
        var_parm = float(model_rst.tables[1].data[2][1])
        
        print('\n')
        print('=='*30)
        print(var_predict_ls[1])
        if var_parm != 0:
            new_ser = intercept + df[var_predict_ls[0]]*var_parm
            #new_ser = (new_ser/(new_ser.mean())).map(lambda x: math.floor(x))
            new_ser = new_ser.map(lambda x: math.floor(x))
            new_ser_avg = new_ser.mean()
        
            print('Actual:', '\n', df[var_predict_ls[1]].value_counts().sort_index())
            print('Predict:', '\n', new_ser.value_counts().sort_index())
        else :
            new_ser_avg = np.nan
        
        var_predict_coef.loc[indexs] = [var_predict_ls[0], var_predict_ls[1], intercept, var_parm, new_ser_avg]
        
    return var_predict_coef


###申请次数规则
var_x = ['last_m1_bank_org_num', 'last_m1_ins_org_num', 'last_m1_fintech_org_num', 'last_m1_cons_org_num', 'last_m1_other_org_num', 
        'last_m3_bank_org_num', 'last_m3_ins_org_num', 'last_m3_fintech_org_num', 'last_m3_cons_org_num', 'last_m3_other_org_num', 
        'last_m6_bank_org_num', 'last_m6_ins_org_num', 'last_m6_fintech_org_num', 'last_m6_cons_org_num', 'last_m6_other_org_num', 
        'last_m12_bank_org_num', 'last_m12_ins_org_num', 'last_m12_fintech_org_num', 'last_m12_cons_org_num', 'last_m12_other_org_num']
var_y = ['last_m1_bank_query_num', 'last_m1_ins_query_num', 'last_m1_fintech_query_num', 'last_m1_cons_query_num', 
        'last_m1_other_query_num', 'last_m3_bank_query_num', 'last_m3_ins_query_num', 'last_m3_fintech_query_num', 
        'last_m3_cons_query_num', 'last_m3_other_query_num', 'last_m6_bank_query_num', 'last_m6_ins_query_num', 
        'last_m6_fintech_query_num', 'last_m6_cons_query_num', 'last_m6_other_query_num', 'last_m12_bank_query_num', 
        'last_m12_ins_query_num', 'last_m12_fintech_query_num', 'last_m12_cons_query_num', 'last_m12_other_query_num']
repair_query = var_tz_apl_repair(var_x=var_x, var_y=var_y, df_analy=df_analy)




var_x = ['last_m1_bank_org_num', 'last_m1_ins_org_num', 'last_m1_fintech_org_num', 'last_m1_cons_org_num', 'last_m1_other_org_num', 
        'last_m3_bank_org_num', 'last_m3_ins_org_num', 'last_m3_fintech_org_num', 'last_m3_cons_org_num', 'last_m3_other_org_num', 
        'last_m6_bank_org_num', 'last_m6_ins_org_num', 'last_m6_fintech_org_num', 'last_m6_cons_org_num', 'last_m6_other_org_num', 
        'last_m12_bank_org_num', 'last_m12_ins_org_num', 'last_m12_fintech_org_num', 'last_m12_cons_org_num', 'last_m12_other_org_num']
var_y = ['last_m1_bank_query_num', 'last_m1_ins_query_num', 'last_m1_fintech_query_num', 'last_m1_cons_query_num', 
        'last_m1_other_query_num', 'last_m3_bank_query_num', 'last_m3_ins_query_num', 'last_m3_fintech_query_num', 
        'last_m3_cons_query_num', 'last_m3_other_query_num', 'last_m6_bank_query_num', 'last_m6_ins_query_num', 
        'last_m6_fintech_query_num', 'last_m6_cons_query_num', 'last_m6_other_query_num', 'last_m12_bank_query_num', 
        'last_m12_ins_query_num', 'last_m12_fintech_query_num', 'last_m12_cons_query_num', 'last_m12_other_query_num']
var_pred = ['tz_last1_apl_plat_bank_cnt','tz_last1_apl_plat_insure_cnt','tz_last1_apl_plat_itfin_cnt','tz_last1_apl_plat_consume_cnt',
            'tz_last1_apl_plat_other_cnt','tz_last3_apl_plat_bank_cnt','tz_last3_apl_plat_insure_cnt','tz_last3_apl_plat_itfin_cnt',
            'tz_last3_apl_plat_consume_cnt','tz_last3_apl_plat_other_cnt','tz_last6_apl_plat_bank_cnt','tz_last6_apl_plat_insure_cnt',
            'tz_last6_apl_plat_itfin_cnt','tz_last6_apl_plat_consume_cnt','tz_last6_apl_plat_other_cnt','tz_last12_apl_plat_bank_cnt',
            'tz_last12_apl_plat_insure_cnt','tz_last12_apl_plat_itfin_cnt','tz_last12_apl_plat_consume_cnt','tz_last12_apl_plat_other_cnt']
repair_pred_reg_query = pd.DataFrame({'x_var': var_x, 'y_var': var_y, 'to_pred': var_pred})
repair_pred_reg_query = repair_pred_reg_query.merge(repair_query, on=['x_var','y_var'], how='left')


def var_pred_func(x, intercept, coef, y_avg):
    if pd.isnull(x):
        return np.nan
    elif x>0:
        if math.floor(intercept + x*coef) < 0:
            return 0
        else :
            return math.floor(intercept + x*coef)
    else:
        return 0

for rows in range(repair_pred_reg_query.shape[0]):
    row_value = repair_pred_reg_query.iloc[rows].to_dict() 
    print(row_value['to_pred'])
    if row_value['coef'] == 0:
        df_analy['{}_pred'.format(row_value['to_pred'])] = np.nan
    else:
        df_analy['{}_pred'.format(row_value['to_pred'])] = df_analy[row_value['to_pred']].apply(lambda x: var_pred_func(x, row_value['intercept'], row_value['coef'], row_value['y_pred_avg']))
        print(df_analy['{}_pred'.format(row_value['to_pred'])].value_counts().sort_index())











##--------------------------------------------------------------------------------------##
## 变量最优合并

def var_combine(var_ls1, var_ls2, df_analy):
    '''
    通过计算IV确定两个变量组合的权重
    '''
    
    def coperate(x, y, br_weights):
        if pd.isnull(x)==1 and pd.isnull(y)==1:
            return np.nan
        elif pd.isnull(x)==1 and pd.isnull(y) != 1:
            return y
        elif pd.isnull(x) != 1 and pd.isnull(y)==1:
            return x
        else :
            return math.ceil(x * br_weights + y * (1-br_weights))
            
    
    var_df = pd.DataFrame({'var1': var_ls1,
                           'var2': var_ls2})    
    var_comb_res = pd.DataFrame(columns=['var1_weight', 'var2_weight', 'IV'])
    for indexs in range(var_df.shape[0]):
        var_comb_ls = var_df.iloc[indexs].values.tolist()
        best_iv = 0
        best_weight = 0
        for br_weights in range(1,11):
            br_weights = br_weights*0.1
            df_analy['new_var'] = df_analy.apply(lambda x: coperate(x['{}'.format(var_comb_ls[0])], x['{}'.format(var_comb_ls[1])], br_weights), axis=1)         
            one_step_iv = iv_df_auto_calculate(inDf = df_analy,
                                 xVarList = ['new_var'],
                                 yVar = 'if_now_m1p')
            if best_iv < one_step_iv['IV'].values[0]:
                best_iv = one_step_iv['IV'].values[0]
                best_weight = round(br_weights,1)
         
        var_comb_res.loc[indexs] = ["{}={}".format(var_comb_ls[1], best_weight),
                                     "{}={}".format(var_comb_ls[1], 1-best_weight),
                                     best_iv]
        print('  ',var_comb_ls, ' Finished!')  
    return var_comb_res
    

    

##注册机构数合并       
var_ls1 = ['last_m1_bank_org_num_pred','last_m1_ins_org_num_pred','last_m1_fintech_org_num_pred','last_m1_cons_org_num_pred',
            'last_m1_other_org_num_pred','last_m3_bank_org_num_pred','last_m3_ins_org_num_pred','last_m3_fintech_org_num_pred',
            'last_m3_cons_org_num_pred','last_m3_other_org_num_pred','last_m6_bank_org_num_pred','last_m6_ins_org_num_pred',
            'last_m6_fintech_org_num_pred','last_m6_cons_org_num_pred','last_m6_other_org_num_pred','last_m12_bank_org_num_pred',
            'last_m12_ins_org_num_pred','last_m12_fintech_org_num_pred','last_m12_cons_org_num_pred','last_m12_other_org_num_pred']
var_ls2 = ['tz_last1_reg_plat_bank_cnt','tz_last1_reg_plat_insure_cnt','tz_last1_reg_plat_itfin_cnt','tz_last1_reg_plat_consume_cnt',
            'tz_last1_reg_plat_other_cnt','tz_last3_reg_plat_bank_cnt','tz_last3_reg_plat_insure_cnt','tz_last3_reg_plat_itfin_cnt',
            'tz_last3_reg_plat_consume_cnt','tz_last3_reg_plat_other_cnt','tz_last6_reg_plat_bank_cnt','tz_last6_reg_plat_insure_cnt',
            'tz_last6_reg_plat_itfin_cnt','tz_last6_reg_plat_consume_cnt','tz_last6_reg_plat_other_cnt','tz_last12_reg_plat_bank_cnt',
            'tz_last12_reg_plat_insure_cnt','tz_last12_reg_plat_itfin_cnt','tz_last12_reg_plat_consume_cnt','tz_last12_reg_plat_other_cnt',]
comb_reg_org = var_combine(var_ls1, var_ls2, df_analy)



##查询机构数合并       
var_ls1 = ['last_m1_bank_org_num', 'last_m1_ins_org_num', 'last_m1_fintech_org_num', 'last_m1_cons_org_num', 'last_m1_other_org_num', 
            'last_m3_bank_org_num', 'last_m3_ins_org_num', 'last_m3_fintech_org_num', 'last_m3_cons_org_num', 'last_m3_other_org_num', 
            'last_m6_bank_org_num', 'last_m6_ins_org_num', 'last_m6_fintech_org_num', 'last_m6_cons_org_num', 'last_m6_other_org_num', 
            'last_m12_bank_org_num', 'last_m12_ins_org_num', 'last_m12_fintech_org_num', 'last_m12_cons_org_num', 'last_m12_other_org_num']
var_ls2 = ['tz_last1_apl_plat_bank_cnt', 'tz_last1_apl_plat_insure_cnt', 'tz_last1_apl_plat_itfin_cnt', 'tz_last1_apl_plat_consume_cnt', 
            'tz_last1_apl_plat_other_cnt', 'tz_last3_apl_plat_bank_cnt', 'tz_last3_apl_plat_insure_cnt', 'tz_last3_apl_plat_itfin_cnt', 
            'tz_last3_apl_plat_consume_cnt', 'tz_last3_apl_plat_other_cnt', 'tz_last6_apl_plat_bank_cnt', 'tz_last6_apl_plat_insure_cnt', 
            'tz_last6_apl_plat_itfin_cnt', 'tz_last6_apl_plat_consume_cnt', 'tz_last6_apl_plat_other_cnt', 'tz_last12_apl_plat_bank_cnt', 
            'tz_last12_apl_plat_insure_cnt', 'tz_last12_apl_plat_itfin_cnt', 'tz_last12_apl_plat_consume_cnt', 'tz_last12_apl_plat_other_cnt']
comb_apply_org = var_combine(var_ls1, var_ls2, df_analy)



##查询笔数合并       
var_ls1 = ['last_m1_bank_query_num','last_m1_ins_query_num','last_m1_fintech_query_num','last_m1_cons_query_num','last_m1_other_query_num',
            'last_m3_bank_query_num','last_m3_ins_query_num','last_m3_fintech_query_num','last_m3_cons_query_num','last_m3_other_query_num',
            'last_m6_bank_query_num','last_m6_ins_query_num','last_m6_fintech_query_num','last_m6_cons_query_num','last_m6_other_query_num',
            'last_m12_bank_query_num','last_m12_ins_query_num','last_m12_fintech_query_num','last_m12_cons_query_num','last_m12_other_query_num']
var_ls2 = [ 'tz_last1_apl_plat_bank_cnt_pred', 'tz_last1_apl_plat_insure_cnt_pred', 'tz_last1_apl_plat_itfin_cnt_pred', 'tz_last1_apl_plat_consume_cnt_pred', 
             'tz_last1_apl_plat_other_cnt_pred', 'tz_last3_apl_plat_bank_cnt_pred', 'tz_last3_apl_plat_insure_cnt_pred', 'tz_last3_apl_plat_itfin_cnt_pred', 
             'tz_last3_apl_plat_consume_cnt_pred', 'tz_last3_apl_plat_other_cnt_pred', 'tz_last6_apl_plat_bank_cnt_pred', 'tz_last6_apl_plat_insure_cnt_pred', 
             'tz_last6_apl_plat_itfin_cnt_pred', 'tz_last6_apl_plat_consume_cnt_pred', 'tz_last6_apl_plat_other_cnt_pred', 'tz_last12_apl_plat_bank_cnt_pred', 
             'tz_last12_apl_plat_insure_cnt_pred', 'tz_last12_apl_plat_itfin_cnt_pred', 'tz_last12_apl_plat_consume_cnt_pred', 'tz_last12_apl_plat_other_cnt_pred']
comb_apply_cnt = var_combine(var_ls1, var_ls2, df_analy)


































