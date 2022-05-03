# -*- coding: utf-8 -*-

import pandas as pd
import datetime
import decimal
import numpy as np
import xlwt
import xlrd
from xlutils.copy import copy as xl_copy
from intervals import FloatInterval

def to_new_excel(filePath, fileName, sheetName, dataFrm):    
    '''
    Function Description:
        创建不存在的excel,并把数据框输出至excel,生成新的sheet,并命名sheet
        
    Parameters
    ----------
    filePath : 创建excel的文件夹路径，例如：filePath = 'F:\Python\AutoBuildScorecard\ModelCreditReport'
    fileName : 创建excel文件的名称
    sheetName : 创建sheet
    dataFrm : 需要输出的数据框
    
    Returns
    -------
    把结果保存至前面创建的excel中
    
    Examples
    --------
    filePath = file_path
    fileName = '变量预测力一致性计算'
    sheetName = 'var_power_psi'
    dataFrm = var_psi_df
    to_new_excel(
            filePath='F:\Python\AutoBuildScorecard\Result', 
            fileName='Summary', 
            sheetName='var_power_psi', 
            dataFrm=var_psi_df)
    
    ''' 
    
    df = dataFrm
    #创建新工作簿
    wb = xlwt.Workbook(encoding='utf-8')
    worksheet = wb.add_sheet("%s" % sheetName)
        
    if dataFrm.shape[0] > 0:
        # 首行冻结
        worksheet.panes_frozen = True
        worksheet.horz_split_pos = 1
        #worksheet.vert_split_pos = 1
        
        # 设置边框
        '''
        borders = xlwt.Borders()
        borders.left = 1
        borders.right = 1
        borders.top = 1
        borders.bottom = 1
        borders.bottom_colour = 0x3A
        style_edge1 = xlwt.XFStyle()
        style_edge1.borders = borders
        '''
        
        '''
        # 合并单元格并设置样式
        worksheet.row(0).height_mismatch = True
        worksheet.row(0).height = 20 * 25
        # 合并单元格并设置样式
        worksheet.write_merge(0, 0, 0, len(bodytitle), tabletitle, style=style1)
        worksheet.write_merge(1, 1, 0, len(bodytitle), remark)
        '''
        # 确定栏位宽度,根据字符长度来确定
        field_num = df.columns.size
        col_dict = dict()
        for i in range(0, field_num):
            if df[df.columns[i]].dtypes == 'str':
                print(df.columns[i])
                col_dict_tmp = {"%s" % df.columns[i]:max((max(df[df.columns[i]].str.len()),len(str(df.columns[i]))))}
                col_dict.update(col_dict_tmp)
            elif max((max(df[df.columns[i]].astype('str').str.len()),len(str(df.columns[i])))) > 60:
                col_dict_tmp = {"%s" % df.columns[i]:60}  
                col_dict.update(col_dict_tmp) 
            else:
                col_dict_tmp = {"%s" % df.columns[i]:max((max(df[df.columns[i]].astype('str').str.len()),len(str(df.columns[i]))))}  
                col_dict.update(col_dict_tmp)   
    
        # 设置栏位宽度，栏位宽度小于10时候采用默认宽度
        for j in range(0, field_num):
            worksheet.col(j).width = int(256 * (col_dict["%s" % df.columns[j]] + 6))
     
        # 设置行高
        #header_rowstyle = xlwt.easyxf('font:height 240;') 
        #row0 = worksheet.row(0)
        #row0.set_style(header_rowstyle)
        
        # 设置单元格格式
        header_cell_style = xlwt.easyxf('font:bold on, height 200; align: wrap yes,vert centre, horiz center; \
                                    pattern: pattern solid, fore-colour gray25; ')
                        ## colour_index 2(修改字体)
        normal_cell_style = xlwt.easyxf('borders: left thin,right thin,top thin,bottom thin; align: wrap on') 
        date_cell_style = xlwt.easyxf('borders: left thin,right thin,top thin,bottom thin', 
                                          num_format_str='yyyy-mm-dd')
        datetime_cell_style = xlwt.easyxf('borders: left thin,right thin,top thin,bottom thin', 
                                          num_format_str='yyyy-mm-dd hh:mm:ss')    
        
    
        
        # excel表头写入
        for k in range(0, field_num):
            worksheet.write(0, k, df.columns[k], header_cell_style)  #最后一个参数为格式
        
        # excel内容写入
        for nrow in range(0,len(df)):
            for ncol in range(0,field_num):
                if df.iloc[nrow, ncol]==None:      
                    worksheet.write(nrow + 1, ncol, None, normal_cell_style)
                    #worksheet.write(nrow + 1, ncol, )
                elif type(df.iloc[nrow, ncol]) == FloatInterval:
                    worksheet.write(nrow + 1, ncol, "{}".format(df.iloc[nrow, ncol]), normal_cell_style)   
                elif type(df.iloc[nrow, ncol]) is list:
                    worksheet.write(nrow + 1, ncol, '{}'.format(df.iloc[nrow, ncol]), normal_cell_style)
                elif type(df.iloc[nrow, ncol]) == str:
                    if df.iloc[nrow, ncol].strip() == '':
                        worksheet.write(nrow + 1, ncol, None, normal_cell_style)
                    else:
                        worksheet.write(nrow + 1, ncol, df.iloc[nrow, ncol], normal_cell_style)
                elif type(df.iloc[nrow, ncol]) in (pd._libs.tslib.Timestamp, datetime.date):
                    if type(df.iloc[nrow, ncol]) == pd._libs.tslib.Timestamp:
                        worksheet.write(nrow + 1, ncol, df.iloc[nrow, ncol], datetime_cell_style)      
                    else :
                        worksheet.write(nrow + 1, ncol, df.iloc[nrow, ncol], date_cell_style)                      
                elif type(df.iloc[nrow, ncol]) == decimal.Decimal:   
                    worksheet.write(nrow + 1, ncol, df.iloc[nrow, ncol], normal_cell_style)    
                else:
                    if np.isnan(df.iloc[nrow, ncol]):
                        worksheet.write(nrow + 1, ncol, None, normal_cell_style)
                    elif type(df.iloc[nrow, ncol]) is np.ndarray:
                        worksheet.write(nrow + 1, ncol, np.asscalar(df.iloc[nrow, ncol]), normal_cell_style)
                    elif (type(df.iloc[nrow, ncol]) is np.int32) | (type(df.iloc[nrow, ncol]) is np.int64):
                        worksheet.write(nrow + 1, ncol, int(df.iloc[nrow, ncol]), normal_cell_style)
                    else:
                        worksheet.write(nrow + 1, ncol, df.iloc[nrow, ncol], normal_cell_style)

    wb.save(filePath + '\\' + fileName + '.xls')




def to_exist_excel(filePath, fileName, sheetName, dataFrm):
    # 函数功能：从数据库中读取数据，并保存至格式定制好的excel中    
    '''
    Function Description:
        把数据框输出至已经存在的excel中,并生成新的sheet,并命名sheet
        
    Parameters
    ----------
    filePath : 已存在的excel文件夹路径，例如：filePath = 'F:\Python\AutoBuildScorecard\ModelCreditReport'
    fileName : 已存在的excel文件名称
    sheetName : 创建新sheet
    dataFrm : 需要输出的数据框
    
    Returns
    -------
    把结果保存至前面创建的excel中
    
    Examples
    --------
    to_exist_excel(
            filePath='F:\Python\AutoBuildScorecard\Result', 
            fileName='Summary', 
            sheetName='分类变量', 
            dataFrm=tmpDfClass)
    
    ''' 
    
    df = dataFrm
    
    #创建新工作簿ex
    eb = xlrd.open_workbook(filePath + '\\' + fileName + '.xls', formatting_info=True)
    wb = xl_copy(eb)
    worksheet = wb.add_sheet("%s" % sheetName)
    
    if dataFrm.shape[0]>0:
        # 首行冻结
        worksheet.panes_frozen = True
        worksheet.horz_split_pos = 1
        #worksheet.vert_split_pos = 1
        
        # 设置边框
        '''
        borders = xlwt.Borders()
        borders.left = 1
        borders.right = 1
        borders.top = 1
        borders.bottom = 1
        borders.bottom_colour = 0x3A
        style_edge1 = xlwt.XFStyle()
        style_edge1.borders = borders
        '''
        
        '''
        # 合并单元格并设置样式
        worksheet.row(0).height_mismatch = True
        worksheet.row(0).height = 20 * 25
        # 合并单元格并设置样式
        worksheet.write_merge(0, 0, 0, len(bodytitle), tabletitle, style=style1)
        worksheet.write_merge(1, 1, 0, len(bodytitle), remark)
        '''
        # 确定栏位宽度,根据字符长度来确定
        field_num = df.columns.size
        col_dict = dict()
        for i in range(0, field_num):
            if df[df.columns[i]].dtypes == 'str':
                col_dict_tmp = {"%s" % df.columns[i]:max((max(df[df.columns[i]].str.len()),len(str(df.columns[i]))))}
                col_dict.update(col_dict_tmp)
            elif max((max(df[df.columns[i]].astype('str').str.len()),len(str(df.columns[i])))) > 60:
                col_dict_tmp = {"%s" % df.columns[i]:60}  
                col_dict.update(col_dict_tmp) 
            else:
                col_dict_tmp = {"%s" % df.columns[i]:max((max(df[df.columns[i]].astype('str').str.len()),len(str(df.columns[i]))))}  
                col_dict.update(col_dict_tmp)   
    
        # 设置栏位宽度，栏位宽度小于10时候采用默认宽度
        for j in range(0, field_num):
            worksheet.col(j).width = int(256 * (col_dict["%s" % df.columns[j]] + 6))
     
        # 设置行高
        #header_rowstyle = xlwt.easyxf('font:height 240;') 
        #row0 = worksheet.row(0)
        #row0.set_style(header_rowstyle)
        
        # 设置单元格格式
        header_cell_style = xlwt.easyxf('font:bold on, height 200; align: wrap yes,vert centre, horiz center; \
                                    pattern: pattern solid, fore-colour gray25; ')
                        ## colour_index 2(修改字体)
        normal_cell_style = xlwt.easyxf('borders: left thin,right thin,top thin,bottom thin') 
        date_cell_style = xlwt.easyxf('borders: left thin,right thin,top thin,bottom thin', 
                                          num_format_str='yyyy-mm-dd')
        datetime_cell_style = xlwt.easyxf('borders: left thin,right thin,top thin,bottom thin', 
                                          num_format_str='yyyy-mm-dd hh:mm:ss')    
        
    
        
        # excel表头写入
        for k in range(0, field_num):
            worksheet.write(0, k, df.columns[k], header_cell_style)  #最后一个参数为格式
    
        
        # excel内容写入
        for nrow in range(0,len(df)):
            for ncol in range(0,field_num):
                if df.iloc[nrow, ncol]==None:      
                    worksheet.write(nrow + 1, ncol, None, normal_cell_style)
                    #worksheet.write(nrow + 1, ncol, )
                elif type(df.iloc[nrow, ncol]) == FloatInterval:
                    worksheet.write(nrow + 1, ncol, "{}".format(df.iloc[nrow, ncol]), normal_cell_style)   
                elif type(df.iloc[nrow, ncol]) is list:
                    worksheet.write(nrow + 1, ncol, '{}'.format(df.iloc[nrow, ncol]), normal_cell_style)
                elif type(df.iloc[nrow, ncol]) == str:
                    #print(df.iloc[nrow, ncol])
                    if df.iloc[nrow, ncol].strip() == '':
                        worksheet.write(nrow + 1, ncol, None, normal_cell_style)
                    else:
                        worksheet.write(nrow + 1, ncol, df.iloc[nrow, ncol], normal_cell_style)
                elif type(df.iloc[nrow, ncol]) in (pd._libs.tslib.Timestamp, datetime.date):
                    if type(df.iloc[nrow, ncol]) == pd._libs.tslib.Timestamp:
                        worksheet.write(nrow + 1, ncol, df.iloc[nrow, ncol], datetime_cell_style)      
                    else :
                        worksheet.write(nrow + 1, ncol, df.iloc[nrow, ncol], date_cell_style)                      
                elif type(df.iloc[nrow, ncol]) == decimal.Decimal:   
                    worksheet.write(nrow + 1, ncol, df.iloc[nrow, ncol], normal_cell_style)                     
                else:
                    if np.isnan(df.iloc[nrow, ncol]):
                        worksheet.write(nrow + 1, ncol, None, normal_cell_style)
                    elif type(df.iloc[nrow, ncol]) is np.ndarray:
                        worksheet.write(nrow + 1, ncol, np.asscalar(df.iloc[nrow, ncol]), normal_cell_style)
                    elif (type(df.iloc[nrow, ncol]) is np.int32) | (type(df.iloc[nrow, ncol]) is np.int64):
                        worksheet.write(nrow + 1, ncol, int(df.iloc[nrow, ncol]), normal_cell_style)
                    else:
                        worksheet.write(nrow + 1, ncol, df.iloc[nrow, ncol], normal_cell_style)

    wb.save(filePath + '\\' + fileName + '.xls')




