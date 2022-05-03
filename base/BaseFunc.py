
import pandas as pd
import numpy as np
#import openpyxl
import datetime
import decimal
import xlwt
import xlrd
from xlutils.copy import copy as xl_copy

from pandas import DataFrame
from intervals import FloatInterval


def char_to_number(xSeries):
    top_ser = xSeries[~xSeries.isnull()].reset_index(drop=True)[0:100]
    digit_num = 0
    dot_num = 0
    dash_num = 0
    for value_item in top_ser:
        digit_num = digit_num + str(value_item).replace('.','').isdigit()
        dot_num = dot_num + str(value_item).count('.')
        dash_num = dash_num + str(value_item).count('-')
    if ((digit_num == 100) & (dot_num > 0) & (dash_num == 0)):
        xSeries = xSeries.replace('','-999999')
        res_ser = xSeries.map(lambda x: float(str(x)) if x is not None else np.NaN)
    elif ((digit_num == 100) & (dash_num == 0)):
        xSeries = xSeries.replace('','-999999')
        res_ser = xSeries.map(lambda x: int(x) if x is not None else np.NaN)
    else :
        res_ser = xSeries
    
    return res_ser



def cross_table(inDf, varX, varY):
    #Function Decription: 列联表频数计算
    
    # 计算非缺失值样本量
    TmpTab = pd.crosstab(inDf[varX],inDf[varY], dropna=False,margins=False)  
    # 创建空置列
    TmpTab['MissCnt'] = np.nan
    # 计算列合计样本量
    TmpSeri = inDf.groupby(varX).size()
    TmpSeri.name = 'Total'
    TmpTabs = TmpTab.join(TmpSeri, how='left').reset_index(drop=False)
    # 计算分类字段空值的样本量v
    TmpDfMiss = inDf[inDf[varX].isnull()]
    MissDict = dict(TmpDfMiss[varY].value_counts())  # 非缺失值分类样本量
    MissDict['MissCnt'] = TmpDfMiss[varY].isnull().sum()  # 缺失值样本量 
    MissDict['Total'] = TmpDfMiss.shape[0]  # 总样本量
    MissDict[varX] = 'Missing'
    
    if TmpDfMiss.shape[0] > 0:
        TableResult = TmpTabs.append(MissDict, ignore_index=True)
    else:
        TableResult = TmpTabs
    # 列合计    
    TableColumnTotal = TableResult.sum()
    TableColumnTotal[varX] = 'Total'
    TableResult = TableResult.append(TableColumnTotal, ignore_index=True)
    TableResult = TableResult.fillna(0)
    
    return TableResult



def var_freq_dist(x, pctFormat=True):
    #Function Decription:单指标频数分布
    Freq = x.value_counts(dropna=False)
    TotalFreq = pd.Series(sum(Freq),index=['Total'])
    Freqs = pd.concat([Freq,TotalFreq])
    Freqs.name = 'Freq'
    FreqRate = round(Freqs / sum(Freq),2)
    FreqRate.name = 'Rate'
    FreqDf = pd.concat([Freqs,FreqRate],axis=1)
    if pctFormat == True:
        FreqDf['Rate'] = FreqDf['Rate'].apply(lambda x: format(x, '.0%'))
    else :
        pass
    return FreqDf
        



def var_operate(x1, x2, sign):
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
    

def var_unique_value_cnt(inDf):
    NameList = list(inDf.columns)
    VarValueCntDict = dict()
    for i in range(len(NameList)): 
        VarValueCntDict.update({NameList[i]:inDf[NameList[i]].unique().size})    
    VarValueCntDat = DataFrame.from_dict(VarValueCntDict,orient='index').rename(columns={0:'VarValueCnt'})
    return VarValueCntDat   
    


def to_new_excel(filePath, fileName, sheetName, dataFrm):
    # 函数功能：从数据库中读取数据，并保存至格式定制好的excel中    
    
    '''
    filePath = 'F:\Python\AutoBuildScorecard\ModelCreditReport'
    fileName = '最优分箱结果'
    sheetName = '名义变量分箱频数分布'
    dataFrm = rate_nom_cmb_freq_df
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
                col_dict_tmp = {"%s" % df.columns[i]:max((max(df[df.columns[i]].str.len()),len(str(df.columns[i]))))}
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
                    else:
                        worksheet.write(nrow + 1, ncol, df.iloc[nrow, ncol], normal_cell_style)

    wb.save(filePath + '\\' + fileName + '.xls')

    
'''
to_new_excel(filePath='F:\Python\AutoBuildScorecard\Result', 
             fileName='Summary', 
             sheetName='连续变量', 
             dataFrm=tmpDfClass)
'''


def to_exist_excel(filePath, fileName, sheetName, dataFrm):
    # 函数功能：从数据库中读取数据，并保存至格式定制好的excel中    
    '''
    filePath = 'F:\Python\AutoBuildScorecard\ModelCreditReport'
    fileName = '最优分箱结果'
    sheetName= '有序变量分箱频数分布'
    dataFrm=rate_ord_cmb_freq_df
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
                elif type(df.iloc[nrow, ncol]) is list:
                    worksheet.write(nrow + 1, ncol, df.iloc[nrow, ncol], normal_cell_style)
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
                    else:
                        worksheet.write(nrow + 1, ncol, df.iloc[nrow, ncol], normal_cell_style)

    wb.save(filePath + '\\' + fileName + '.xls')



'''
to_exist_excel(filePath='F:\Python\AutoBuildScorecard\Result', 
               fileName='Summary', 
               sheetName='分类变量', 
               dataFrm=tmpDfClass)
'''




## 连续变量等频分箱
def cut_equal_freq(xVar, n):        
    '''
    xVar = ins_clean_df['loan_first_months']
    n = 10
    '''
    
    nbin = len(pd.Series(pd.qcut(xVar, n, duplicates = 'drop').dtype.categories.values))  ## 确定分箱的数量    
    var_bin = pd.qcut(xVar, n, labels=list(range(nbin)), duplicates = 'drop')  ## 变量分箱
    var_bin.name = "bin_%s" % var_bin.name  ## 分箱序列重命名
    
    return var_bin


## 名义变量频数分组合并
class NominalFreqBinFunc(object):
    def __init__(self):
        pass
    
    ## 单一值与列表比对，并赋值列表值
    def single_value(self, xValue, valueList):
        for item in valueList:
            if xValue in str(item).split(','):
                return item
    
    ##名义变量频数合并分箱
    def nominal_freq_bin_combine(self, inDf, inVar, freqCutOff):
        
        '''
        inDf = ins_clean_df
        inVar='work_phone_rate'
        freqCutOff=0.03
        '''
        
        #原始变量的频数统计
        TmpFreqResult=var_freq_dist(inDf[inVar], pctFormat=False)
        #获取变量值列表
        TmpValueList = list(TmpFreqResult.index)
        TmpValueList.pop()
    
        if TmpFreqResult['Rate'].min() < freqCutOff:
            #计算样本占比大于等于CutOff的分类的累计百分比。
            TmpFreqDict = TmpFreqResult.to_dict()['Rate']
            TmpPct=0
            for TmpValue in TmpValueList: 
                if TmpFreqDict[TmpValue] >= freqCutOff :
                    TmpPct = TmpPct + TmpFreqDict[TmpValue]
                  
            #获取样本占比大于CutOff的分类。
            FreqDat = TmpFreqResult[(TmpFreqResult['Rate']>=freqCutOff) & (TmpFreqResult.index != 'Total')]
            #获取满足条件的样本占比最小的百分比        
            TmpFloorDat = FreqDat.sort_values(by='Freq').head(1)
            TmpFloorLevel=TmpFloorDat.index[0]
            TmpFloorFreqDict = TmpFloorDat.to_dict()['Freq']
            TmpFloorRateDict = TmpFloorDat.to_dict()['Rate']
            #索引变为列值
            FreqDat=FreqDat.reset_index()
            
            #频数合并
            if (TmpPct>0.9) & (TmpPct<=0.95): #假如频数占比大于5%分类的和大于90%，则把所有频数占比小于5%的分类合并为一类
                TmpFreqDict = TmpFreqResult.to_dict()['Freq']
                TmpRateDict = TmpFreqResult.to_dict()['Rate']
                TmpPctList=list()
                TmpCulPct=0
                TmpCulFreq=0
                for TmpValue in TmpValueList:
                    if TmpRateDict[TmpValue]<0.05:
                        TmpPctList.append(TmpValue)
                        TmpCulFreq=TmpCulFreq+TmpFreqDict[TmpValue]
                        TmpCulPct=TmpCulPct+TmpRateDict[TmpValue]
                FreqDat.loc[FreqDat['Rate'].count()]=[",".join(TmpPctList),TmpCulFreq,TmpCulPct]
    
            elif TmpPct>0.95: 
                TmpFreqDict = TmpFreqResult.to_dict()['Freq']
                TmpRateDict = TmpFreqResult.to_dict()['Rate']
                TmpPctList=list()
                TmpCulPct=0
                TmpCulFreq=0
                for TmpValue in TmpValueList:
                    if TmpRateDict[TmpValue]<0.05:
                        TmpPctList.append(TmpValue)
                        TmpCulFreq=TmpCulFreq+TmpFreqDict[TmpValue]
                        TmpCulPct=TmpCulPct+TmpRateDict[TmpValue]
                ##若不满足条件的所有类之和小于5%，则向上累加
                TmpPctList.append(TmpFloorLevel)
                TmpCulFreq=TmpCulFreq+TmpFloorFreqDict[TmpFloorLevel]
                TmpCulPct=TmpCulPct+TmpFloorRateDict[TmpFloorLevel]
                FreqDat=FreqDat[FreqDat['index']!=TmpFloorLevel]
                FreqDat.loc[FreqDat['Rate'].count()]=[",".join(TmpPctList),TmpCulFreq,TmpCulPct]
    
            else:#假如频数占比大于5%分类的和小于等于90%，则把剩余频数占比小于5%的类，按照由小到大的顺序进行合并。
                #选择频数占比小于FreqCutOff的所有类
                TmpSmallCell = TmpFreqResult[TmpFreqResult['Rate']<freqCutOff].sort_values(by='Freq',ascending=False)
                TmpSmallValueList = list(TmpSmallCell.index)
                TmpFreqDict = TmpSmallCell.to_dict()['Freq']
                TmpRateDict = TmpSmallCell.to_dict()['Rate']
                TmpCulList=list()
                TmpCulPct=0
                TmpCulFreq=0
                for TmpSmall in TmpSmallValueList:
                    TmpPct = TmpPct + TmpRateDict[TmpSmall]
                    TmpCulFreq = TmpCulFreq+TmpFreqDict[TmpSmall]
                    TmpCulPct = TmpCulPct+TmpRateDict[TmpSmall]
                    TmpCulList.append(TmpSmall) 
                    #累计分类的频数占比大于FreqCutOff
                    if (TmpPct <= 0.9) & (TmpCulPct > 0.05):
                        FreqDat.loc[FreqDat['Rate'].count()]=[",".join(TmpCulList),TmpCulFreq,TmpCulPct]
                        TmpCulList=list()
                        TmpCulPct=0
                        TmpCulFreq=0
                        
                if TmpCulFreq > 0:
                    FreqDat.loc[FreqDat['Rate'].count()]=[",".join(TmpCulList),TmpCulFreq,TmpCulPct]
    
        else:
             FreqDat = TmpFreqResult[TmpFreqResult.index != 'Total'] 
             FreqDat=FreqDat.reset_index() 
        
        var_bin = inDf[inVar].map(lambda x: self.single_value(x, valueList=FreqDat['index'].tolist()))
        var_bin.name = "bin_%s" % var_bin.name  ## 分箱序列重命名
             
        return var_bin





## 有序变量频数分组合并
class BinOrderFreqCombined(object):
    def __init__(self):
        pass
    
    ## 单一值与列表比对，并赋值列表值
    def ordere_single_value(self, xValue, valueList):
        for item in valueList:
            if str(xValue) in item.split(', '):
                return item    

   
    ##单变量单次频数合并
    def OrderFreqBin01(self, inFreqDf, cutOff):
        
        RecordCnt = len(inFreqDf.index) 
        for i in range(RecordCnt):
            if (i==0) & (inFreqDf['Rate'][i] < cutOff):
                TmpCombDat=inFreqDf.iloc[[i,i+1]].sum()
                TmpComFreq=TmpCombDat['Freq']
                TmpComRate=TmpCombDat['Rate']
                TmpComIndex=i+1
                TmpComBin=[inFreqDf['Bins'][i],inFreqDf['Bins'][i+1]]            
                TmpFreqKeepDat=inFreqDf[~inFreqDf.index.isin([i,i+1])]
                TmpDat=pd.DataFrame([[TmpComIndex,TmpComFreq,TmpComRate,TmpComBin]], 
                                 columns=['index','Freq','Rate','Bins'])
                inFreqDf = pd.concat([TmpFreqKeepDat,TmpDat]).sort_values(by='index').reset_index(drop=True)
                break
            elif (i<RecordCnt-1) & (inFreqDf['Rate'][i] < cutOff):
                if inFreqDf['Rate'][i-1]>inFreqDf['Rate'][i+1]:
                    TmpCombDat=inFreqDf.iloc[[i,i+1]].sum()
                    TmpComFreq=TmpCombDat['Freq']
                    TmpComRate=TmpCombDat['Rate']
                    TmpComIndex=i+1
                    TmpComBin=[inFreqDf['Bins'][i],inFreqDf['Bins'][i+1]]            
                    TmpFreqKeepDat=inFreqDf[~inFreqDf.index.isin([i,i+1])]
                    TmpDat=pd.DataFrame([[TmpComIndex,TmpComFreq,TmpComRate,TmpComBin]], 
                                     columns=['index','Freq','Rate','Bins'])
                    inFreqDf = pd.concat([TmpFreqKeepDat,TmpDat]).sort_values(by='index').reset_index(drop=True)
                    break
                elif inFreqDf['Rate'][i-1]<inFreqDf['Rate'][i+1]:
                    TmpCombDat=inFreqDf.iloc[[i-1,i]].sum()
                    TmpComFreq=TmpCombDat['Freq']
                    TmpComRate=TmpCombDat['Rate']
                    TmpComIndex=i
                    TmpComBin=[inFreqDf['Bins'][i-1],inFreqDf['Bins'][i]]            
                    TmpFreqKeepDat=inFreqDf[~inFreqDf.index.isin([i-1,i])]
                    TmpDat=pd.DataFrame([[TmpComIndex,TmpComFreq,TmpComRate,TmpComBin]], 
                                     columns=['index','Freq','Rate','Bins'])
                    inFreqDf = pd.concat([TmpFreqKeepDat,TmpDat]).sort_values(by='index').reset_index(drop=True)
                    break
                else :
                    TmpCombDat=inFreqDf.iloc[[i,i+1]].sum()
                    TmpComFreq=TmpCombDat['Freq']
                    TmpComRate=TmpCombDat['Rate']
                    TmpComIndex=i+1
                    TmpComBin=[inFreqDf['Bins'][i],inFreqDf['Bins'][i+1]]            
                    TmpFreqKeepDat=inFreqDf[~inFreqDf.index.isin([i,i+1])]
                    TmpDat=pd.DataFrame([[TmpComIndex,TmpComFreq,TmpComRate,TmpComBin]], 
                                     columns=['index','Freq','Rate','Bins'])
                    inFreqDf = pd.concat([TmpFreqKeepDat,TmpDat]).sort_values(by='index').reset_index(drop=True)
                    break
            elif (i==RecordCnt-1) & (inFreqDf['Rate'][i] < cutOff):
                TmpCombDat=inFreqDf.iloc[[i-1,i]].sum()
                TmpComFreq=TmpCombDat['Freq']
                TmpComRate=TmpCombDat['Rate']
                TmpComIndex=i
                TmpComBin=[inFreqDf['Bins'][i-1],inFreqDf['Bins'][i]]            
                TmpFreqKeepDat=inFreqDf[~inFreqDf.index.isin([i-1,i])]
                TmpDat=pd.DataFrame([[TmpComIndex,TmpComFreq,TmpComRate,TmpComBin]], 
                                 columns=['index','Freq','Rate','Bins'])
                inFreqDf = pd.concat([TmpFreqKeepDat,TmpDat]).sort_values(by='index').reset_index(drop=True)
                break 
            
        return inFreqDf


    ##单变量多次频数合并
    def OrderFreqBin02(self, inDf, inVarName, cutOff):
        '''
        inDf = ins1
        inVarName = 'qq_length'
        cutOff = 0.05
        '''
        
        TmpFreqResult=var_freq_dist(inDf[inVarName], pctFormat=False)
        TmpFreqResult = TmpFreqResult[TmpFreqResult.index != 'Total'].sort_index().reset_index()
        TmpFreqResult['Bins'] = TmpFreqResult['index'] 
        StepTimes = TmpFreqResult[TmpFreqResult['Rate']<cutOff].index.size
        while StepTimes>0 :
            OrderOneStepDat = self.OrderFreqBin01(inFreqDf=TmpFreqResult, cutOff=cutOff)
            TmpFreqResult=OrderOneStepDat.drop('index',axis=1)
            TmpFreqResult=TmpFreqResult.reset_index()
            StepTimes = TmpFreqResult[TmpFreqResult['Rate']<cutOff].index.size
        
        FreqSer = TmpFreqResult['Bins'].astype(str).apply(lambda x: x.replace('[','').replace(']',''))
        
        var_bin = inDf[inVarName].map(lambda x: self.ordere_single_value(x, valueList=FreqSer.tolist()))
        var_bin.name = "bin_%s" % var_bin.name  ## 分箱序列重命名
        
        return var_bin         

        
        


def equal_freq_cut(x, nBin):
    '''
    x = inDf[xVarName]
    nBin = 10
    '''
    x = x[x.notnull()]
    def bin_map(x, binList):
        if x <= binList[0]:
            return FloatInterval.open_closed(binList[0], binList[1])
        for i in range(len(binList)+1):
            if x > binList[i] and x <= binList[i+1]:
                return FloatInterval.open_closed(binList[i], binList[i+1])
    
    bin_ls = [np.percentile(x, 100/nBin*i) for i in range(nBin+1)]
    bin_ls = sorted(list(set(bin_ls)))
    
    bin_range_ls = list()    
    for i in range(len(bin_ls)-1):
        bin_range_ls.append([i+1,FloatInterval.open_closed(bin_ls[i], bin_ls[i+1])])
        
    return {'bin_range_ls': bin_range_ls,
            'x_bin_ser': x.map(lambda x: bin_map(x, bin_ls))
            }
            


