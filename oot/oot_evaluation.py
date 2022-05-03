# -*- coding: utf-8 -*-

import math
import pandas as pd
import matplotlib.pyplot as plt
import os

from base import equal_freq_cut


def _value_bin_map(xVar, binRangeSer):
    '''
    变量值与等频分箱的映射
    xVar = 0.00703912
    binRangeSer = bin_range_ls
    
    '''
    bins = len(binRangeSer)
    for i in range(bins):
        if (i == 0) & (xVar <= binRangeSer[i][1].upper):
                return i+1
        elif (i == bins-1) & (xVar > binRangeSer[i][1].lower):
                return i+1
        else :
            if (xVar > binRangeSer[i][1].lower) & (xVar <= binRangeSer[i][1].upper):
                return i+1
            
        
def psi_cal(inEvalDf, xVarName, psiBy, psiByVarName  = 'PassMth'):
    #inBaseDf = model_pred_df
    '''
    Function Descriptions:
        计算评分群体稳定性指标PSI，根据psiBy所确定的计算类型，进行计算。
        
    Parameters
    ----------
    inEvalDf      : 含有预测值及样本分类指标的数据框
    xVarName      : 预测变量
    psiBy         : 预测类型，分为两类：'oot','time'
    psiByVarNamae : 变量名称，假如指定time类型，则需要正确填写时间变量名称
    
    Returns
    -------
    PSI计算过程及其值的数据框
    
    Examples
    --------
    inEvalDf = eval_pred_df   
    xVarName = 'y_pred' 
    psiBy = 'OOT'
    psiByVarName = 'PassMth'
    psi_calculate(inEvalDf, xVarName, psiBy, psiByVarName  = 'PassMth')
    '''
    ## 获取建模样本，做为分箱的标准
    ins_df = inEvalDf[inEvalDf['SampleType'] == 'INS']
    ## 对预测值进行等频分箱
    bin_range_ls = equal_freq_cut(ins_df[xVarName],10)
    ## 对验证样本按照INS分箱方式，对预测值进行分箱
    sample_df = inEvalDf.copy()
    var_bin = sample_df[xVarName].map(lambda x: _value_bin_map(x, binRangeSer=bin_range_ls))      
    sample_df["bin_{}".format(var_bin.name)] = var_bin 
    
    ins_bin_df = sample_df[sample_df['SampleType']=='INS']  
    oot_bin_df = sample_df[sample_df['SampleType']=='OOT']
    
    if psiBy.lower() == 'oot':
        
        ins_freq_df = pd.DataFrame({'ins_freq': ins_bin_df.groupby("bin_{}".format(var_bin.name))[xVarName].count()})
        oot_freq_df = pd.DataFrame({'oot_freq': oot_bin_df.groupby("bin_{}".format(var_bin.name))[xVarName].count()})
    
        psi_df = pd.merge(ins_freq_df, oot_freq_df, left_index=True, right_index=True)
        ## OOT与INS的对应分箱样本占比的差
        psi_df['ins_dist'] = round(psi_df['ins_freq']/psi_df['ins_freq'].sum(),4)
        psi_df['oot_dist'] = round(psi_df['oot_freq']/psi_df['oot_freq'].sum(),4)
        ## OOT与INS的对应分箱样本占比的比值的对数
        psi_df['dist_dif'] = psi_df['oot_dist']-psi_df['ins_dist']
        psi_df['dist_log'] = (psi_df['oot_dist']/psi_df['ins_dist']).map(lambda x: round(math.log(x),6))
        ## 计算每个分箱的PSI值
        psi_df['psi_item'] = psi_df['dist_dif']*psi_df['dist_log']
        psi_df['psi'] = psi_df['psi_item'].sum()
        
        print("***************************************************************************")
        print("OOT PSI为：\n", round(psi_df['psi_item'].sum(),4))
        return psi_df
    
    elif psiBy.lower() == 'month':
        
        mth_ls = sorted(oot_bin_df[psiByVarName].unique().tolist())
        psi_df = pd.DataFrame(columns=['ins_freq','oot_freq','ins_dist','oot_dist','dist_dif','dist_log','psi_item','psi',psiByVarName])
        for mth_value in mth_ls:
            ## 获取对应月份数据
            mth_bin_df = oot_bin_df[oot_bin_df[psiByVarName]==mth_value]
            
            ins_freq_df = pd.DataFrame({'ins_freq': ins_bin_df.groupby("bin_{}".format(var_bin.name))[xVarName].count()})
            mth_freq_df = pd.DataFrame({'oot_freq': mth_bin_df.groupby("bin_{}".format(var_bin.name))[xVarName].count()})
        
            mth_psi_df = pd.merge(ins_freq_df, mth_freq_df, left_index=True, right_index=True)
            ## OOT与INS的对应分箱样本占比的差
            mth_psi_df['ins_dist'] = round(mth_psi_df['ins_freq']/mth_psi_df['ins_freq'].sum(),4)
            mth_psi_df['oot_dist'] = round(mth_psi_df['oot_freq']/mth_psi_df['oot_freq'].sum(),4)
            ## OOT与INS的对应分箱样本占比的比值的对数
            mth_psi_df['dist_dif'] = mth_psi_df['oot_dist']-mth_psi_df['ins_dist']
            mth_psi_df['dist_log'] = (mth_psi_df['oot_dist']/mth_psi_df['ins_dist']).map(lambda x: round(math.log(x),6))
            ## 计算每个分箱的PSI值
            mth_psi_df['psi_item'] = mth_psi_df['dist_dif']*mth_psi_df['dist_log']
            mth_psi_df['psi'] = mth_psi_df['psi_item'].sum()
            mth_psi_df[psiByVarName] = mth_value
                        
            print ("PSI: ", mth_value)
            
            psi_df = pd.concat([psi_df,mth_psi_df]).reset_index(drop=True)
                    
        print("***************************************************************************")
        print("OOT PSI为：\n", psi_df[[psiByVarName, 'psi']].drop_duplicates())
        
        return psi_df
        



def _add_labels(rects):  
    for rect in rects:  
        height = round(rect.get_height(),3)  
        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')  
        # horizontalalignment='center' plt.text(x坐标，y坐标，text,位置)  
        # 柱形图边缘用白色填充，为了更加清晰可分辨  
        rect.set_edgecolor('white')  



def _bin_map(x, binDict, plotType):
    
    if plotType == 'LiftChart':
        for level in binDict.keys():
            if level == min(list(binDict.keys())):
                if x <= binDict[level].upper:
                    return level
            elif level == max(list(binDict.keys())):
                if x > binDict[level].lower:
                    return level
            elif x > binDict[level].lower and x <= binDict[level].upper:
                return level

    elif plotType == 'KS':
        for level in binDict.keys():
            if level == min(list(binDict.keys())):
                if x <= binDict[level].upper:
                    return len(binDict.keys())-level+1
            elif level == max(list(binDict.keys())):
                if x > binDict[level].lower:
                    return len(binDict.keys())-level+1
            elif x > binDict[level].lower and x <= binDict[level].upper:
                return len(binDict.keys())-level+1
        
        

  
def oot_prob_lift_chart(inDf, predVarName, yVarName, inBinDf, binVarName, filePath, namePrefix):
    '''
    Function Descriptions:
        使用INS的分箱标准对OOT数据进行分箱，并计算LiftChart。
        
    Parameters
    ----------
    inDf         : OOT数据框
    predVarName  : 预测变量
    yVarName     : 目标变量
    inBinDf      : 变量分箱标准数据框
    binVarName   : inBinDf中的分箱变量
    filePath     : 变量存放路径
    namePrefix   : 文件前缀
    
    Returns
    -------
    计算lift chart的过程数据框
    
    Examples
    --------
    inDf = oot_pred_df
    predVarName = 'y_pred'
    yVarName = 'TargetBad'
    inBinDf = model_eval_rst['lift_chart_rst']
    binVarName = 'bin_y_pred'
    filePath = 'F:\\Python\\AutoBuildScorecard\\Result'
    namePrefix = 'OOT'
    oot_prob_lift_chart(inDf, predVarName, yVarName, inBinDf, binVarName, filePath, namePrefix)
    '''
    
    if os.path.exists(filePath):
        print ("------------Old File------------")
    else :
        os.makedirs(filePath)
    
    BinVarName = 'bin_{}'.format(binVarName)
    bin_dict = inBinDf.set_index('Decile')[binVarName].to_dict()
    inDf[BinVarName] = inDf[predVarName].map(lambda x: _bin_map(x, binDict=bin_dict, plotType='LiftChart'))
    
    LiftChartDat = pd.DataFrame(inDf.groupby(BinVarName)[yVarName].mean())
    sample_ser = inDf.groupby(BinVarName)[yVarName].count()
    sample_ser.name = 'Total'
    LiftChartDat = LiftChartDat.merge(sample_ser, left_index=True, right_index=True, how='left')
    bad_ser = inDf.groupby(BinVarName)[yVarName].sum()
    bad_ser.name = 'Bad'
    LiftChartDat = LiftChartDat.merge(bad_ser, left_index=True, right_index=True, how='left')
    
    LiftChartDat = LiftChartDat.reset_index().reset_index()
    LiftChartDat['Decile'] = LiftChartDat['index']+1        
    plt.figure(figsize=(10,6))
    #plt.plot.bar(LiftChartDat['is_bad'],color='r')
    LiftChartBar = plt.bar(LiftChartDat['Decile'],LiftChartDat[yVarName],color='r',width=0.6)
    plt.xticks(LiftChartDat['Decile'])
    _add_labels(LiftChartBar)
    plt.savefig(filePath+'\\'+namePrefix+'LiftChart.png',dpi=200)
    
    return LiftChartDat



def oot_prob_ks(inDf, predVarName, yVarName, inBinDf, binVarName, filePath, namePrefix):
    '''
    Function Descriptions:
        使用INS的分箱标准对OOT数据进行分箱，并计算KS值，生成KS图。
        
    Parameters
    ----------
    inDf         : OOT数据框
    predVarName  : 预测变量
    yVarName     : 目标变量
    inBinDf      : 变量分箱标准数据框
    binVarName   : inBinDf中的分箱变量
    filePath     : 变量存放路径
    namePrefix   : 文件前缀
    
    Returns
    -------
    计算KS的过程数据框
    
    Examples
    --------
    inDf = oot_pred_df
    predVarName = 'y_pred'
    yVarName = 'TargetBad'
    inBinDf = model_eval_rst['ks_rst']['KSCurveDat']
    binVarName = 'bin_y_pred'
    filePath = 'F:\\Python\\AutoBuildScorecard\\Result'
    namePrefix = 'OOT'
    oot_prob_ks(inDf, predVarName, yVarName, inBinDf, binVarName, filePath, namePrefix)
    '''
    
    if os.path.exists(filePath):
        print ("------------Old File------------")
    else :
        os.makedirs(filePath)
    import matplotlib.pyplot as plt
    
    bin_df = inBinDf.copy()
    bin_df = bin_df.sort_values(by = ['index'], ascending=False).reset_index(drop=True)
    bin_df['index'] = bin_df.index+1
    bin_dict = bin_df.set_index('index')[binVarName].to_dict()
    ProbQtl = inDf[predVarName].map(lambda x: _bin_map(x=x, binDict=bin_dict, plotType='KS'))
    GrpResult = inDf.groupby(ProbQtl)[yVarName]
    
    ProbQtlFreq = pd.DataFrame(columns=['TotalCnt','BadCnt'])
    ProbQtlFreq['TotalCnt'] = GrpResult.count()
    ProbQtlFreq['BadCnt'] = GrpResult.sum()
    ProbQtlFreq['GoodCnt'] = ProbQtlFreq['TotalCnt']-ProbQtlFreq['BadCnt'] 
    #ProbQtlFreq = ProbQtlFreq.sort_index(ascending=False)
    ProbQtlFreq['BadCulCnt'] = ProbQtlFreq['BadCnt'].cumsum()
    ProbQtlFreq['GoodCulCnt'] = ProbQtlFreq['GoodCnt'].cumsum()
    ProbQtlFreq['BadCulRate'] = ProbQtlFreq['BadCulCnt'] / ProbQtlFreq['BadCnt'].sum()
    ProbQtlFreq['GoodCulRate'] = ProbQtlFreq['GoodCulCnt'] / ProbQtlFreq['GoodCnt'].sum()
    ProbQtlFreq['KS_Curve'] = ProbQtlFreq['BadCulRate'] - ProbQtlFreq['GoodCulRate']
    ProbQtlFreq=ProbQtlFreq.reset_index()
    
    KS_Value = ProbQtlFreq['KS_Curve'].max()
    
    KSCurveDat = ProbQtlFreq[['BadCulRate','GoodCulRate','KS_Curve']]
    KSCurveDat = KSCurveDat.reset_index()
    KSCurveDat['index'] = KSCurveDat['index']+1
    KSCurveDat = KSCurveDat.set_index('index')
    ZeroDat = pd.DataFrame([[0,0,0]],columns=['BadCulRate','GoodCulRate','KS_Curve'])
    KSCurveDat = pd.concat([KSCurveDat,ZeroDat])
    KSCurveDat = KSCurveDat.sort_index()   
    KSCurveDat['Decile'] = KSCurveDat.reset_index()['index']
    plt.figure(figsize=(10,6))
    plt.plot(KSCurveDat['Decile'], KSCurveDat['BadCulRate'], color='steelblue')
    plt.plot(KSCurveDat['Decile'], KSCurveDat['GoodCulRate'], color='orange')
    plt.plot(KSCurveDat['Decile'], KSCurveDat['KS_Curve'], color='g')
    plt.xticks(KSCurveDat['Decile'])
    plt.margins(x=0,y=0) 
    plt.title("KS = %4.2f" % KS_Value)
    plt.savefig(filePath+'\\'+namePrefix+'KS.png',dpi=200)
    
    ProbQtlFreq = ProbQtlFreq.reset_index(drop=False)
    ProbQtlFreq['index'] = ProbQtlFreq['index']+1
    
    return {'KSCurveDat': ProbQtlFreq,
            'KS_Value': KS_Value }

def oot_prob_roc(inDf, predVarName, yVarName, filePath, namePrefix, nBins=100):
    '''
    Function Descriptions:
        使用INS的分箱标准对OOT数据进行分箱，并计算AUC值，生成ROC图。
        
    Parameters
    ----------
    inDf         : OOT数据框
    predVarName  : 预测变量
    yVarName     : 目标变量
    filePath     : 变量存放路径
    namePrefix   : 文件前缀
    nBins        : 生成ROC的分箱数量
    
    Returns
    -------
    计算AUC的过程数据框
    
    Examples
    --------
    inDf = model_pred_df
    predVarName = 'y_pred'
    yVarName = 'TargetBad'
    nCutOff = 100
    filePath = 'F:\\Python\\AutoBuildScorecard\\Result'
    namePrefix = "OOT"
    '''
    if os.path.exists(filePath):
        print ("------------Old File------------")
    else :
        os.makedirs(filePath)
    import matplotlib.pyplot as plt
    ROC_Dat = inDf.sort_values(predVarName,ascending=False)
    ROC_Dat['Records'] = 1
    ROC_Dat['CumRecords'] = ROC_Dat['Records'].cumsum()
    ROC_Dat['CumBads'] = ROC_Dat[yVarName].cumsum()
    TotalCnt = ROC_Dat.index.size
    TotalBadCnt = ROC_Dat[yVarName].sum()
    TotalGoodCnt = TotalCnt - TotalBadCnt
    ROC_Dat['TP'] = ROC_Dat['CumBads']
    ROC_Dat['FN'] = TotalBadCnt - ROC_Dat['CumBads']
    ROC_Dat['FP'] = ROC_Dat['CumRecords'] - ROC_Dat['CumBads']
    ROC_Dat['TN'] = TotalGoodCnt - ROC_Dat['FP']
    ROC_Dat['TPR'] = round(ROC_Dat['TP'] / TotalBadCnt,8)
    ROC_Dat['FPR'] = round(ROC_Dat['FP'] / TotalGoodCnt,8)
    
    ROC_Dat = ROC_Dat.reset_index(drop=True)
    ROC_Dat = ROC_Dat.reset_index()
    #按等距分箱，画出ROC曲线
    QuantileList = list()
    for i in range(nBins):
        qt = (i+1) / nBins
        QuantileList.append(round(ROC_Dat['index'].quantile(qt),0))    
    ROC_CurveDat=ROC_Dat.loc[QuantileList,['TPR','FPR']]
    #增加起始坐标（0，0）
    ROC_CurveDat = pd.concat([ROC_CurveDat,pd.DataFrame([[0,0]],columns=['TPR','FPR'])])
    ROC_CurveDat = ROC_CurveDat.sort_values('TPR')

    ##AUC计算
    AUC=0
    for j in range(ROC_Dat.index.size-1):
        TmpAUC = (ROC_Dat.loc[j+1,'TPR']+ROC_Dat.loc[j,'TPR']) * (ROC_Dat.loc[j+1,'FPR']-ROC_Dat.loc[j,'FPR']) * 0.5
        AUC = AUC + TmpAUC
    AUC = round(AUC,4)
    
    plt.figure()
    plt.plot(ROC_CurveDat['FPR'],ROC_CurveDat['TPR'])
    plt.title("AUC = %4.2f" % AUC)
    plt.margins(x=0,y=0) 
    plt.savefig(filePath+'\\'+namePrefix+'ROC.png',dpi=200)
    plt.show()
    
    return {'ROC_Dat': ROC_Dat,
            'ROC_CurveDat': ROC_CurveDat,
            'AUC': AUC }
    
    
def oot_prob_evaluation(inDf, predVarName, yVarName, liftChartDf, liftChartBinName, ksDf, ksBinName, filePath, namePrefix = ''):
    '''
    Function Descriptions:
        使用INS的分箱标准对OOT数据进行分箱，并生成LiftChart、KS、ROC图。
        
    Parameters
    ----------
    inDf             : OOT数据框
    predVarName      : 预测变量
    yVarName         : 目标变量
    liftChartDf      : INS中LiftChart变量分箱标准数据框
    liftChartBinName : liftChartDf中分箱变量名称
    ksDf             : INS中ks变量分箱标准数据框
    ksBinName        : ksDf中分箱变量名称
    filePath         : 变量存放路径
    namePrefix       : 文件前缀
    
    Returns
    -------
    计算AUC的过程数据框
    
    Examples
    --------
    inDf = oot_pred_df, 
    predVarName = 'y_pred'
    yVarName = var_target
    liftChartDf = model_eval_rst['lift_chart_rst'] 
    liftChartBinName = 'bin_y_pred' 
    ksDf = model_eval_rst['ks_rst']['KSCurveDat'], 
    ksBinName = 'bin_y_pred'
    filePath = file_path+'\pic' 
    namePrefix='OOT_'
    '''
    lift_chart_rst = oot_prob_lift_chart(inDf=inDf,
                                  predVarName=predVarName,
                                  yVarName=yVarName,
                                  inBinDf=liftChartDf,
                                  binVarName=liftChartBinName,
                                  filePath = filePath,
                                  namePrefix = namePrefix)
    
    ks_rst = oot_prob_ks(inDf=inDf, 
                                  predVarName=predVarName, 
                                  yVarName=yVarName, 
                                  inBinDf=ksDf,
                                  binVarName=ksBinName,
                                  filePath=filePath,
                                  namePrefix = namePrefix)
    
    roc_rst = oot_prob_roc(inDf=inDf, 
                               predVarName=predVarName, 
                               yVarName=yVarName, 
                               nBins=100,
                               filePath = filePath,
                                  namePrefix = namePrefix)
    
    return {'lift_chart_rst': lift_chart_rst,
            'ks_rst': ks_rst,
            'roc_rst': roc_rst }        



