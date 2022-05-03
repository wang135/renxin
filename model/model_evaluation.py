# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import pandas as pd
import os

from base import equal_freq_cut_map


def _add_labels(rects):  
    for rect in rects:  
        height = round(rect.get_height(),3)  
        plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')  
        # horizontalalignment='center' plt.text(x坐标，y坐标，text,位置)  
        # 柱形图边缘用白色填充，为了更加清晰可分辨  
        rect.set_edgecolor('white')  
  
def prob_lift_chart(inDf, predVarName, yVarName, filePath, namePrefix):
    '''
    inDf = model_pred_df
    predVarName = 'y_pred'
    yVarName = 'TargetBad'
    filePath = 'F:\\Python\\Test'
    namePrefix = 'INS_'
    prob_lift_chart(inDf, predVarName, yVarName, filePath, namePrefix)
    '''
    
    if os.path.exists(filePath):
        print ("------------Old File------------")
    else :
        os.makedirs(filePath)
        
    ProbCut = equal_freq_cut_map(inDf[predVarName],10)['x_bin_ser']        
    LiftChartDat = pd.DataFrame(inDf[yVarName].groupby(ProbCut).mean())
    sample_size = inDf[yVarName].groupby(ProbCut).count()
    sample_size.name = 'Total'
    bad_size = inDf[yVarName].groupby(ProbCut).sum()
    bad_size.name = 'Bad'
    LiftChartDat = LiftChartDat.merge(sample_size, left_index=True, right_index=True, how='left')
    LiftChartDat = LiftChartDat.merge(bad_size, left_index=True, right_index=True, how='left')
    LiftChartDat = LiftChartDat.reset_index().reset_index()
    LiftChartDat['Decile'] = LiftChartDat['index']+1        
    plt.figure(figsize=(10,6))
    #plt.plot.bar(LiftChartDat['is_bad'],color='r')
    LiftChartBar = plt.bar(LiftChartDat['Decile'],LiftChartDat[yVarName],color='r',width=0.6)
    plt.xticks(LiftChartDat['Decile'])
    _add_labels(LiftChartBar)
    plt.savefig(filePath+'\\'+namePrefix+'LiftChart.png',dpi=200)
    return LiftChartDat

def prob_ks(inDf, predVarName, yVarName, filePath, namePrefix, nBins=10):
    '''
    inDf = model_pred_df
    predVarName = 'y_pred'
    yVarName = 'TargetBad'
    filePath = 'F:\\Python\\Test'
    namePrefix = 'INS_'
    prob_ks(inDf, predVarName, yVarName, filePath, namePrefix, nBins=10)
    '''
    
    if os.path.exists(filePath):
        print ("------------Old File------------")
    else :
        os.makedirs(filePath)
    import matplotlib.pyplot as plt
    
    ProbQtl = equal_freq_cut_map(inDf[predVarName],nBins)['x_bin_ser']  
    GrpResult = inDf.groupby(ProbQtl)[yVarName]
    
    ProbQtlFreq = pd.DataFrame(columns=['TotalCnt','BadCnt'])
    ProbQtlFreq['TotalCnt'] = GrpResult.count()
    ProbQtlFreq['BadCnt'] = GrpResult.sum()
    ProbQtlFreq['GoodCnt'] = ProbQtlFreq['TotalCnt']-ProbQtlFreq['BadCnt'] 
    ProbQtlFreq = ProbQtlFreq.sort_index(ascending=False)
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
            'KS_Value': KS_Value
            }

def prob_roc(inDf, predVarName, yVarName, filePath, namePrefix, nBins=100):
    '''
    inDf = model_pred_df
    predVarName = 'y_pred'
    yVarName = 'TargetBad'
    nBins = 100
    filePath = 'F:\\Python\\Test'
    namePrefix = 'INS_'
    prob_roc(inDf, predVarName, yVarName, filePath, namePrefix, nBins=100)
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
            'ROC_CurveDat':ROC_CurveDat,
            'AUC':AUC
            }



def model_prob_evaluation(inDf, predVarName, yVarName, filePath, namePrefix = ''):
    
    '''
    inDf = model_pred_df
    predVarName = 'y_pred'
    yVarName = 'TargetBad'
    nBins = 100
    filePath = 'F:\\Python\\Test'
    namePrefix = 'INS_'
    model_pred_evaluation(inDf, predVarName, yVarName, filePath, namePrefix)
    '''
    
    plt.figure()
    plt.hist(inDf[predVarName], bins=25, histtype='stepfilled')
    plt.title("Probability Histogram")
    plt.margins(x=0,y=0) 
    plt.savefig(filePath+'\\'+namePrefix+'histogram.png',dpi=200)
    plt.show()
    print("模型效果：Histogram over!")
    
    lift_chart_rst = prob_lift_chart(inDf, predVarName, yVarName, filePath, namePrefix)
    print("模型效果：Lift Chart over!")

    ks_rst = prob_ks(inDf, predVarName, yVarName, filePath, namePrefix, nBins=10)
    print("模型效果：KS over!！")

    roc_rst = prob_roc(inDf, predVarName, yVarName, filePath, namePrefix, nBins=100)
    print("模型效果：ROC over!！")

    return {'lift_chart_rst': lift_chart_rst,
            'ks_rst': ks_rst,
            'roc_rst':roc_rst }



  
def score_lift_chart(inDf, predVarName, yVarName, filePath, namePrefix):
    '''
    inDf = model_score_df
    predVarName = 'score'
    yVarName = 'TargetBad'
    filePath = 'F:\\Python\\Test'
    namePrefix = 'INS_'
    score_lift_chart(inDf, predVarName, yVarName, filePath, namePrefix)
    '''
    
    if os.path.exists(filePath):
        print ("------------Old File------------")
    else :
        os.makedirs(filePath)
        
    ProbCut = equal_freq_cut_map(inDf[predVarName],10)['x_bin_ser']        
    LiftChartDat = pd.DataFrame(inDf[yVarName].groupby(ProbCut).mean()).sort_index(ascending=False)
    sample_size = inDf[yVarName].groupby(ProbCut).count()
    sample_size.name = 'Total'
    bad_size = inDf[yVarName].groupby(ProbCut).sum()
    bad_size.name = 'Bad'
    LiftChartDat = LiftChartDat.merge(sample_size, left_index=True, right_index=True, how='left')
    LiftChartDat = LiftChartDat.merge(bad_size, left_index=True, right_index=True, how='left')
    LiftChartDat = LiftChartDat.reset_index().reset_index()
    LiftChartDat['Decile'] = LiftChartDat['index']+1        
    plt.figure(figsize=(10,6))
    #plt.plot.bar(LiftChartDat['is_bad'],color='r')
    LiftChartBar = plt.bar(LiftChartDat['Decile'],LiftChartDat[yVarName],color='r',width=0.6)
    plt.xticks(LiftChartDat['Decile'])
    _add_labels(LiftChartBar)
    plt.savefig(filePath+'\\'+namePrefix+'LiftChart.png',dpi=200)
    
    return LiftChartDat


def score_ks(inDf, predVarName, yVarName, filePath, namePrefix, nBins=10):
    '''
    inDf = model_score_df
    predVarName = 'score'
    yVarName = 'TargetBad'
    filePath = 'F:\\Python\\Test'
    namePrefix = 'INS_'
    score_ks(inDf, predVarName, yVarName, filePath, namePrefix, nBins=10)
    '''
    
    if os.path.exists(filePath):
        print ("------------Old File------------")
    else :
        os.makedirs(filePath)
    import matplotlib.pyplot as plt
    
    ProbQtl = equal_freq_cut_map(inDf[predVarName],nBins)['x_bin_ser']  
    GrpResult = inDf.groupby(ProbQtl)[yVarName]
    
    ProbQtlFreq = pd.DataFrame(columns=['TotalCnt','BadCnt'])
    ProbQtlFreq['TotalCnt'] = GrpResult.count()
    ProbQtlFreq['BadCnt'] = GrpResult.sum()
    ProbQtlFreq['GoodCnt'] = ProbQtlFreq['TotalCnt']-ProbQtlFreq['BadCnt'] 
    ProbQtlFreq = ProbQtlFreq.sort_index(ascending=True)
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


def score_roc(inDf, predVarName, yVarName, filePath, namePrefix, nBins=100):
    '''
    inDf = model_score_df
    predVarName = 'score'
    yVarName = 'TargetBad'
    filePath = 'F:\\Python\\Test'
    namePrefix = 'INS_'
    nBins = 100
    score_roc(inDf, predVarName, yVarName, filePath, namePrefix, nBins=100)
    '''
    if os.path.exists(filePath):
        print ("------------Old File------------")
    else :
        os.makedirs(filePath)
    import matplotlib.pyplot as plt
    ROC_Dat = inDf.sort_values(predVarName,ascending=True)
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
            'ROC_CurveDat':ROC_CurveDat,
            'AUC':AUC }



def model_score_evaluation(inDf, predVarName, yVarName, filePath, namePrefix = ''):
    
    '''
    inDf = model_score_df
    predVarName = 'score'
    yVarName = 'TargetBad'
    filePath = 'F:\\Python\\Test'
    namePrefix = 'INS_'
    model_score_evaluation(inDf, predVarName, yVarName, filePath, namePrefix)
    '''
    
    plt.figure()
    plt.hist(inDf[predVarName], bins=25, histtype='stepfilled')
    plt.title("Probability Histogram")
    plt.margins(x=0,y=0) 
    plt.savefig(filePath+'\\'+namePrefix+'histogram.png',dpi=200)
    plt.show()
    print("模型效果：Histogram over!")
    
    lift_chart_rst = score_lift_chart(inDf, predVarName, yVarName, filePath, namePrefix)
    print("模型效果：Lift Chart over!")

    ks_rst = score_ks(inDf, predVarName, yVarName, filePath, namePrefix, nBins=10)
    print("模型效果：KS over!！")

    roc_rst = score_roc(inDf, predVarName, yVarName, filePath, namePrefix, nBins=100)
    print("模型效果：ROC over!！")

    return {'lift_chart_rst': lift_chart_rst,
            'ks_rst': ks_rst,
            'roc_rst':roc_rst }





