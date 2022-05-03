
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sbn
sbn.set(context='notebook', font='simhei', style='whitegrid', font_scale=1.5)
import warnings
warnings.filterwarnings('ignore')
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams['font.sans-serif']=['Microsoft YaHei']



def variable_plot(inDf, inVarClassDf, savUrl ):
    '''
    Function Description:
        对数据框中的指定变量输出图，分类变量输出柱状图，连续变量输出概率分布图
        
    Parameters
    ----------
    inDf         : 分析对象数据框
    inVarClassDf : 变量类型数据框
    savUrl       : pdf结果存放路径
    
    Returns
    -------
    pdf文件
    '''
    matplotlib.use('pdf')
    for varType in ['Continuous','Nominal','Binary','Order']:
        if varType == 'Continuous':
            pdf = PdfPages(r"{}\PlotContinuousDist.pdf".format(savUrl))
            tmpConList = inVarClassDf[inVarClassDf['Dclass'] == 'Continuous']['index'].reset_index(drop=True)
            for varItem in tmpConList:
                TmpMissCnt = inDf[varItem].isnull().sum()
                plt.figure(figsize=(15, 10))
                plt.subplots_adjust(left=0.1,right=0.9,wspace=1,hspace=1,bottom=0.2,top=0.85)
                plt.title(u"%s分布图如下： \n" % (varItem), loc='left', fontdict={'fontweight':'bold', 'color':'forestgreen', 'fontsize':24})
                sbn.distplot(inDf[varItem].dropna(), 
                             kde=0,  # 不显示密度图 
                             hist_kws={'color':'grey'},
                             axlabel = "\n%s  \n%s: %d \n" % (varItem,'Missing',TmpMissCnt),
                             norm_hist=0) 
            
                pdf.get_pagecount()
                pdf.savefig()
            plt.close()
            pdf.close()
        else:
            pdf = PdfPages(r'{}\PlotClassDist.pdf'.format(savUrl))
            tmpClsList = inVarClassDf[inVarClassDf['Dclass'].isin(['Nominal','Binary','Order'])]['index'].reset_index(drop=True)
            for varItem in tmpClsList:
                plt.figure(figsize=(15, 10))
                plt.subplots_adjust(left=0.1,right=0.9,wspace=1,hspace=1,bottom=0.2,top=0.85)
                plt.title(u"%s分布图如下： \n" % (varItem), loc='left', fontdict={'fontweight':'bold', 'color':'forestgreen', 'fontsize':24})
                if inDf[varItem].isna().sum() ==0:
                    value_list = list(inDf[varItem].dropna().unique())
                else:
                    value_list = list(inDf[varItem].dropna().unique())
                    value_list.append('Null')
                sbn.countplot(x=inDf[varItem].fillna('Null'), order=value_list)
                pdf.savefig()
            plt.close()
            pdf.close()
    matplotlib.use('Qt5Agg')

