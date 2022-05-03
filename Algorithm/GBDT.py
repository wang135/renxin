# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:43:55 2020

@author: finup
"""

import numpy as np
import pandas as pd
import graphviz
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.tree import export_graphviz

from sklearn import model_selection, metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV   #Perforing grid search
import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

x,y = make_classification(n_samples=80000)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

gbdt = GBDT(n_estimators=10)

gbdt.fit(X_train, y_train)



##=========================================================================================================##
##                                         超参调优步骤                                                 ##
##=========================================================================================================##

##案例网址
##https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/


def modelfit(alg, dtrain, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = model_selection.cross_val_score(alg, dtrain[predictors], dtrain['Disbursed'], cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
    
    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')



## baseline
train = X_train.copy()
target = 'Disbursed'
IDcol = 'ID'

predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm0 = GradientBoostingClassifier(random_state=10)
#%timeit modelfit(gbm0, train, predictors)

## 1、学习率（初始设置0.05-0.2）和分类器（决策树）数量（初始设置：40-70）
# 在固定学习率的情况下，首先尝试分类器的数量，假如分类器的数量最终确定在40-70范围内，则无需进行学习率的调整。
predictors = [x for x in train.columns if x not in [target, IDcol]]
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,
                                                               max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
                        param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch1.fit(train[predictors],train[target])

pd.DataFrame({'mean_test_score':gsearch1.cv_results_['mean_test_score'],
              'std_test_score':gsearch1.cv_results_['std_test_score'],
              'params':gsearch1.cv_results_['params']}), \
gsearch1.best_params_, gsearch1.best_score_

modelfit(gsearch1.best_estimator_, train, predictors)


## 2、决策树：深度（初始值5-8）、分割点样本量（总样本量的0.5%-1%)
param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, 
                                                               max_features='sqrt', subsample=0.8, random_state=10), 
                        param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
pd.DataFrame({'mean_test_score':gsearch2.cv_results_['mean_test_score'],
              'std_test_score':gsearch2.cv_results_['std_test_score'],
              'params':gsearch2.cv_results_['params']}), \
gsearch2.best_params_, gsearch2.best_score_

modelfit(gsearch2.best_estimator_, train, predictors)

## 3、决策树：叶子节点最小样本量
param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,
                                                               max_features='sqrt', subsample=0.8, random_state=10), 
                        param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train[predictors],train[target])

pd.DataFrame({'mean_test_score':gsearch3.cv_results_['mean_test_score'],
              'std_test_score':gsearch3.cv_results_['std_test_score'],
              'params':gsearch3.cv_results_['params']}), \
gsearch3.best_params_, gsearch3.best_score_

# =====
modelfit(gsearch3.best_estimator_, train, predictors)


## 4、决策树：用于搭建决策树的特征数量（初始值为sqrt）
param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9, min_samples_split=1200, 
                                                               min_samples_leaf=60, subsample=0.8, random_state=10),
                        param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
pd.DataFrame({'mean_test_score':gsearch4.cv_results_['mean_test_score'],
              'std_test_score':gsearch4.cv_results_['std_test_score'],
              'params':gsearch4.cv_results_['params']}), \
gsearch4.best_params_, gsearch4.best_score_


## 5、训练模型的样本量（常用初始值为：0.8）
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=9,min_samples_split=1200, 
                                                               min_samples_leaf=60, subsample=0.8, random_state=10,max_features=7),
                        param_grid = param_test5, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch5.fit(train[predictors],train[target])
pd.DataFrame({'mean_test_score':gsearch5.cv_results_['mean_test_score'],
              'std_test_score':gsearch5.cv_results_['std_test_score'],
              'params':gsearch5.cv_results_['params']}), \
gsearch5.best_params_, gsearch5.best_score_


# =====
## Final1:缩小学习率2倍，扩大分类器数量10倍
predictors = [x for x in train.columns if x not in [target, IDcol]]
gbm_tuned_1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=9, min_samples_split=1200,min_samples_leaf=60, subsample=0.85, random_state=10, max_features=7)

modelfit(gbm_tuned_1, train, predictors)

## Final1:缩小学习率10倍，扩大分类器数量10倍




## Final1:缩小学习率20倍，扩大分类器数量20倍




##=========================================================================================================##
##                                         重要输出变量解释                                                 ##
##=========================================================================================================##


## 阶段函数结果
np.array(list(gbdt.staged_decision_function(X_train)))[:,:,0].shape
np.array(list(gbdt.staged_predict(X_train))).shape
np.array(list(gbdt.staged_predict_proba(X_train)))[:,:,0].shape
stage_prob2 = np.array(list(gbdt.staged_predict_proba(X_train))).transpose(1,0,2)

## 阶段损失函数值
gbdt.train_score_

## 生成叶子节点
leaf = gbdt.apply(X_train)
leaf = leaf[:,:,0]

## 变量重要性
gbdt.feature_importances_

## 模型效果评估
gbdt.score(X_train, y_train)


## 决策树结果
estimator = gbdt.estimators_[i][0]
estimator.tree_.feature
estimator.tree_.threshold
estimator.tree_.impurity
estimator.tree_.value  ##残差



##=========================================================================================================##
##                                             决策树图输出                                                 ##
##=========================================================================================================##


for i in list(range(len(gbdt.estimators_))):
    
    i=0
    estimator = gbdt.estimators_[i][0]
    export_graphviz(estimator,
                    out_file = r'F:\Python\Test\tree.dot',
                    feature_names = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10',
                                     'v11','v12','v13','v14','v15','v16','v17','v18','v19','v20'],
                    class_names = 'is_bad',
                    rounded = True,
                    proportion = False,
                    precision = 2,
                    filled = True)

    with open(r'F:\Python\Test\tree.dot') as f:
        dot_graph = f.read()
    graph = graphviz.Source(dot_graph)
    graph.render(cleanup=True, filename = 'tree_{}'.format(i), directory = r'F:\Python\Test', format = 'pdf')
 


##=========================================================================================================##
##                                          决策树叶子节点统计量                                            ##
##=========================================================================================================##


## 随机森林结果转化为决策树节点
gbdt_tree = pd.DataFrame(gbdt.apply(X_train)[:,:,0], 
                       columns=[('tree_' + str(i)) for i in range(len(gbdt.estimators_))])
gbdt_tree['y'] = y_train
##随机森林的决策树叶子节点统计量
tree_leaf_stat = pd.DataFrame(columns=['tree_name', 'leaf_node', 'feature', 'threshold', 
                                       'impurity', 'value', 'sizes', 'pos_size','pos_rate'])    
for i in list(range(len(gbdt.estimators_))):
    ets = gbdt.estimators_[i][0].tree_
    #节点分割属性    
    node_stat = pd.DataFrame({'feature': ets.feature,
                              'threshold': ets.threshold,
                              'impurity': ets.impurity,
                              'value':ets.value[:,0,0]})
    node_stat = node_stat.reset_index(drop=False).rename(columns={'index':'leaf_node'})
    #叶子节点样本量
    sizes = round(gbdt_tree.groupby('tree_{}'.format(i))['y'].count())
    sizes.name = 'sizes'
    #正样本量（坏人）
    pos_size = gbdt_tree.groupby('tree_{}'.format(i))['y'].sum()
    pos_size.name = 'pos_size'
    leaf_stat = node_stat.merge(sizes, left_index=True, right_index=True, how='left')
    leaf_stat = leaf_stat.merge(pos_size, left_index=True, right_index=True, how='left')
    leaf_stat = leaf_stat[~leaf_stat['pos_size'].isnull()]
    #正样本量比例
    leaf_stat['pos_rate'] = round(leaf_stat['pos_size']/leaf_stat['sizes'],4)
    leaf_stat['tree_name'] = 'tree_{}'.format(i)
    tree_leaf_stat = pd.concat([tree_leaf_stat, leaf_stat], sort=True, axis=0,ignore_index=True)
tree_leaf_stat = tree_leaf_stat[['tree_name', 'leaf_node','feature', 'threshold', 
                                 'impurity', 'value', 'sizes', 'pos_size','pos_rate']]



##=========================================================================================================##
##                                             输出决策树规则                                                 ##
##=========================================================================================================##


from sklearn.tree import _tree
## 定义输出决策树规则函数
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print ("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print ("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {} - {}".format(indent, tree_.value[node], node))

    recurse(0, 1)


## 决策树规则输出
for i in list(range(len(gbdt.estimators_))):
    print("tree_{}".format(i))
    tree_to_code(gbdt.estimators_[i][0], 
                 feature_names = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10',
                                  'v11','v12','v13','v14','v15','v16','v17','v18','v19','v20'])  
    print('='*80)




##=========================================================================================================##
##                                             决策树onehot                                                 ##
##=========================================================================================================##


from sklearn.preprocessing import OneHotEncoder

## 引入onehot函数
enc = OneHotEncoder()
## 训练集生成随机森林多棵决策树叶子节点
leaf = gbdt.apply(X_train)[:,:,0].astype('int')
## 叶子节点转化为二值变量
leaf_onehot_encode = enc.fit(leaf)
leaf_onehot_feature = leaf_onehot_encode.transform(leaf).toarray()
## 叶子节点二值变量与原始变量拼接
rf_X_train = np.hstack([leaf_onehot_feature, X_train])

































