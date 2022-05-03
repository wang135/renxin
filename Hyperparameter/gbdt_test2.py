# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:02:08 2020

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
from sklearn.model_selection import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


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



train = pd.read_csv(r'F:\data\hyperparameter_data.csv')

target = 'Disbursed'
IDcol = 'ID'
predictors = [x for x in train.columns if x not in [target, IDcol]]


## =============================================
## 初始值
estimator0 = GradientBoostingClassifier(random_state=10)
modelfit(estimator0, train, predictors)

# 0.7459168


## =============================================
par_test = {'n_estimators': range(60,141,20)}
estimator = GradientBoostingClassifier(random_state=10)
gsearch1 = GridSearchCV(estimator=estimator,
                        param_grid=par_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch1.fit(train[predictors], train[target])
pd.DataFrame({'mean_test_score':gsearch1.cv_results_['mean_test_score'],
              'std_test_score':gsearch1.cv_results_['std_test_score'],
              'params':gsearch1.cv_results_['params']}), \
gsearch1.best_params_, gsearch1.best_score_

# 0.7885845974399508


## =============================================
par_test = {'n_estimators': [50, 60, 70],
             'learning_rate': [0.01,0.03,0.05]}
estimator = GradientBoostingClassifier(random_state=10)
gsearch2 = GridSearchCV(estimator=estimator,
                        param_grid=par_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch2.fit(train[predictors], train[target])
pd.DataFrame({'mean_test_score':gsearch2.cv_results_['mean_test_score'],
              'std_test_score':gsearch2.cv_results_['std_test_score'],
              'params':gsearch2.cv_results_['params']}), \
gsearch2.best_params_, gsearch2.best_score_

# 0.8019241432719472


## =============================================
#初始尝试使用0.6-1
par_test = {'max_depth': range(6,13,2)}
estimator = GradientBoostingClassifier(learning_rate=0.05, n_estimators=50,
                                       random_state=10)
gsearch3 = GridSearchCV(estimator=estimator,
                        param_grid=par_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch3.fit(train[predictors], train[target])
pd.DataFrame({'mean_test_score':gsearch3.cv_results_['mean_test_score'],
              'std_test_score':gsearch3.cv_results_['std_test_score'],
              'params':gsearch3.cv_results_['params']}), \
gsearch3.best_params_, gsearch3.best_score_

# 


## =============================================
par_test3 = {'min_samples_split': [2]}
estimator = GradientBoostingClassifier(max_features='sqrt', max_depth=4, subsample=0.6,
                                       random_state=10)
gsearch3 = GridSearchCV(estimator=estimator,
                        param_grid=par_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch3.fit(train[predictors], train[target])
pd.DataFrame({'mean_test_score':gsearch3.cv_results_['mean_test_score'],
              'std_test_score':gsearch3.cv_results_['std_test_score'],
              'params':gsearch3.cv_results_['params']}), \
gsearch3.best_params_, gsearch3.best_score_

# 无提升


## =============================================
par_test4 = {'max_features': [15,20,25,30,35,40,42,'sqrt','log2']}
estimator = GradientBoostingClassifier(max_depth=4, subsample=0.6,
                                       random_state=10)
gsearch4 = GridSearchCV(estimator=estimator,
                        param_grid=par_test4, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch4.fit(train[predictors], train[target])
pd.DataFrame({'mean_test_score':gsearch4.cv_results_['mean_test_score'],
              'std_test_score':gsearch4.cv_results_['std_test_score'],
              'params':gsearch4.cv_results_['params']}), \
gsearch4.best_params_, gsearch4.best_score_

# 0.7916961276558322


## =============================================
par_test5 = {'n_estimators': range(60,121,10)}
estimator = GradientBoostingClassifier(max_depth=4, subsample=0.6, max_features=35,
                                       random_state=10)
gsearch5 = GridSearchCV(estimator=estimator,
                        param_grid=par_test5, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch5.fit(train[predictors], train[target])
pd.DataFrame({'mean_test_score':gsearch5.cv_results_['mean_test_score'],
              'std_test_score':gsearch5.cv_results_['std_test_score'],
              'params':gsearch5.cv_results_['params']}), \
gsearch5.best_params_, gsearch5.best_score_

# 0.7916961276558322


## =============================================
par_test6 = {'learning_rate': [0.001,0.003,0.005,0.007, 0.01]}
estimator = GradientBoostingClassifier(max_depth=4, subsample=0.6, max_features=35,n_estimators=100,
                                       random_state=10)
gsearch6 = GridSearchCV(estimator=estimator,
                        param_grid=par_test6, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch6.fit(train[predictors], train[target])
pd.DataFrame({'mean_test_score':gsearch6.cv_results_['mean_test_score'],
              'std_test_score':gsearch6.cv_results_['std_test_score'],
              'params':gsearch6.cv_results_['params']}), \
gsearch6.best_params_, gsearch6.best_score_




## =============================================
par_test6 = {'learning_rate': [0.001,0.003,0.005,0.007, 0.01],
             'n_estimators': range(60,151,20)}
estimator = GradientBoostingClassifier(max_depth=4, subsample=0.6, max_features=35,
                                       random_state=10)
gsearch6 = GridSearchCV(estimator=estimator,
                        param_grid=par_test6, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch6.fit(train[predictors], train[target])
pd.DataFrame({'mean_test_score':gsearch6.cv_results_['mean_test_score'],
              'std_test_score':gsearch6.cv_results_['std_test_score'],
              'params':gsearch6.cv_results_['params']}), \
gsearch6.best_params_, gsearch6.best_score_





























