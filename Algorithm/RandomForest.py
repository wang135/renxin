# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:02:48 2020

@author: finup
"""

import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_auc_score
import graphviz

X,y = make_classification(n_samples=40000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

rf = RandomForestClassifier(max_depth=3, n_estimators=10)
rf.fit(X_train, y_train)
y_pred = rf.predict_proba(X_train)[:,1]

roc_auc_score(y_train, y_pred)


##====== 超参调优
## https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/random_forest_explained/Improving%20Random%20Forest%20Part%202.ipynb
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

## ------RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier(random_state = 42)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 10, scoring='neg_mean_absolute_error', 
                              cv = 4, verbose=1, random_state=42, n_jobs=-1,
                              return_train_score=True)
# Fit the random search model
rf_random.fit(X_train, y_train)
# The best parameters of random search
rf_random.best_params_
# cross validation result
rf_random.cv_results_

# evaluation function 
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

# evaluate the default model
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(train_features, train_labels)
base_accuracy = evaluate(base_model, test_features, test_labels

# evaluate the best random search model
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, test_features, test_labels)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


## ------GridSearchCV
from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}


# Create a base model
rf = RandomForestClassifier(random_state = 42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
# get the best parameters
grid_search.best_params_
# evaluate the best model from grid search 
grid_accuracy = evaluate(best_grid, test_features, test_labels)

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))









##=========================================================================================================##
##                                         重要输出变量解释                                                 ##
##=========================================================================================================##

## 各棵决策树
rf.estimators_
## 变量重要性
rf.feature_importances_  

## 生成每颗决策树的叶子节点
rf.apply(X_train)


## 生成预测分类
rf.predict(X_train)
## 生成预测概率
rf.predict_proba(X_train)




##=========================================================================================================##
##                                             决策树图输出                                                 ##
##=========================================================================================================##
## 决策树图输出
for i in list(range(len(rf.estimators_))):
    
    i = 0
    estimator = rf.estimators_[i]
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
    graph.render(cleanup=True, filename = 'tree_{}'.format(estimator.get_params()['random_state']), directory = r'F:\Python\Test', format = 'pdf')
 


##=========================================================================================================##
##                                          决策树叶子节点统计量                                            ##
##=========================================================================================================##

## 随机森林结果转化为决策树节点
rf_leaf = pd.DataFrame(rf.apply(X_train), 
                       columns=[('tree_' + str(i)) for i in range(len(rf.estimators_))])
rf_leaf['y'] = y_train
##随机森林的决策树叶子节点统计量
tree_leaf_stat = pd.DataFrame(columns=['tree_name','random_state','leaf_node', 'feature', 'threshold', 
                                       'impurity', 'sizes', 'pos_size','pos_rate'])    
for i in list(range(len(rf.estimators_))):

    
    ets = rf.estimators_[i].tree_
    #节点分割属性    
    node_stat = pd.DataFrame({'feature': ets.feature,
                              'threshold': ets.threshold,
                              'impurity': ets.impurity})
    node_stat = node_stat.reset_index(drop=False).rename(columns={'index':'leaf_node'})
    #叶子节点样本量
    sizes = round(rf_leaf.groupby('tree_{}'.format(i))['y'].count())
    sizes.name = 'sizes'
    #正样本量（坏人）
    pos_size = rf_leaf.groupby('tree_{}'.format(i))['y'].sum()
    pos_size.name = 'pos_size'
    leaf_stat = node_stat.merge(sizes, left_index=True, right_index=True, how='left')
    leaf_stat = leaf_stat.merge(pos_size, left_index=True, right_index=True, how='left')
    leaf_stat = leaf_stat[~leaf_stat['pos_size'].isnull()]
    #正样本量比例
    leaf_stat['pos_rate'] = round(leaf_stat['pos_size']/leaf_stat['sizes'],4)
    leaf_stat['tree_name'] = 'tree_{}'.format(i)
    leaf_stat['random_state'] = rf.estimators_[i].get_params()['random_state']
    tree_leaf_stat = pd.concat([tree_leaf_stat, leaf_stat], ignore_index=True)
tree_leaf_stat = tree_leaf_stat[['tree_name','random_state','leaf_node', 'feature', 'threshold', 
                                 'impurity', 'sizes', 'pos_size','pos_rate']]




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
for i in list(range(len(rf.estimators_))):
    print("tree_{}".format(i))
    tree_to_code(rf.estimators_[i], 
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
leaf = rf.apply(X_train)
## 叶子节点转化为二值变量
leaf_onehot_encode = enc.fit(leaf)
leaf_onehot_feature = leaf_onehot_encode.transform(leaf).toarray()
## 叶子节点二值变量与原始变量拼接
rf_X_train = np.hstack([leaf_onehot_feature, X_train])






























  