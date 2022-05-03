# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:24:45 2020

@author: finup
"""

import pandas as pd



##================================================================================================================##
##                                               Label encoding                         
##================================================================================================================##

##---------------------------  单变量 encoding

## 方法一：LabelEncoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

city_list = ["paris", "paris", "tokyo", "amsterdam"]

le.fit(city_list)
print(le.classes_)

city_list_le = le.transform(city_list)
print(city_list_le)

le.fit_transform(city_list)

city_list_new = le.inverse_transform(city_list_le)
print(city_list_new)


## 方法二：factorize
pd.factorize(city_list)


## 方法三：LabelBinarizer
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(city_list)
print(lb.classes_)
city_list_le = lb.transform(city_list)
print(city_list_le)

city_list_new = lb.inverse_transform(city_list_le)
print(city_list_new)






##---------------------------  多变量 encoding

df = pd.DataFrame({
    'pets': ['cat', 'dog', 'cat', 'monkey', 'dog', 'dog'], 
    'owner': ['Champ', 'Ron', 'Brick', 'Champ', 'Veronica', 'Ron'], 
    'location': ['San_Diego', 'New_York', 'New_York', 'San_Diego', 'San_Diego', 'New_York']
})

## 方法一：
df.apply(LabelEncoder().fit_transform)

## 方法二：
OneHotEncoder().fit_transform(df).toarray()


## 方法三：
from collections import defaultdict
d = defaultdict(LabelEncoder)
# Encoding the variable
fit = df.apply(lambda x: d[x.name].fit_transform(x))
# Inverse the encoded
fit.apply(lambda x: d[x.name].inverse_transform(x))
# Using the dictionary to label future data
df.apply(lambda x: d[x.name].transform(x))


## 方法四：
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


encoding_pipeline = Pipeline([
    ('encoding',MultiColumnLabelEncoder(columns=df.columns.tolist()))
    # add more pipeline steps as needed
])
encoding_pipeline.fit_transform(df)







##---------------------------  字典 encoding
	
from sklearn.feature_extraction import DictVectorizer

measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Fransisco', 'temperature': 18.},
]

vec = DictVectorizer()
measurements_vec = vec.fit_transform(measurements)
print(measurements_vec)

print(measurements_vec.toarray())

feature_names = vec.get_feature_names()
print(feature_names)







##================================================================================================================##
##                                               one-hot encoding                        
##================================================================================================================##


	
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame([
    ['green', 'Chevrolet', 2017],
    ['blue', 'BMW', 2015],
    ['yellow', 'Lexus', 2018],
])
df.columns = ['color', 'make', 'year']



def cols_one_hot(inDf, varLst, missValue='Miss', method = 'dummy'):
    '''
    Function Description:
        多个分类变量，进行one-hot转换
    
    Parameters
    ----------
    inDf     : 待处理数据框
    varLst   : 待处理变量列表
    missValue : 缺失值填充值
    
    Returns
    -------
    数据框：one-hot转换后的数据框
    
    '''
    df = inDf.copy()
    trans_df = df[varLst].fillna(missValue)
    if method == 'dummy':
        onehot = OneHotEncoder()
        onehot_df = pd.DataFrame(onehot.fit_transform(trans_df).toarray(),
                                 columns = onehot.get_feature_names(varLst), 
                                 dtype = 'int')
    elif method == 'dummy':
        onehot_df = pd.get_dummies(trans_df)
    
    df = df.reset_index(drop=True)
    df = pd.concat([df, onehot_df], axis=1)
    return df


one_hot_df = cols_one_hot(inDf = df, 
                          varLst = df.columns.tolist(), 
                          missValue='Miss')














