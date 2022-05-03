# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:43:08 2020

@author: finup
"""



import numpy as np
np.random.seed(seed=12)  ## for reproducibility
dataset = np.random.binomial(1, 0.1, 20000000).reshape(2000,10000)  ## dummy data
y = np.random.binomial(1, 0.5, 2000)  ## dummy target variable

# 生成稀疏矩阵
from scipy.sparse import csr_matrix
sparse_dataset = csr_matrix(dataset)



# 评估压缩后的稀疏矩阵是否能够提高BernoulliNB的执行速度
from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB(binarize=None)
%timeit nb.fit(dataset, y)
%timeit nb.fit(sparse_dataset, y)

# 评估压缩后的稀疏矩阵是否能够提高LogisticRegression的执行速度
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=99)
%timeit lr.fit(dataset, y)
%timeit lr.fit(sparse_dataset, y)

# 评估压缩后的稀疏矩阵是否能够提高LinearSVC的执行速度
from sklearn.svm import LinearSVC
svc = LinearSVC()
%timeit svc.fit(dataset, y)
%timeit svc.fit(sparse_dataset, y)



##=========================================================================================================#
##                                              稀疏空间压缩(数据结构)
##=========================================================================================================#

## 因为零值没有太多的意义，所以我们可以忽略零值，并且仅需要存储或操作稀疏矩阵中数据或非零值。
## 有多种数据结构可用于有效地构造稀疏矩阵，下面列出了三个常见的例子。

## 数据字典转换（Dictionary of Keys）

## 行转换（List of Lists）
from scipy.sparse import csr_matrix
sparse_dataset = csr_matrix(dataset)
# 评估是否有效的进行压缩内存
import seaborn as sns
dense_size = np.array(dataset).nbytes/1e6
sparse_size = (sparse_dataset.data.nbytes + sparse_dataset.indptr.nbytes + sparse_dataset.indices.nbytes)/1e6
sns.barplot(['DENSE', 'SPARSE'], [dense_size, sparse_size])
plt.ylabel('MB')
plt.title('Compression') 

## 列转换（Coordinate List）




##=========================================================================================================#
##                                              稀疏矩阵压缩后的应用
##=========================================================================================================#
import numpy as np
np.random.seed(seed=12)  ## for reproducibility
dataset = np.random.binomial(1, 0.1, 20000000).reshape(2000,10000)  ## dummy data
y = np.random.binomial(1, 0.5, 2000)  ## dummy target variable

# 生成稀疏矩阵
from scipy.sparse import csr_matrix
sparse_dataset = csr_matrix(dataset)

# 评估是否有效的进行压缩内存
import seaborn as sns
dense_size = np.array(dataset).nbytes/1e6
sparse_size = (sparse_dataset.data.nbytes + sparse_dataset.indptr.nbytes + sparse_dataset.indices.nbytes)/1e6
sns.barplot(['DENSE', 'SPARSE'], [dense_size, sparse_size])
plt.ylabel('MB')
plt.title('Compression') 

# 评估压缩后的稀疏矩阵是否能够提高BernoulliNB的执行速度

# BernoulliNB: X为稀疏二项离散值
# GaussianNB：X为服从正态分布的连续变量矢量
# MultinomialNB：X为服从多项分布的矢量
# ComplementNB:对BernoulliNB的补充，处理非平衡样本
 
from sklearn.naive_bayes import BernoulliNB
nb = BernoulliNB(binarize=None)
%timeit nb.fit(dataset, y)
%timeit nb.fit(sparse_dataset, y)

# 评估压缩后的稀疏矩阵是否能够提高LogisticRegression的执行速度
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=99)
%timeit lr.fit(dataset, y)
%timeit lr.fit(sparse_dataset, y)

# 评估压缩后的稀疏矩阵是否能够提高LinearSVC的执行速度
from sklearn.svm import LinearSVC
svc = LinearSVC()
%timeit svc.fit(dataset, y)
%timeit svc.fit(sparse_dataset, y)



##=========================================================================================================#
##                                              业务知识压缩
##=========================================================================================================#

# 可以通过对数据的理解来合并或者删除部分列或者行



##=========================================================================================================#
##                                              降维算法
##=========================================================================================================#

## LDA

from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.datasets import make_classification

X, y = make_classification(n_samples = 1000, n_features = 20, n_informative = 15,
                           n_redundant = 5, random_state = 7, n_classes =2)

lda = LinearDiscriminantAnalysis(n_components = None)
lda.fit(X, y)
lda.transform(X)

lda.predict(X)
lda.predict_proba(X)
lda.predict_log_proba(X)







## PCA (PCA、IncrementalPCA、KernelPCA、MiniBatchSparsePCA、SparsePCA)

from sklearn.decomposition import PCA

from sklearn.decomposition import SparsePCA


## SVD

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np



## embedding

import torch



