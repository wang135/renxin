# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 10:37:10 2020

@author: finup
"""

'''
非平衡样本的处理方法：
1、抽样：
   a.上采样（过采样）：增加样本比例过低类的样本数量，缺点：过拟合
   b.下采样（欠采样）：采用随机抽样的方法减少样本比例过大的样本数量，缺点：欠拟合
2、人工合成少数样本过采样技术（SMOTE）
   选择样本比例过低类的样本，对该类中的每个样本分别通过聚类分析，找出其距离最近的几个样本，
   采用距离随机加权的方法，计算生成新样本。
   (1)对于少数类中每一个样本x，以欧氏距离为标准计算它到少数类样本集中所有样本的距离，得到其k近邻。
   (2)根据样本不平衡比例设置一个采样比例以确定采样倍率N，对于每一个少数类样本x，从其k近邻中随机选择若干个样本，假设选择的近邻为xn。
   (3)对于每一个随机选出的近邻xn，分别与原样本按照如下的公式构建新的样本。
      x_new = x + rand(0,1)*(x_i - x)
3、模型组合（stacking）
   （1）并行模式：首先通过随机欠采样分别构建模型，然后进行模型组合
   （2）串行模式：

'''



