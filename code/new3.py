# -*- coding:utf-8 -*-
import operator

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from pandas import Series,DataFrame
from sklearn.cluster import KMeans #导入K均值聚类算法


db='Building_Permits.csv'
clean_file='../output/clean_file.csv'
#读取csv文件，生成data frame
result_file='result_file.csv'
fillno='../output/fillno.csv'
data=pd.read_csv(db,low_memory=False)

#查看前十条数据内容

#print(data.iloc[:10])

# 定义两类数据：标称型和数值型

# attri1=DataFrame(data,columns=['Proposed Units','Revised Cost','Revised Cost','Number of Proposed Stories',
#                                'Number of Existing Stories',
# 'Proposed Construction Type','Existing Construction Type','Plansets',
#                              'Current Status','Street Name','Site Permit'])
attri=DataFrame(data,columns=['Proposed Units','Revised Cost','Revised Cost','Number of Proposed Stories',
                               'Number of Existing Stories'])
name_value=DataFrame(data,columns=['Proposed Units','Revised Cost','Revised Cost','Number of Proposed Stories',
                               'Number of Existing Stories'])
name_cate=DataFrame(data,columns=['Site Permit',
'Proposed Construction Type','Existing Construction Type','Plansets',
                             'Current Status','Street Name'])
#attri.to_csv(clean_file)

explore=attri.describe(percentiles=[],include='all').T
explore['null']=len(attri)-explore['count']
explore=explore[['null','max','min']]
explore.columns=[u'空值数',u'最大值',u'最小值']
explore.to_csv(result_file)

#attri['Site Permit'] = attri['Site Permit'].fillna('N')

# 通过众数来填充缺失值
for i in range(0,5):
    MostFrequentElement = attri.iloc[:,[i]].apply(pd.value_counts).idxmax()
    attri.iloc[:, [i]] = attri.iloc[:,[i]].fillna(value=MostFrequentElement)
    print('success')
    print(attri.info())
#attri.to_csv(fillno)

#l
# for i in range(0,10):
#     print('频数为:\n', name_cate.iloc[:,[i]].apply(pd.value_counts).sum(), '\n')
attri.to_csv(result_file)



