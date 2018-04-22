# -*- coding:utf-8 -*-
import pandas as pd

from pandas import Series,DataFrame
from sklearn.cluster import KMeans #导入K均值聚类算法

datafile = '../output/result_file.csv' #待聚类的数据文件
processedfile = '../output/data_processed.csv' #数据处理后文件
apriori='../output/apriori.csv'
typelabel ={u'Proposed Units':'A', u'Revised Cost':'B', u'Number of Proposed Stories':'C',
            u'Number of Existing Stories':'D'}
k = 8 #需要进行的聚类类别数

#读取数据并进行聚类分析
data1 = pd.read_csv(datafile) #读取数据
data=(data1.iloc[:1000])
#数据的规范化
data=(data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
keys = list(typelabel.keys())
print('keys')


result = pd.DataFrame()

if __name__ == '__main__': #判断是否主窗口运行，如果是将代码保存为.py后运行，则需要这句，如果直接复制到命令窗口运行，则不需要这句。
  for i in range(len(keys)):
    #调用k-means算法，进行聚类离散化
    print(u'正在进行“%s”的聚类...' % keys[i])
    kmodel = KMeans(n_clusters = k) #n_jobs是并行数，一般等于CPU数较好
    kmodel.fit(data[[keys[i]]].as_matrix()) #训练模型

    r1 = pd.DataFrame(kmodel.cluster_centers_, columns = [typelabel[keys[i]]]) #聚类中心
    r2 = pd.Series(kmodel.labels_).value_counts() #分类统计
    r2 = pd.DataFrame(r2, columns = [typelabel[keys[i]]+'n']) #转为DataFrame，记录各个类别的数目
    print('222')
    print(pd.concat([r1, r2], axis = 1))
    print(typelabel[keys[i]])
    r = pd.concat([r1, r2], axis = 1).sort_values(typelabel[keys[i]]) #匹配聚类中心和类别数目
    r.index = [1, 2, 3, 4, 5, 6, 7, 8]

    r[typelabel[keys[i]]] = pd.rolling_mean(r[typelabel[keys[i]]], 2) #rolling_mean()用来计算相邻2列的均值，以此作为边界点。
    r[typelabel[keys[i]]][1] = 0.0 #这两句代码将原来的聚类中心改为边界点。
    result = result.append(r.T)

  result = result.sort_index() #以Index排序，即以A,B,C,D,E,F顺序排
  result.to_csv(processedfile)

#数据变换
  aprioriData=pd.DataFrame()
  for i in range(len(keys)):
    N=typelabel[keys[i]] #A,B,C,D
    C=keys[i]

    bool1 = result.loc[N, 1] <= data[C]
    bool2 = result.loc[N, 2] <= data[C]
    bool3 = result.loc[N, 3] <= data[C]
    bool4 = result.loc[N, 4] <= data[C]
    bool5 = result.loc[N, 5] <= data[C]
    bool6 = result.loc[N, 6] <= data[C]
    bool7 = result.loc[N, 7] <= data[C]
    bool8 = result.loc[N, 8] <= data[C]

    typeN=1*(bool1 & ~bool2)+2*(bool2 & ~bool3)+3*(bool3 &~bool4)+4*(bool4 &~bool5)+5*(bool5&~bool6)+6*(bool6&~bool7)+7*(bool7&~bool8)+8*(bool8)
    typeN=typeN.replace({1:N+'1',2:N+'2',3:N+'3',4:N+'4',5:N+'5',6:N+'6',7:N+'7',8:N+'8'})
    aprioriData=pd.concat([aprioriData,typeN],axis=1)

aprioriData.to_csv(apriori)