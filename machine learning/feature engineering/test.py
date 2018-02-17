# -*- coding:utf8 -*-
import pandas as pd

data = pd.read_csv('kaggle_bike_competition_train.csv', header=0, error_bad_lines=False)
# print data.head()
temp = pd.DatetimeIndex(data['datetime'])
data['date'] = temp.date
# print type(data.date)
# print data['date']==data.date
data['time'] = temp.time
temp = pd.to_datetime(data.time, format="%H:%M:%S")
data['hour'] = pd.Index(temp).hour
data['dayofweek'] = pd.DatetimeIndex(data.date).dayofweek
data['dateDays'] = (data.date - data.date[0]).astype('timedelta64[D]')
byday = data.groupby('dayofweek')
# print byday['casual'].sum()
#
# print byday['registered'].sum()
data['Saturday'] = 0
data.Saturday = 1
data['Sunday'] = 0
data.Saturday[data.dayofweek == 5] = 1
data.Sunday[data.dayofweek == 6] = 1
# 删除没用的列，drop函数中axis =1 表示删列，=0表示删行
dataRel = data.drop(['datetime', 'count', 'date', 'time', 'dayofweek'], axis=1)

from sklearn.feature_extraction import DictVectorizer

featureConCols = ['temp', 'atemp', 'humidity', 'windspeed', 'dateDays', 'hour']
dataFeatureCon = dataRel[featureConCols]
dataFeatureCon = dataFeatureCon.fillna('NA')
# print dataFeatureCon
X_dictCon = dataFeatureCon.T.to_dict().values()  # .T表示 转置

featureCatCols = ['season', 'holiday', 'workingday', 'weather', 'Saturday', 'Sunday']
dataFeatureCat = dataRel[featureCatCols]
# print type(dataFeatureCat)
X_dictCat = dataFeatureCat.T.to_dict()  # .T表示 转置
print type(X_dictCat)
X_dictCat = X_dictCat.values()
# print X_dictCat
vec = DictVectorizer(sparse=False)
X_vec_cat = vec.fit_transform(X_dictCat)
print X_vec_cat
# print X_vec_cat
X_vec_con = vec.fit_transform(X_dictCon)
# print dataFeatureCon.head()
# print type(dataFeatureCat)
# print data

from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_vec_con)
# print type(scaler)
X_vec_con = scaler.transform(X_vec_con)
# print X_vec_con.shape
enc = preprocessing.OneHotEncoder().fit(X_vec_cat)
X_vec_cat = enc.transform(X_vec_cat).toarray()
# print X_vec_cat.toarray()
import numpy as np

X_vec = np.concatenate((X_vec_con, X_vec_cat), axis=1)
Y_vec_reg = dataRel['registered'].values.astype(float)
print type(dataRel['casual'])
Y_vec_cas = dataRel['casual'].values.astype(float)
# print data
