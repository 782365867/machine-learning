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
data['dateDays'] = (data.date - data.date[0])
print data

# print data
