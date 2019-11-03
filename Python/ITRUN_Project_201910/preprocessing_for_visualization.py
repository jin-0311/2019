# 데이터 출처 : https://data.seoul.go.kr/dataList/datasetView.do?infId=OA-15182&srvType=F&serviceKind=1&currentPageNo=1
# 2017.09.01~2018.08.31 (training dataset visualization)

import pandas as pd
import numpy as np
import xlrd


# 전체 데이터 20170901~20180831
import pandas as pd
import numpy as np
import xlrd

d1=pd.read_csv('./pre/data/train/1.csv',
               usecols=[0,1,2,4,9,10],
               names=['bike_n','start_date','start_p','start_now','using_t','dist'],
               header=0,
               encoding='euc-kr')
d1.head()
d1.tail()
d1.columns

d2=pd.read_csv('./pre/data/train/2.csv',
               usecols=[0, 1, 2, 4, 9, 10],
               names=['bike_n', 'start_date', 'start_p', 'start_now', 'using_t', 'dist'],
               header=0,
               encoding='euc-kr')
d2.head()
d2.tail()

d3=pd.read_csv('./pre/data/train/3.csv',
               usecols=[0, 1, 2, 4, 9, 10],
               names=['bike_n', 'start_date', 'start_p', 'start_now', 'using_t', 'dist'],
               header=0,
               encoding='euc-kr')
d3.head()
d3.tail()

d4=pd.read_csv('./pre/data/train/4.csv',
               usecols=[0, 1, 2, 4, 9, 10],
               names=['bike_n', 'start_date', 'start_p', 'start_now', 'using_t', 'dist'],
               header=0,

               encoding='euc-kr')
d4.head()
d4.tail()

d5=pd.read_csv('./pre/data/train/5.csv',
               usecols=[0, 1, 2, 4, 9, 10],
               names=['bike_n', 'start_date', 'start_p', 'start_now', 'using_t', 'dist'],
               header=0,

               encoding='euc-kr')
d5.head()
d5.tail()

d6=pd.read_csv('./pre/data/train/6.csv',
               usecols=[0, 1, 2, 4, 9, 10],
               names=['bike_n', 'start_date', 'start_p', 'start_now', 'using_t', 'dist'],
               header=0,

               encoding='euc-kr')
d6.head()
d6.tail()

d7=pd.read_csv('./pre/data/train/7.csv',
               usecols=[0, 1, 2, 4, 9, 10],
               names=['bike_n', 'start_date', 'start_p', 'start_now', 'using_t', 'dist'],
               header=0,

               encoding='euc-kr')
d7.head()
d7.tail()

d8=pd.read_csv('./pre/data/train/8.csv',
               usecols=[0, 1, 2, 4, 9, 10],
               names=['bike_n', 'start_date', 'start_p', 'start_now', 'using_t', 'dist'],
               header=0,

               encoding='euc-kr')
d8.head()
d8.tail()



d9=pd.read_excel('./pre/data/train/9.xlsx',
                 usecols=[0, 1, 2, 4, 9, 10],
                 names=['bike_n', 'start_date', 'start_p', 'start_now', 'using_t', 'dist'],
                 header=0,
 encoding='euc-kr')

d9.head()
d9.tail()

d10=pd.read_excel('./pre/data/train/10.xlsx',
                  usecols=[0, 1, 2, 4, 9, 10],
                  names=['bike_n', 'start_date', 'start_p', 'start_now', 'using_t', 'dist'],
                  header=0,
                  encoding='euc-kr')
d10.head()
d10.tail()

d11=pd.read_excel('./pre/data/train/11.xlsx',
                  usecols=[0, 1, 2, 4, 9, 10],
                  names=['bike_n', 'start_date', 'start_p', 'start_now', 'using_t', 'dist'],
                  header=0,

                  encoding='euc-kr')
d11.head()
d11.tail()

data=pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11], ignore_index=True)
data.info()
data.tail()


data=data.rename(columns={data.columns[1]: 'date',
                     data.columns[2]: 'place'})

data.sample(10)
data['place']

data['place']=data['place'].astype(str)
data['place']=data['place'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)

data['date']=data['date'].astype(str)
data['date']=data['date'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)

data['date']
data['date']=pd.to_datetime(data['date'], format='%Y%m%d%H%M%S')

data['bike_n']=data['bike_n'].astype(str)
data['bike_n']=data['bike_n'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
data['bike_n']


data.to_csv('./pre/data/raw_data_201709_201808.csv',header=True)





import numpy as np
import pandas as pd

gu=pd.read_excel('./pre/data/gu_name.xlsx', names=['gu_name', 'place'],usecols=[0,1])
gu.head()
gu.tail()

gu.to_csv('./pre/data/gu_name_.csv',header=True)


data=pd.read_csv('./pre/data/raw_data_201709_201808.csv')
name=pd.read_csv('./pre/data/gu_name_.csv')


data.head()
del data['Unnamed: 0']
del data['Unnamed: 0.1']
data.columns
del data['bike_n']
del data['start_now']

data['date']=pd.to_datetime(data['date'], format='%Y-%m-%d')
data.index=data['date']
del data['date']




import datetime
data['year_month']=0
data['year_month']=data.index.strftime('%Y%m')

data.head()
data.index=data['year_month']
del data['year_month']



count_data=data.groupby(['place','dist','using_t'] )['place'].count().reset_index(name='count')
count_data

count2=pd.pivot_table(count_data, index=['place'],
                      columns=['dist','using_t'],
                      values=['dist','using_t'],
                      aggfunc=[np.mean, np.mean])

print(count2)


data=pd.read_csv('./pre/data/raw_data_201709_201808.csv')
data.info()
data['place']=data['place'].astype(str)
data['date']
del data['start_now']


gu=gu[gu.gu_name != '합계']
gu['place']=gu['place'].astype(str)


gu_data=pd.merge(data,gu, how='outer',on='place' )
gu_data.head()
data['place']=data['place'].astype(str)


# place별 place count , using time평균, dist 평균
place=pd.pivot_table(data, values='place', aggfunc='sum', fill_value=0, index='date')
place

# bike별 시간별 place
bikenum=data.groupby()



import numpy as np
import pandas as pd


data.head()
del data['Unnamed: 0']
del data['Unnamed: 0.1']
data.columns
del data['bike_n']
del data['start_now']

data['date']=pd.to_datetime(data['date'], format='%Y-%m-%d')
data.index=data['date']
del data['date']

# place정리
data['place'].isnull().sum()
data['place'].unique()
# data['place'].fillna(0)

# data['place']=data['place'].astype(str)

import datetime
data['year_month']=0
data['year_month']=data.index.strftime('%Y%m')


data.head()
data.index=data['year_month']
del data['year_month']

count_data=data.groupby([data.index,'place','dist','using_t'] )['place'].count().reset_index(name='count')
count_data.shape
count_data.to_csv('./pre/data/visual/count_.csv')


# 수정
count_data=pd.read_csv('./pre/data/visual/count_.csv')
count_data
del count_data['Unnamed: 0']



month_data=count_data.groupby(['year_month','place'])['count','dist','using_t'].sum()
month_data.to_csv('./pre/data/visual/all_u_want_3sum.csv', header=True)




# 구    년도/월    count평균 거리평균 사용시간평균


data.shape
name.head()
del name['Unnamed: 0']

month_data=pd.read_csv('./pre/data/visual/all_u_want_3sum.csv')

month_data.head()
month_data['place']=month_data['place'].astype(int)
month_data.shape

test=pd.merge(month_data, name, how='inner', on='place')
test.shape
test1=test.groupby(['gu_name','year_month'])['count','dist','using_t'].mean()
test1
test1.to_csv('./pre/data/visual/all_u_want_4sum.csv', header=True)




# 구별 요일별 아침3개, 저녁3개
data.head()
del data['Unnamed: 0']
del data['Unnamed: 0.1']
data.columns
del data['bike_n']
del data['start_now']
del data['using_t']
del data['dist']

data['date']=pd.to_datetime(data['date'], format='%Y-%m-%d')
data.index=data['date']
del data['date']

data.head()







#final.loc[(final.date == '0924') | (final.date == '0925') | (final.date == '0926') | (final.date == '1003') | (
 #       final.date == '1009') | (final.date == '1225'), 'holiday'] = 1


# 구/ 날짜/ 요일 / 시간대별총6개 / count
data['day']=0
data['day']=data.index.strftime('%a')

data['time_range']=0
data['time_range']=data.index.strftime('%H')
data['time_range']=data['time_range'].astype(int)

data['date']=0
data['date']=data.index.strftime('%Y-%m-%d')

data.head()
data.sample(10)

dataset=data.loc[(data.time_range == 6) | (data.time_range == 7) | (data.time_range == 8) | (data.time_range == 18) | (
        data.time_range == 19) | (data.time_range == 20)]

dataset.info()
dataset.head()
dataset.index=dataset['date']
count_data=dataset.groupby([dataset.index,'place','day','time_range'] )['place'].count().reset_index(name='count')

count_data['place']=count_data['place'].astype(int)
count_data.info()  # 1045228 개
name.head()
del name['Unnamed: 0']

test=pd.merge(count_data, name, how='inner', on='place')
test.shape

test

boram=test.groupby(['gu_name','date','day','time_range'])['count'].sum().reset_index(name='total')
boram.to_csv('./pre/data/visual/boram_wants.csv', header=True)


boram1=boram.groupby(['gu_name','date','day','time_range'])['place','count'].sum().reset_index(name='total')
boram1.to_csv('./pre/data/visual/boram_wants1.csv', header=True)