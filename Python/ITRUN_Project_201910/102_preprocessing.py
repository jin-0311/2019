import numpy as np
import pandas as pd

data=pd.read_csv('./102/data/raw_data_201709_201808.csv', usecols=[3,4])
# 201709~201808
data.head()
b1709_1808=data.loc[data['place']==102.0]
b1709_1808['place']=b1709_1808['place'].astype(int)
b1709_1808

a=pd.read_csv('./102/data/2015.csv', encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a.head()
a.tail()
b2015=a.loc[a['place']=='102']
b2015

a=pd.read_csv('./102/data/20161.csv', encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a.head()
a.tail()
b20161=a.loc[a['place']=='102']
b20161

a=pd.read_csv('./102/data/20162.csv', encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a.head()
a.tail()
b20162=a.loc[a['place']=='102']
b20162


a=pd.read_csv('./102/data/20163.csv', encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a.head()
a.tail()
b20163=a.loc[a['place']=='102']
b20163

a=pd.read_csv('./102/data/20171.csv', encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a.head()
a.tail()
b20171=a.loc[a['place']=='102']
b20171

a=pd.read_csv('./102/data/20172.csv', encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a.head()
a.tail()
a['place']=a['place'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
b20172=a.loc[a['place']=='102']
b20172

a=pd.read_csv('./102/data/20173.csv', encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a.head()
a.tail()
a['place']=a['place'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
b20173=a.loc[a['place']=='102']
b20173

a=pd.read_csv('./102/data/20174.csv', encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a.head()
a.tail()
a['place']=a['place'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
b20174=a.loc[a['place']=='102']
b20174

df2015_2018=pd.concat([b2015,b20161,b20162,b20163,b20171,b20172,b20173,b20174,b1709_1808], axis=0)
df2015_2018.reset_index()

df2015_2018.to_csv('./102/data/2015_2018.csv',header=True)

 
a=pd.read_excel('./102/data/2018091.xlsx',encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a['place']=a['place'].astype(str)
a['place']=a['place'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
a1=a.loc[a['place']=='102']
a1



a=pd.read_excel('./102/data/2018092.xlsx',encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a
a['place']=a['place'].astype(str)
a['place']=a['place'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
a2=a.loc[a['place']=='102']
a2

 
a=pd.read_excel('./102/data/2018101.xlsx',encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a['place']=a['place'].astype(str)
a['place']=a['place'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
a3=a.loc[a['place']=='102']
a3



a=pd.read_excel('./102/data/2018102.xlsx',encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a['place']=a['place'].astype(str)
a['place']=a['place'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
a4=a.loc[a['place']=='102']
a4

a=pd.read_excel('./102/data/201811.xlsx',encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a['place']=a['place'].astype(str)
a['place']=a['place'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
a5=a.loc[a['place']=='102']
a5


a=pd.read_csv('./102/data/201812.csv', encoding='euc-kr', usecols=[1,2], names=['date','place'], header=0)
a.head()
a['place']=a['place'].astype(str)
a['place']=a['place'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
a6=a.loc[a['place']=='102']
a6


a_all=pd.concat([a1,a2,a3,a4,a5,a6], axis=0)

a_all

df102=pd.concat([df2015_2018,a_all],axis=0)

df102


df102['date']=pd.to_datetime(df102['date'])
df102.index=df102['date']
del df102['date']

df102.to_csv('./df102.csv',header=True)
df102['date']=0

df102['date']=df102.index.strftime('%Y%m%d')
df102

df102['h']=0
df102['h']=df102.index.strftime('%H')
df102.to_csv('./df102hour.csv',header=True)

eight=df102.loc[df102['h']=='08']
eight.index=eight['date']
eight.info()
#del eight['place']
eight
count_data=eight.groupby(eight['date'] )['date'].count().reset_index(name='count')
count_data

count_data.index=count_data['date']
count_data
del count_data['date']
count_data.to_csv('./102final.csv',header=True)

a=count_data.copy()
a.index = pd.to_datetime(a.index, format='%Y-%m-%d')
a = a.reindex(pd.date_range('2015-09-26', '2018-12-31'), fill_value=0)
