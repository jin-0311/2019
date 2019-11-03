# 데이터 출처 : https://data.seoul.go.kr/dataList/datasetView.do?infId=OA-15182&srvType=F&serviceKind=1&currentPageNo=1

# preprocessing the bike
# 2017.09.01~ 2018.08.31 : Train DataSet

# < collecting the data
import pandas as pd
import numpy as np
import xlrd

d1=pd.read_csv('./pre/data/train/1.csv',
               usecols=[1,2,9,10],
               names = ['startdate', 'place', 'using_time', 'dist'],
               header=1,
               encoding='euc-kr')
d1.head()
d1.tail()

d2=pd.read_csv('./pre/data/train/2.csv',
               usecols=[1,2,9,10],
               names = ['startdate', 'place', 'using_time', 'dist'],
               header=1,
               encoding='euc-kr')
d2.head()
d2.tail()

d3=pd.read_csv('./pre/data/train/3.csv',
               usecols=[1,2,9,10],
               names = ['startdate', 'place', 'using_time', 'dist'],
               header=1,
               encoding='euc-kr')
d3.head()
d3.tail()

d4=pd.read_csv('./pre/data/train/4.csv',
               usecols=[1,2,9,10],
               names = ['startdate', 'place', 'using_time', 'dist'],
               header=1,
               encoding='euc-kr')
d4.head()
d4.tail()

d5=pd.read_csv('./pre/data/train/5.csv',
               usecols=[1,2,9,10],
               names = ['startdate', 'place', 'using_time', 'dist'],
               header=1,
               encoding='euc-kr')
d5.head()
d5.tail()

d6=pd.read_csv('./pre/data/train/6.csv',
               usecols=[1,2,9,10],
               names = ['startdate', 'place', 'using_time', 'dist'],
               header=1,
               encoding='euc-kr')
d6.head()
d6.tail()

d7=pd.read_csv('./pre/data/train/7.csv',
               usecols=[1,2,9,10],
               names = ['startdate', 'place', 'using_time', 'dist'],
               header=1,
               encoding='euc-kr')
d7.head()
d7.tail()

d8=pd.read_csv('./pre/data/train/8.csv',
               usecols=[1,2,9,10],
               names = ['startdate', 'place', 'using_time', 'dist'],
               header=1,
               encoding='euc-kr')
d8.head()
d8.tail()


import pandas as pd
import numpy as np
import xlrd

d9=pd.read_excel('./pre/data/train/9.xlsx', usecols=[1,2,9,10], names=['startdate', 'place', 'using_time', 'dist'], encoding='euc-kr')

d9.head()
d9.tail()

d10=pd.read_excel('./pre/data/train/10.xlsx', usecols=[1,2,9,10], names=['startdate', 'place', 'using_time', 'dist'],
                 encoding='euc-kr')
d10.head()
d10.tail()

d11=pd.read_excel('./pre/data/train/11.xlsx', usecols=[1,2,9,10], names=['startdate', 'place', 'using_time', 'dist'],
                 encoding='euc-kr')
d11.head()
d11.tail()

data=pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11], ignore_index=True)
data.info()
data.tail()
data['place']=data['place'].astype(str)
data['place']=data['place'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)


'''
rawdata=data.to_csv('./pre/data/output/rawdata.csv', header=True)

data.rename(columns= {data.columns[0] :'startdate'}, inplace=True )
data.head()

'''

# 구디역만 뽑아내기
data.tail()
data.head()
gudi=data[data['place'].isin(['282','1828','1854','1856','1911','1924','1955','1986','2012','2110','2113','2115'])]
gudi
pd.value_counts(gudi['place'])
gudi.info()
raw=gudi.copy()

# 구디 역 12개중 8개 사용
'''
a=raw[raw['place']==282]  # 0        2018-10-18 18:34
b=raw[raw['place']==1854] # 0       2018-10-18 7:29
c=raw[raw['place']==1856] # 0       2018-09-15 1:25
d=raw[raw['place']==1986] # 0
e=raw[raw['place']==1828] # 1703       2017-05-23 11:40
f=raw[raw['place']==1911] # 13585      2017-05-31 17:50
g=raw[raw['place']==1924] # 2105        2017 09 01
h=raw[raw['place']==2012] # 4275      2017 08 29
i=raw[raw['place']==2110] # 2910       2017 08 29
j=raw[raw['place']==2113] # 6422       2017-07-27 15:35
k=raw[raw['place']==2115] # 3233       2017 07 03
l=raw[raw['place']==1995] # 5362

'''
gudi.head()
gudi.head()
gudi=gudi.reset_index(drop=True)

gudi.head()
gudi.tail()

gudi.isnull().sum()


# 한글만 추출하기
s=['상암센터','123'][1]
h=re.compile('[ㄱ-ㅣ가-힣]+') # 한글과 띄어쓰기
result=h.findall(s)
result

# 연습
s=gudi['place'][0]
s
h=re.compile('[ㄱ-ㅣ가-힣]+')
result=h.findall(s)
result

hangul=[]
for i in range(len(gudi)-1):
    s=gudi['place'][i]
    h=re.compile('[ㄱ-ㅣ가-힣]+')
    result=h.findall(s)
    if result != []:
        hangul.append(result)
hangul  # 없다고 나옴


# data로 해보기 너무 오래걸림
hangul=[]
for i in range(len(data)-1):
    s=data['place'][i]
    h=re.compile('[ㄱ-ㅣ가-힣]+')
    result=h.findall(s)
    if result != []:
        hangul.append(result)
hangul
'''
# 더 추가 place가 문자열인것 찾아서 변경하기
data.loc[data['place']=='중랑센터', 'place'] = 934
data.loc[data['place']=='위트콤', 'place'] = 935
data.loc[data['place']=='상암센터 정비실', 'place'] = 936
'''

# 특수문자 제거
gudi['using_time']=gudi['using_time'].astype(str)
gudi['using_time']=gudi['using_time'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
gudi['using_time']=gudi['using_time'].astype(int)

gudi['dist']=gudi['dist'].astype(str)
gudi['dist']=gudi['dist'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)
gudi['dist']=gudi['dist'].astype(int)


gudi['using_time'].describe()
gudi['dist'].describe()

import datetime
gudi['startdate']=pd.to_datetime(gudi['startdate'])
gudi.info()


gudi['date'] =0
gudi['day']=0
gudi['hour']=0

gudi.sample(10)
'''

for i in range(len(gudi) - 1):
    gudi['date'][i] = gudi['startdate'][i].strftime('%m%d')

for i in range(len(gudi) - 1):
    gudi['day'][i] = gudi['startdate'][i].strftime('%a')

for i in range(len(gudi) - 1):
    gudi['hour'][i] = gudi['startdate'][i].hour
    
-> 이렇게 말고, datetime을 인덱스로 변경해서 뽑아내면 금방함! 

'''

gudi.to_csv('./pre/data/output/gudi7.csv',header=True)
gudi.head()
gudi.tail()

count_data=gudi.groupby(['date','day','hour','place'] )['place'].count().reset_index(name='count')
count_data
count_data.to_csv('./pre/data/output/gudi_count2.csv',header=True)

eight=count_data.loc[count_data['hour']==8,:]
eight=eight.reset_index(drop=True)
eight

eight.to_csv('./pre/data/output/eight2.csv',header=True)



# holiday 공휴일, 주말
eight['holiday']=0
eight

eight.loc[eight['day']=='Sat', 'holiday'] = 1
eight.loc[eight['day']=='Sun', 'holiday'] = 1

eight.sample(10)
eight.info()


# 안됨 eight['test']=eight['holiday'].apply(lambda x:1 if x in gong else 0)
pd.value_counts(eight['holiday'])

eight.loc[(eight.date ==1003) | (eight.date ==1004) |(eight.date ==1005)|(eight.date ==1006)|(eight.date ==1009)|(
        eight.date ==1225)|(eight.date ==101)|(eight.date ==215)|(eight.date ==216)|(eight.date ==217)|(eight.date
                                                                                                         ==301)|(
        eight.date ==505)|(eight.date ==507)|(eight.date ==522)|(eight.date ==606)|(eight.date ==613)|(eight.date
                                                                                                         ==815),
          'holiday'] = 1

eight.loc[eight['day']=='Sat', 'holiday'] = 1
eight.loc[eight['day']=='Sun', 'holiday'] = 1


eight.info()
eight.head()
eight.to_csv('./pre/data/output/final_eight.csv',header=True)


del eight['test']
eight['place'].unique()

p1=eight.loc[eight['place']=='1911',:]
p1  # 238

p2=eight.loc[eight['place']=='1924',:]
p2 #205

p3=eight.loc[eight['place']=='2115',:]
p3  # 186

p4=eight.loc[eight['place']=='2113',:]
p4  #123

p5=eight.loc[eight['place']=='2110',:]
p5 #110

p6=eight.loc[eight['place']=='1955',:]
p6 #147

p7=eight.loc[eight['place']=='2012',:]
p7 #143

p8=eight.loc[eight['place']=='1828',:]
p8  #145

p1.to_csv('./pre/data/output2/p1.csv',header=True)
p2.to_csv('./pre/data/output2/p2.csv',header=True)
p3.to_csv('./pre/data/output2/p3.csv',header=True)
p4.to_csv('./pre/data/output2/p4.csv',header=True)
p5.to_csv('./pre/data/output2/p5.csv',header=True)
p6.to_csv('./pre/data/output2/p6.csv',header=True)
p7.to_csv('./pre/data/output2/p7.csv',header=True)
p8.to_csv('./pre/data/output2/p8.csv',header=True)


import datetime
p1['tmpp']=pd.to_datetime(p1['tmpp'], format='%Y%m%d')  # 101 -> 1002가 됨 ㅜㅜ 수정
p1['tmp']=20170
p1['tmp']=p1['tmp'].astype(str)
p1['date']=p1['date'].astype(str)
p1['tmpp']=0


p1['tmpp']=p1['tmp']+p1['date']
p1.info()
p1['date']=p1['date'].astype(int)

a=p1[p1['date'] >=101][p1['date']<901]
b=p1[p1['date'] >= 901][p1['date'] <1001]
c=p1[p1['date'] >1001]


a['tmp']=20180
a['tmp'] = a['tmp'].astype(str)
a['date']=a['date'].astype(str)
a['date']=a['tmp']+a['date']
a

b['tmp']=20170
b['tmp'] = b['tmp'].astype(str)
b['date']=b['date'].astype(str)
b['date']=b['tmp']+b['date']
b

c['tmp']=2017
c['tmp'] = c['tmp'].astype(str)
c['date']=c['date'].astype(str)
c['date']=c['tmp']+c['date']
c


## p111마지막 단계 확인후 함수로
p11=pd.concat([a,b,c],ignore_index=True)
p11
p11=p11.iloc[:,[0,1,3,4,5] ]
p11
p11['date']=pd.to_datetime(p11['date'], format='%Y%m%d')
p11.info()
p11.to_csv('./pre/data/output2/p111.csv',header=True)

p111=pd.read_csv('./pre/data/output2/p111.csv', index_col='date')


p111.index=pd.DatetimeIndex(p111.index)
p112=p11.copy()
p112.index=p112['date']

del p112['date']
p112=p112.reindex(pd.date_range('2017-09-01','2018-08-31'), fill_value=0)

p112.index=pd.to_datetime(p112.index)




p111=p111.reindex(pd.date_range('2017-09-01','2018-08-31'), fill_value=0)
p111
p111.index=pd.to_datetime(p111.index)

del p111['Unnamed: 0']

del p111['day']

p111['day']='0'

for i in range(len(p111)):
    p111['day'][i] = p111.index[i].strftime('%a')



p111['place'] =1911
p111['date']=p111.index.strftime('%m%d')
p111

p111.loc[p111['day']=='Sat', 'holiday'] = 1
p111.loc[p111['day']=='Sun', 'holiday'] = 1
pd.value_counts(p111['holiday'])

p111.loc[(p111.date =='1003') | (p111.date =='1004') |(p111.date =='1005')|(p111.date =='1006')|(p111.date =='1009')|(
        p111.date =='1225')|(p111.date =='0101')|(p111.date =='0215')|(p111.date =='0216')|(p111.date =='0217')|(p111.date
                                                                                                         =='0301')|(
        p111.date =='0505')|(p111.date =='0507')|(p111.date =='0522')|(p111.date =='0606')|(p111.date =='0613')|(p111.date
                                                                                                         =='0815'),
          'holiday'] = 1

p111.info()

p111.head()
del p111['date']

p111.to_csv('./pre/data/output2/p1.csv', header=True)



'''
Names1= 0
Names1 = {'First_name': ['Jon','Bill','Maria','Emma']}
df = pd.DataFrame(Names1,columns=['First_name'])
df
df['name_match'] = df['First_name'].apply(lambda x: 'Match' if x == 'Bill' else 'Mis-Match')

df.loc[(df.First_name == 'Bill') | (df.First_name == 'Emma'), 'name_match'] = 'Match'
df.loc[(df.First_name != 'Bill') & (df.First_name != 'Emma'), 'name_match'] = 'Mis-Match'
df

# df.sort_values( by=col_list )
# && and , || or
'''



import numpy as np
import pandas as pd
import datetime

def fianl_func(df):
    df['date'] = df['date'].astype(int)

    a = df[df['date'] >= 101][df['date'] < 901]
    b = df[df['date'] >= 901][df['date'] < 1001]
    c = df[df['date'] > 1001]

    a['tmp'] = 20180
    a['tmp'] = a['tmp'].astype(str)
    a['date'] = a['date'].astype(str)
    a['date'] = a['tmp'] + a['date']

    b['tmp'] = 20170
    b['tmp'] = b['tmp'].astype(str)
    b['date'] = b['date'].astype(str)
    b['date'] = b['tmp'] + b['date']

    c['tmp'] = 2017
    c['tmp'] = c['tmp'].astype(str)
    c['date'] = c['date'].astype(str)
    c['date'] = c['tmp'] + c['date']

    df1=pd.concat([a,b,c],ignore_index=True)
    df1=df1.iloc[:,[0,1,3,4,5] ]
    df1['date']=pd.to_datetime(df1['date'], format='%Y%m%d')


    df1.index = pd.DatetimeIndex(df1.index)
    final= df1.copy()
    final.index = final['date']

    del final['date']
    final = final.reindex(pd.date_range('2017-09-01', '2018-08-31'), fill_value=0)

    final.index = pd.to_datetime(final.index)
    final
    final.info()
    del final['day']

    final['day']= final.index.strftime('%a')
    final['date']=final.index.strftime('%m%d')
    final

    final.loc[final['day']=='Sat', 'holiday'] = 1
    final.loc[final['day']=='Sun', 'holiday'] = 1


    final.loc[(final.date =='1003') | (final.date =='1004') |(final.date =='1005')|(final.date =='1006')|(final.date =='1009')|(
            final.date =='1225')|(final.date =='0101')|(final.date =='0215')|(final.date =='0216')|(final.date =='0217')|(final.date
                                                                                                             =='0301')|(
            final.date =='0505')|(final.date =='0507')|(final.date =='0522')|(final.date =='0606')|(final.date =='0613')|(final.date
                                                                                                             =='0815'),
              'holiday'] = 1


    del final['date']

    return final


# p1~p8까지 진행
p8=pd.read_csv('./pre/data/output2/p8.csv')
del p8['Unnamed: 0']
p8.head()

final_p8=fianl_func(p8)
final_p8['place']=1828

final_p8
final_p8.to_csv('./pre/data/output2/final_p8.csv',header=True)


final_all =pd.concat([final_p1,final_p2,final_p3, final_p4, final_p5, final_p6, final_p7, final_p8] )
final_all
final_all.to_csv('./pre/data/output2/final_all.csv',header=True)
pd.value_counts(final_all['place'])



# < subway and weather preprocessing


import numpy as np
import pandas as pd

# sub
sub=pd.read_csv('./pre/data/output2/raw_sub.csv',header=0, names=['date','s0_h1','num'])
sub.head()
sub.tail()
sub.index=sub['date']
del sub['date']

sub.index = pd.to_datetime(sub.index, format='%Y-%m-%d')

# 승차 in == 0  / 하차 out ==1
sub.loc[sub['s0_h1']==0, 'in'] = 0
sub.loc[sub['s0_h1']==1, 'out'] = 1

in_data=sub[sub['s0_h1']==0]
in_data.info()
in_data = in_data.reindex(pd.date_range('2017-09-01', '2018-08-31'), fill_value=np.NAN)
in_data

out_data=sub[sub['s0_h1']==1]
out_data.info()
out_data = out_data.reindex(pd.date_range('2017-09-01', '2018-08-31'), fill_value=np.NAN)
out_data.sample(10)

# rename 하기
in_data=in_data.rename(columns={'num':'in_num'})
out_data=out_data.rename(columns={'num':'out_num'})
del in_data['s0_h1']
del out_data['s0_h1']

in_data
out_data

del in_data['in']
del in_data['out']
del out_data['in']
del out_data['out']

s=pd.concat([in_data, out_data], axis=1)
s.to_csv('./pre/data/output2/final_subway.csv',header=True)
s



# weather
w=pd.read_csv('./pre/data/output2/raw_w.csv')
w.head()
w.tail()
w.info()
w.isnull().sum()

del w['측정소명']
del w['Column1']


w.rename(columns={w.columns[0]:'date',
                          w.columns[1]:'nitro',
                          w.columns[2]:'ozone',
                          w.columns[3]:'co2',
                          w.columns[4]:'acid_gas',
                          w.columns[5]:'dust',
                          w.columns[6]:'fine_dust',
                          w.columns[7]:'mean_t',
                          w.columns[8]:'min_t',
                          w.columns[9]:'max_t',
                          w.columns[10]:'rainfall',
                          w.columns[11]:'wind',
                          w.columns[12]: 'moist',
                          w.columns[13]: 'cloud',
                          w.columns[14]: 'weather',
                          w.columns[15]: 'sun',
                          }, inplace=True)

w.head()
w.tail()
w.index=w['date']
del w['date']
w.index = pd.to_datetime(w.index, format='%Y-%m-%d')

# w.to_csv('./pre/data/output2/w.csv', header=True)
# train = [ 2017/9/1~2018/8/31]
a=w[w.index >='2017-09-01']

b=a[a.index <'2018-09-01']

# 맑음 0, 흐림 1, 비2, 눈3
b.loc[b['weather']=='맑음', 'weather']=0
b.loc[b['weather']=='흐림', 'weather']=1
b.loc[b['weather']=='비', 'weather']=2
b.loc[b['weather']=='눈', 'weather']=3

b.info()
b.head()
b.tail()

b = b.reindex(pd.date_range('2017-09-01', '2018-08-31'), fill_value=np.NAN)
b.sample(10)
b.isnull().sum()
b.to_csv('./pre/data/output2/ing_w.csv',header=True)

# 다하고 train_w로 바꾸기


w = w.reindex(pd.date_range('2017-09-01', '2018-08-31'))
w.sample(10)
w.isnull().sum()
w.to_csv('./pre/data/output2/final_real_w.csv',header=True)




# p1
p1=pd.read_csv('./pre/data/output2/final_p1.csv')
p1.head()

p1.index=p1['Unnamed: 0']
del p1['Unnamed: 0']
p1.index = pd.to_datetime(p1.index, format='%Y-%m-%d')
p1.head()

# nan 처리하기
s=pd.read_csv('./pre/data/output2/final_subway.csv')
s
s.isnull().sum()
s.index=s['Unnamed: 0']
del s['Unnamed: 0']
s.index = pd.to_datetime(s.index, format='%Y-%m-%d')
s.head()
s.to_csv('./pre/data/output2/final_real_subway.csv')


w=pd.read_csv('./pre/data/output2/final_real_w.csv')
w
w.index=w['Unnamed: 0']
del w['Unnamed: 0']
w.index = pd.to_datetime(w.index, format='%Y-%m-%d')
w = w.reindex(pd.date_range('2017-09-01', '2018-08-31'))
w.isnull().sum()

w.sample(10)

pd.value_counts(w['weather'])


w.to_csv('./pre/data/output2/final_real_weather.csv',header=True)
w.info()
w['weather'].unique()



train1=pd.concat([p1,s,w], axis=1)
train1=train1.drop('place', axis=1)
train1
train1.to_csv('./Final_dataset/p_1911.csv', header=True)


# place=1911, p1 and all


def to_train(df):
    df.index = df['Unnamed: 0']
    del df['Unnamed: 0']
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')

    train = pd.concat([df, s, w], axis=1)
    train = train.drop('place', axis=1)

    return train



# fianl_p1~p8 + weather and subway
p8= pd.read_csv('./pre/data/output2/final_p8.csv')
p8.head()
train1=to_train(p8)
train1
train1.to_csv('./Final_dataset/p_1828.csv', header=True)







# 참조
# 숫자만 뽑아내기
a=pd.read_excel('./visual_data/2017_end.xlsx',encoding='euc-kr', usecols=[1,2,3], names=['name','date','end'])
a.head()


a['name']=a['name'].astype(str)
a['name']=a['name'].str.replace(pat=r'[^0-9]+[ㄱ-ㅣ가-힣]', repl=r'', regex=True)
a['name']
a['name']=a['name'].str.replace(pat=r' [0-9]', repl=r'', regex=True)

a.to_csv('./visual_data/a.csv',header=True)


# date: 월일 , day 요일, hour 시간대, minute 분
data['hour'][i] = data['date'][i].hour


# to find the station
# 구디 역
data[data['startplace']==282]  # 0        2018-10-18 18:34
data[data['startplace']==1854] # 0      2018-10-18 7:29
data[data['startplace']==1856] # 0       2018-09-15 1:25
data[data['startplace']==1986] # 0

data[data['startplace']==1828] # 1100       2017-05-23 11:40
data[data['startplace']==1911] # 7531       2017-05-31 17:50
data[data['startplace']==1924] # 955        2017 09 01
data[data['startplace']==2012] # 1852       2017 08 29
data[data['startplace']==2110] # 1310       2017 08 29
data[data['startplace']==2113] # 3301       2017-07-27 15:35
data[data['startplace']==2115] # 1724       2017 07 03


# 2017/9/1 ~ 2018/9/1 : train 기간

# 2018/9/2 ~   : test

# 강남역
data[data['startplace']==2231]   # 5654  # 6-16~
data[data['startplace']==2233]  # 1602  # 6/16~
data[data['startplace']==2407]   # 0
data[data['startplace']==2409] #  0
data[data['startplace']==2505] #   0     2017-05-23 11:40


# 역삼
data[data['startplace']== 2349]  # 1958 8/28
data[data['startplace']== 2357]  # 641 9/23
data[data['startplace']== 2410]  # 0

# 신림
data[data['startplace']== 2102]  # 16357  6/9
data[data['startplace']== 2106]  # 2288  7/8
data[data['startplace']== 2109]  # 2280  9/16
data[data['startplace']== 2127]  # 804   7/20~
data[data['startplace']== 2135]  # 4956  6/18
data[data['startplace']== 2136]  # 2388  9/15
data[data['startplace']== 2140]  # 3887  6/9    7 개
data[data['startplace']== 2164]  # 0
data[data['startplace']== 2173]  # 0
data[data['startplace']== 2174]  # 0
data[data['startplace']== 2175]  # 0
data[data['startplace']== 2180]  # 0   6개


# 삼성
data[data['startplace']== 2316 ]  # 3133 6/30
data[data['startplace']==2322 ]   # 2136 7/24
data[data['startplace']== 2326]   # 1343 8/3
data[data['startplace']== 2348]   # 2374 7/27
data[data['startplace']== 2355]   # 3332 8/30

# 신도림
data[data['startplace']== 241]  #  3435 1/1
data[data['startplace']== 255 ]   # 2679 1/1
data[data['startplace']== 256]   # 3379 1/1
data[data['startplace']== 278 ]   # 0
data[data['startplace']==  1960]   #12/6  몇십개
data[data['startplace']== 1920 ]   #  3201 6/9
data[data['startplace']== 1961 ]   #  12 몇십개  77


