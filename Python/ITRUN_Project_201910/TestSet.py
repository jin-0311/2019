# 데이터 출처 : https://data.seoul.go.kr/dataList/datasetView.do?infId=OA-15182&srvType=F&serviceKind=1&currentPageNo=1

# Test DataSet: 2018년9월1일~ 2018년12월 31일
import pandas as pd
import numpy as np


# weather
w=pd.read_csv('./test_pre/w_raw_data.csv', encoding='euc-kr')
w
del w['Column1']
del w['측정소명']
w.index=w['날짜']
del w['날짜']


w.index=pd.to_datetime(w.index, format='%Y-%m-%d')
w
w.reindex(pd.date_range('2017-01-01', '2018-12-31'), fill_value=np.NAN)



test_w=w.loc[w.index >='2018-09-01']
test_w

test_w.rename(columns={test_w.columns[0]:'nitro',
                          test_w.columns[1]:'ozone',
                          test_w.columns[2]:'co2',
                          test_w.columns[3]:'acid_gas',
                          test_w.columns[4]:'dust',
                          test_w.columns[5]:'fine_dust',
                          test_w.columns[6]:'mean_t',
                          test_w.columns[7]:'min_t',
                          test_w.columns[8]:'max_t',
                          test_w.columns[9]:'rainfall',
                          test_w.columns[10]:'wind',
                          test_w.columns[11]: 'moist',
                          test_w.columns[12]: 'cloud',
                          test_w.columns[13]: 'weather',
                          test_w.columns[14]: 'sun',
                          }, inplace=True)

test_w

test_w.loc[test_w['weather']=='맑음', 'weather']=0
test_w.loc[test_w['weather']=='흐림', 'weather']=1
test_w.loc[test_w['weather']=='비', 'weather']=2
test_w.loc[test_w['weather']=='눈', 'weather']=3

test_w.isnull().sum()


test_w.to_csv('./test_pre/weather_final.csv',header=True)


# sub 됨
sub=pd.read_csv('../bike//test_pre/gudi2018_9_12.csv',header=0, names=['date','s0_h1','num'])
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
in_data = in_data.reindex(pd.date_range('2018-09-01', '2018-12-31'), fill_value=np.NAN)
in_data

out_data=sub[sub['s0_h1']==1]
out_data.info()
out_data = out_data.reindex(pd.date_range('2018-09-01', '2018-12-31'), fill_value=np.NAN)
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
s.to_csv('./test_pre/subway_final.csv',header=True)
s

s.isnull().sum()



# 8-9시의 구로디지털단지역 근처 6개의 데이터
# 2018년9월1일~ 2018년12월 31일


import pandas as pd
import numpy as np
import xlrd

#  usecols=[1,2], names=['startdate', 'place'],

d1=pd.read_excel('../bike/test_pre/data/1.xlsx',
                 usecols=[1,2,9,10],
                 names=['date','place', 'using_time','dist'],
                 encoding='euc-kr')
d1.head()
d1


d2=pd.read_excel('../bike/test_pre/data/2.xlsx',
                 usecols=[1,2,9,10],
                 names=['date','place', 'using_time','dist'],
                 encoding='euc-kr')
d2.head()


d3=pd.read_excel('../bike/test_pre/data/3.xlsx',
                 usecols=[1,2,9,10],
                 names=['date','place', 'using_time','dist'],
                 encoding='euc-kr')
d3.head()

d4=pd.read_excel('../bike/test_pre/data/4.xlsx',
                 usecols=[1,2,9,10],
                 names=['date','place', 'using_time','dist'],
                 encoding='euc-kr')
d4.head()

d5=pd.read_excel('../bike/test_pre/data/5.xlsx',
                 usecols=[1,2,9,10],
                 names=['date','place', 'using_time','dist'],
                 encoding='euc-kr')
d5.head()


d6=pd.read_csv('../bike/test_pre/data/6.csv',
               usecols=[1, 2, 9, 10],
               names=['date', 'place', 'using_time', 'dist'],
               encoding='euc-kr')
d6.head()


data=pd.concat([d1,d2,d3,d4,d5,d6], ignore_index=True)
data.info()
data.tail()
data['place']=data['place'].astype(str)
data['place']=data['place'].str.replace(pat=r'[^A-Za-z0-9]', repl=r'', regex=True)


gudi_t=data[data['place'].isin(['282','1828','1854','1856','1911','1924','1955','1986','2012','2110','2113','2115'])]
gudi_t
pd.value_counts(gudi_t['place'])
gudi_t.to_csv('../bike/data/output/gudi_test.csv',header=True)

gudi_t=gudi_t.reset_index(drop=True)

gudi_t.info()
gudi_t['date']=pd.to_datetime(gudi_t['date'], format='%Y-%m-%d')
gudi_t.index=gudi_t['date']
#del gudi_t['date'] 

gudi_t['finddate'] =0
gudi_t['day']=0
gudi_t['hour']=0

gudi_t['finddate'] = gudi_t.index.strftime('%m%d')
gudi_t['day']  = gudi_t.index.strftime('%a')
gudi_t['hour']  = gudi_t.index.hour


count_data=gudi_t.groupby(['finddate','day','hour','place'] )['place'].count().reset_index(name='count')
count_data
# count_data.to_csv('./gudi_t_count2.csv',header=True)

eight=count_data.loc[count_data['hour']==8,:]
eight=eight.reset_index(drop=True)
eight

# eight.to_csv('./gudi_t_eight2.csv',header=True)

eight['place'].unique()



p1=eight.loc[eight['place']=='1911',:]
p1   

p2=eight.loc[eight['place']=='1924',:]
p2 

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

p1.to_csv('../bike/test_pre/p1.csv',header=True)
p2.to_csv('../bike/test_pre/p2.csv',header=True)
p3.to_csv('../bike/test_pre/p3.csv',header=True)
p4.to_csv('../bike/test_pre/p4.csv',header=True)
p5.to_csv('../bike/test_pre/p5.csv',header=True)
p6.to_csv('../bike/test_pre/p6.csv',header=True)
p7.to_csv('../bike/test_pre/p7.csv',header=True)
p8.to_csv('../bike/test_pre/p8.csv',header=True)

def fianl_func(df):
    df['finddate'] = df['finddate'].astype(int)

    df['tmp'] = 2018
    df['tmp'] = df['tmp'].astype(str)
    df['finddate'] = df['finddate'].astype(str)
    df['date'] = df['tmp'] + df['finddate']

    df['date']=pd.to_datetime(df['date'], format='%Y%m%d')

    final= df.copy()
    final.index = final['date']

    del final['date']
    final = final.reindex(pd.date_range('2018-09-01', '2018-12-31'), fill_value=0)

    final.index = pd.to_datetime(final.index)
    final
    final.info()
    del final['day']

    final['day']= final.index.strftime('%a')
    final['date']=final.index.strftime('%m%d')
    final

    final['holiday']=0
    final.loc[final['day']=='Sat', 'holiday'] = 1
    final.loc[final['day']=='Sun', 'holiday'] = 1


    final.loc[(final.date =='0924') | (final.date =='0925') |(final.date =='0926')|(final.date =='1003')|(
            final.date=='1009')|(final.date =='1225'), 'holiday'] = 1


    del final['finddate']
    del final['tmp']
    del final['hour']
    del final['date']
    return final



# p1~p8까지 진행
p8.head()

final_p8=fianl_func(p8)
final_p8

final_p8['place']=1828
final_p8.to_csv('../bike/test_pre/final_p8.csv',header=True)




final_all_t =pd.concat([final_p1,final_p2,final_p3, final_p4, final_p5, final_p6,final_p7, final_p8] )
final_all_t
# final_all_t.to_csv('../bike/test_pre/final_all_t.csv',header=True)
pd.value_counts(final_all_t['place'])



import pandas as pd
import numpy as np

# final_p1 최종 test set


def to_train(df):
    del df['place']
    train = pd.concat([df, s, test_w], axis=1)

    return train

p8.head()
final_p8.head()
train1=to_train(final_p8)
train1
train1.to_csv('./Final_dataset/test/p_1828.csv', header=True)





