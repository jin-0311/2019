# 출처: 파이썬으로 데이터 주무르기
# http://www.bjpublic.co.kr/


#%% 파이썬 주무르기 - 예제
#%% 주피터 기본 실행 및 설정 

'''
#  web page for reference  
* pandas.pydata.org  판다스
 

# 기본 키 
* 실행 -> shift enter
* 삭제 -> d twice
* m -> markdown 으로 변경
* y -> code cell 으로 변경 
* command mode -> esc & enter 
* select multiple cells -> hold shift and click or press up/down -> shift+m(merge)
    * 셀 나누기 : ctrl+shift+ (-) 커서 위치로 나뉨

* 긴 한줄 코드를 두줄로 쓰려면 \ 쓰고 엔터치고 넘어가면 연결됨
* 함수 자동완성 .누르고 tab
    * shift + tab : python tootip
    * shift + tab(2번) :detailed python tooltip
    
###  주피터 노트북 절대 경로 설정
* cmd) jupyter notebook --generate-config
* 텍스트 편집기에서 notebook_dir 찾아서 주석 삭제, c.NotebookApp.notebook_dir = ' ' 에 지정하고 저장 

###  상대경로
* / -루트    ./ -현재위치     C:/Users/serah/ds/ -상위폴더    C:/Users/serah/ds/C:/Users/serah/ds/ -두단계 상위폴더로 이동할때 


## pandas : 시리즈와 데이터프레임의 분석을 위한 python library
* pandas(pd), numpy(np) 둘다 import  
* 시리즈는 데이터 프레임의 칼럼
    * index, values (리스트 성분의 개수 = index의 개수) / 딕셔너리와 비슷 
* 데이터프레임은 n개의 시리즈로 구성됨
    * python의 딕셔너리 형태{"key":[value]}로 데이터 정의 하고 dataframe으로 정의하기
    * values이면 [[]] 
* NaN : Not a Number == NA
* 
* 판다스 자료형 <- 파이썬 자료형 비교 
    * object <- string
    * int64 <- int
    * float64 <- float
    * datetime64 <- datetime  
'''

# 현재 경로 확인
import os
print(os.getcwd())  #>>> C:/Users/a


#%% Ch1) 서울시 구별 CCTV 현황 분석하기

# 서울시 각 구별 CCTV수를 파악하고, 인구대비 CCTV 비율을 파악해서 순위 비교
# 인구대비 CCTV의 평균치를 확인하고 그로부터 CCTV가 과하게 부족한 구를 확인

import os
print(os.getcwd())

# 1. 데이터 불러오기 
import pandas as pd
cctv = pd.read_csv('C:/Users/serah/ds/data/01. CCTV_in_Seoul.csv', encoding = 'utf-8')
cctv.head()

cctv.columns
cctv.columns[0]

# 기관명 -> 구별 rename 
cctv.rename(columns = {cctv.columns[0]: '구별'}, inplace = True)
cctv.head()

# 2. 엑셀파일 읽기 - 서울시 인구현황
pop_s = pd.read_excel('C:/Users/serah/ds/data/01. population_in_Seoul.xls', encoding = 'utf-8')
pop_s.head()


pop_s = pd.read_excel('C:/Users/serah/ds/data/01. population_in_Seoul.xls',
                       header =2,
                       encoding = 'utf-8')
pop_s.head()

pop_s1 = pop_s.loc[:, ['자치구','계', '계.1', '계.2','65세이상고령자']]
pop_s1.head()

pop_s2 = pd.read_excel('C:/Users/serah/ds/data/01. population_in_Seoul.xls',
                       parse_cols = 'B,D,G,J,N',   #엑셀파일에서 원하는 컬럼만 뽑기 
                       header =2,        # header가 여러개, 2번째꺼 선택 
                       encoding = 'utf-8')
pop_s2.head()

pop_s1.rename(columns = {pop_s1.columns[0] : '구별',
                         pop_s1.columns[1] : '인구수',
                         pop_s1.columns[2] : '한국인',
                         pop_s1.columns[3] : '외국인',
                         pop_s1.columns[4] : '고령자' }, 
                         inplace = True )
pop_s1.head() 

# 3. cctv데이터 파악하기
cctv.head()

# sort_values (by = '', ascending = T/F)
print(cctv.sort_values(by = '소계', ascending = True).head(5))
print(cctv.sort_values(by = '소계', ascending = False).head(5))

#cctv 개수가 가장 적은 도봉/마포/송파/중랑/중구 (강3중 송파 범죄율은 높은편 )

cctv['최근증가율'] = (cctv['2016년'] + cctv['2015년'] + cctv['2014년']) / cctv['2013년도 이전']  * 100
cctv.sort_values(by='최근증가율', ascending=False).head(5)

#3년간 증가율이 높은 곳은 종로/도봉/마포/노원/강동 

# 4. 서울시 인구 데이터 파악하기
pop_s1.head()

# 행삭제 drop, 열 삭제 del
pop_s1.drop([0], inplace= True)  # index[0]인 합계 삭제! 
pop_s1.head()

# .unique : 한번 이상 나타난 데이터 확인 (데이터의 유일한 값 찾기 nan 포함!) 
pop_s1['구별'].unique()   

# pop_s1.duplicated()

pop_s1[pop_s1['구별'].isnull()]  # nan 있는지 확인 -> 26행이 nan! 

print(pop_s1.tail())
# nan행 제거하기
pop_s1.drop([26], inplace= True)
print(pop_s1.head())

pop_s1.sort_values(by = '인구수', ascending = False).head()

pop_s1['외국인비율'] = pop_s1['외국인'] / pop_s1['인구수'] * 100
pop_s1['고령자비율'] = pop_s1['고령자'] / pop_s1['인구수'] * 100
pop_s1.head()

pop_s1.sort_values(by = '외국인비율', ascending = False).head(5)

print(pop_s1.sort_values(by = '고령자', ascending = False).head(5))
print(pop_s1.sort_values(by = '고령자비율', ascending = False).head(5))

# 5. cctv, pop_s1 합치고 분석하기 
data_all = pd.merge(cctv, pop_s1, on = '구별')
data_all.head()

# 행 삭제 drop, 열 삭제 del
del data_all['2013년도 이전']
del data_all['2014년']
del data_all['2015년']
del data_all['2016년']
data_all.head()

# 구별을 index값으로 지정  .set_index('인덱스가 될 칼럼명', inplace = True)
data_all.set_index('구별', inplace = True)
data_all.head()

# 상관관계 확인  Correlation analysis 
# abs(0.1) 이하 => 거의 무시
# abs(0.3) 이하 => 약한 상관관계
# abs(0.7) 이하 => 뚜렷한 상관관계 

# np.correcoef(df['col1'],df['col2'])

np.corrcoef(data_all['고령자비율'],data_all['소계'])

np.corrcoef(data_all['외국인비율'],data_all['소계'])

np.corrcoef(data_all['인구수'],data_all['소계'])

data_all.sort_values(by = '소계', ascending = False).head()

data_all.sort_values(by = '인구수', ascending = False).head()

# 6. cctv, pop_s1 그래프로 분석하기
# 그래프 한글작업 
import platform
from matplotlib import font_manager, rc      # 한글작업때문에  꼭 필요! 
plt.rcParams['axes.unicode_minus'] = False   # 마이너스 코드 때문에

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~') 

data_all.head()

# plt그래프 그리기 ->  판다스데이터.plot(옵션)  
plt.figure() # 그래프 틀 
data_all['소계'].plot(kind='barh', grid=True, figsize=(10,10))
plt.show()

# kind = 'barh' 수평바

# barh plot sort -> sort_values()가지고 오름차순 정렬, 큰값이 맨 위로 
data_all['소계'].sort_values().plot(kind='barh', 
                                     grid=True, figsize=(10,10))
plt.show()

# 인구수 대비 cctv비율 분석 
data_all['CCTV비율'] = data_all['소계'] / data_all['인구수'] * 100

data_all['CCTV비율'].sort_values().plot(kind='barh', 
                                         grid=True, figsize=(10,10))
plt.show()

# 산점도 그리기 
plt.figure(figsize=(6,6))
plt.scatter(data_all['인구수'], data_all['소계'], s=50)
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()

# np의 polyfit :데이터를 대표하는 직선하나 그리기 (cctv와 인구수가 양의 상관관계 )
# x축은 np의 linspace로,  y축은 poly1d로 만들기 
fp1 = np.polyfit(data_all['인구수'], data_all['소계'], 1)
fp1

f1 = np.poly1d(fp1)   # y축 
fx = np.linspace(100000, 700000, 100)  # x축 : 점이 찍힐 수 있는 범위를 정해줌(시작,끝,간격)

# 그래프 그리기  
plt.figure(figsize=(10,10))
plt.scatter(data_all['인구수'], data_all['소계'], s=50)
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')  # 직선 그래프 그려주기 
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()

# 10. 조금 더 설득력 있는 자료 만들기 
# 전체 데이터의 대표값인 하나의 직선 ->  인구수:300000 = cctv:1100 정도이고 
# 직선에서 멀어질수록 다른색 표현하기 위해 오차를 구하기 
# 단순하게 인구수 - 소계 로 오차를 지정 

fp1 = np.polyfit(data_all['인구수'], data_all['소계'], 1)
f1 = np.poly1d(fp1)
fx = np.linspace(100000, 700000, 100)

# 오차를 컬럼으로 만들어서 표에 넣어주기 
data_all['오차'] = np.abs(data_all['소계'] - f1(data_all['인구수']))

df_sort = data_all.sort_values(by='오차', ascending=False)
df_sort.head()

# 각 점에 이름과 색상을 부여해주기 
plt.figure(figsize=(14,10))
plt.scatter(data_all['인구수'], data_all['소계'], 
            c=data_all['오차'], s=50)
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')

for n in range(10):
    plt.text(df_sort['인구수'][n]*1.02,  # x축 약간 오른쪽에
             df_sort['소계'][n]*0.98,    # y축 왼쪽에 
             df_sort.index[n], fontsize=15)
    
plt.xlabel('인구수')
plt.ylabel('인구당비율')
plt.colorbar()
plt.grid()
plt.show()

# 평균값에서 먼것 기준 10개만 뿌림! 

# 서울시에서 다른 구와 비교했을 때, 강남구, 양천구, 서초구, 은평구는 CCTV가 많지만,
# 송파구, 강서구, 도봉구, 마포구는 다른 구에 비해 CCTV 비율이 낮다



#%% Ch2) 서울 범죄 데이터 분석 
## 강남 3구는 안전한가?

# 0. 데이터 불러오기 
import numpy as np
import pandas as pd

crime = pd.read_csv('C:/Users/serah/ds/data/02. crime_in_Seoul.csv', thousands = ',', encoding = 'euc-kr' )
crime.head()

# 1. google maps 
# prompt )pip install googlemaps
import googlemaps
gmaps = googlemaps.Client(key = ' ')

gmaps.geocode('서울중부경찰서', language= 'ko')
# lng위도, lat경도, fomatted_address주소 등이 나옴 

# 2. 관서명 변경 하기 
# 관서명이 '중부서' 이런식으로 되어있어서 '서울+이름-서+경찰서'로 바꿔줌  to get geocode
station_name = []
for i in crime['관서명'] :
    station_name.append('서울' + str(i[:-1]) + '경찰서')
station_name

station_address = []    # list 
station_lat = []
station_lng = []

# 'formatted_address'를 바탕으로 주소 얻어옴 
for i in station_name :
    tmp = gmaps.geocode(i, language = 'ko')
    station_address.append(tmp[0].get('formatted_address'))
    
    tmp_loc = tmp[0].get('geometry')
    
    station_lat.append(tmp_loc['location']['lat'])
    station_lng.append(tmp_loc['location']['lng'])
    print(i +'-->' + tmp[0].get('formatted_address'))


print(tmp[0].get('geometry'))
print(tmp[0].get('formatted_address'))

print(station_address)
print(station_lat)
print(station_lng)

# 3. 주소를 바탕으로 구별 컬럼 만들어서 나누기 
gu_name = []
for i in station_address:
    tmp = i.split()  # 공백을 기준으로 나누고(split())
    tmp_gu = [ j for j in tmp if j[-1] == '구'][0] # tmp중 마지막 글자가 구 이면 tmp_gu에 넣기 
    gu_name.append(tmp_gu)  #tmp_gu를 gu_name리스트에 넣기 

print(tmp)
print(tmp_gu)  # 구 목록
print(gu_name)
crime['구별'] = gu_name
crime.head()

crime[crime['관서명']=='금천서']

# 금천 경찰서는 관악구에 있기 때문에 따로 금천구로 예외처리! 
crime.loc[crime['관서명']=='금천서', ['구별']] = '금천구'
crime[crime['관서명'] == '금천서']

crime.to_csv('C:/Users/serah/ds/data/02. crime_include_guname.csv', sep =',', encoding = 'utf-8')

crime.head()

# 4. 범죄 데이터 구별로 정리하기
crime_raw = pd.read_csv('C:/Users/serah/ds/data/02. crime_include_guname.csv', encoding = 'utf-8')
crime_raw.head()

# index를 정리하고 
crime_raw = pd.read_csv('C:/Users/serah/ds/data/02. crime_include_guname.csv', 
                        encoding = 'utf-8', index_col = 0)  
crime_raw.head()

# pivot 으로 '구별'로 바꾸고, 데이터 정리 
# np.sum => 인덱스 별로 중복값은 더해주기 
crime= pd.pivot_table(crime_raw, index = '구별', aggfunc= np.sum) 
crime.head()

crime['강간검거율'] = crime['강간 검거']/crime['강간 발생'] *  100
crime['강도검거율'] = crime['강도 검거']/crime['강도 발생'] *  100
crime['살인검거율'] = crime['살인 검거']/crime['살인 발생'] *  100
crime['절도검거율'] = crime['절도 검거']/crime['절도 발생'] *  100
crime['폭력검거율'] = crime['폭력 검거']/crime['폭력 발생'] *  100

crime.head()


# 검거는 검거율로 대체할 수 있으니 삭제 - 행 삭제는 drop, 열 삭제는 del
del crime['강간 검거']
del crime['강도 검거']
del crime['살인 검거']
del crime['절도 검거']
del crime['폭력 검거']

crime

# 100퍼센트 넘는 것은 100으로 바꿔주기 (전년도 건수에 대한 검거 포함하기 때문에 )
con_list = ['강간검거율', '강도검거율', '살인검거율','절도검거율', '폭력검거율']

for column in con_list :
    crime.loc[crime[column] > 100, column] = 100

crime

crime.rename(columns = {'강간 발생' : '강간',
                        '강도 발생' : '강도',
                        '살인 발생' : '살인',
                        '절도 발생' : '절도',
                        '폭력 발생' : '폭력' },
                        inplace = True)
crime.head()

# 5. 정규화(스케일 조정)  - 건수가 다르기 때문에 컬럼별 정규화 (+시각화할때 잘보이도록 )
# sklearn에는 4가지 스케일조정방법
# standard(평균0,분산1), robust(중간값,사분위값사용)
# minmax(모든 데이터가 0과1사이에 위치하도록), normalizer

# 그중 sklearn을 이용해 minmax scaler사용 (최대값, 최소값 이용하여 정규화 )


from sklearn import preprocessing # 전처리 도구 preprocessing 

col = ['강간','강도','살인','절도','폭력']

x = crime[col].values
min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x.astype(float))
crime_norm = pd.DataFrame(x_scaled, columns = col, index = crime.index)

col2 = ['강간검거율','강도검거율','살인검거율','절도검거율','폭력검거율']
crime_norm[col2] = crime[col2]
crime_norm.head()





# 6. cctv 데이터와 함께
result_cctv = pd.read_csv('C:/Users/serah/ds/data/01. CCTV_result.csv', 
                          encoding = 'utf-8', index_col = '구별')
result_cctv.head()

crime_norm[['인구수','cctv']] = result_cctv[['인구수','소계']]
crime_norm.head()

# '범죄' 컬럼을 만들어서 전체 합계로 통일 np.sum
col= ['강간','강도','살인','절도','폭력']
crime_norm['범죄'] = np.sum(crime_norm[col], axis = 1) # 오른쪽에 붙임 
crime_norm.head()

# '검거' 컬럼 만들어서 검거율 합계 
col = ['강간검거율','강도검거율','살인검거율','절도검거율','폭력검거율']
crime_norm['검거'] = np.sum(crime_norm[col], axis = 1)
crime_norm.head()

# 7. seaborn 활용 범죄데이터 시각화하기 
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# 한글문제 해결 
import platform
path = "c:/Windows/Fonts/malgun.ttf"
from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~') 

crime_norm.head()

sns.pairplot(crime_norm, 
             vars = ['강도','살인','폭력'], 
             kind = 'reg', 
             size = 3)
plt.title('강도/살인/폭력의 상관관계')  #>>>모두 양의 관계
plt.show()  



sns.pairplot(crime_norm, 
             x_vars = ['인구수','cctv'], 
             y_vars = ['살인', '강도'],
             kind = 'reg', 
             size = 3)
plt.title('인구수,cctv /살인,강도의 상관관계') #>>> cctv개수와 살인과 강도는 +
plt.show()


sns.pairplot(crime_norm, 
             x_vars = ['인구수','cctv'], 
             y_vars = ['살인검거율', '폭력검거율'],
             kind = 'reg', 
             size = 3)
plt.title('인구수,cctv/ 살인,폭력검거율') #>>> -의 관계
plt.show()



sns.pairplot(crime_norm, 
             x_vars = ['인구수','cctv'], 
             y_vars = ['절도검거율', '강도검거율'],
             kind = 'reg', 
             size = 3)
plt.title('인구수,cctv/절도,강도 검거율 ')
plt.show()

# 검거를 100으로 한정하고 정렬하고 히트맵 그리기 
tmp_max = crime_norm['검거'].max()
crime_norm['검거'] = crime_norm['검거'] / tmp_max *100
crime_norm_sort = crime_norm.sort_values (by = '검거', ascending = False)
crime_norm_sort.head()

# heatmap 작성시 cmap ='원하는 색' 옵션 작성 필수
target_col = ['강간검거율', '강도검거율','살인검거율', '절도검거율', '폭력검거율']

crime_norm_sort = crime_norm.sort_values (by = '검거', ascending = False)

plt.figure(figsize = (10,10))
sns.heatmap(crime_norm_sort[target_col], annot = True, fmt = 'f', linewidths = .5, cmap = 'RdPu')

plt.title('범죄 검거 비율(정규화된 검거의 합으로 정렬)')
plt.show()

target_col = ['강간','강도','살인','절도','폭력','범죄']
crime_norm['범죄'] = crime_norm['범죄'] / 5
crime_norm_sort = crime_norm.sort_values (by = '범죄', ascending = False)

plt.figure(figsize = (10,10))
sns.heatmap(crime_norm_sort[target_col], annot = True, fmt = 'f', linewidths = .5, cmap = 'Blues')

plt.title('범죄비율(정규화된 발생 건수로 정렬)')
plt.show()

crime_norm.to_csv('C:/Users/serah/ds/data/02.crime_final.csv', sep=',', encoding = 'utf-8')

# 8. folium을 활ㅇ요한 범죄율 지도 시각화 
# geo_str대신 geo_data사용하기 
# json은 서울시 구별 경계선을 위해 
import folium
import json
geo_path = 'C:/Users/serah/ds/data/02. skorea_municipalities_geo_simple.json'
geo_data = json.load(open(geo_path, encoding='utf-8'))

map = folium.Map(location=[37.5502, 126.982], zoom_start=11, 
                 tiles='Stamen Toner')

map.choropleth(geo_data = geo_data,
               data = crime_norm['살인'],
               columns = [crime_norm.index, crime_norm['살인']],
               fill_color = 'PuRd', #PuRd, YlGnBu
               key_on = 'feature.id')
map

map = folium.Map(location=[37.5502, 126.982], zoom_start=11, 
                 tiles='Stamen Toner')

map.choropleth(geo_data = geo_data,
               data = crime_norm['강간'],
               columns = [crime_norm.index, crime_norm['강간']],
               fill_color = 'PuRd', #PuRd, YlGnBu
               key_on = 'feature.id')
map

map = folium.Map(location=[37.5502, 126.982], zoom_start=11, 
                 tiles='Stamen Toner')

map.choropleth(geo_data = geo_data,
               data = crime_norm['범죄'],
               columns = [crime_norm.index, crime_norm['범죄']],
               fill_color = 'PuRd', #PuRd, YlGnBu
               key_on = 'feature.id')
map

# 인구수를 고려하여 -> 인구대비 범죄 발생 비율을 알아보기 

tmp_criminal = crime_norm['살인'] /  crime_norm['인구수'] * 1000000

map = folium.Map(location=[37.5502, 126.982], zoom_start=11, 
                 tiles='Stamen Toner')

map.choropleth(geo_data = geo_data,
               data = tmp_criminal,
               columns = [crime.index, tmp_criminal],
               fill_color = 'PuRd', #PuRd, YlGnBu
               key_on = 'feature.id')
map


tmp_criminal = crime_norm['범죄'] /  crime_norm['인구수'] * 1000000

map = folium.Map(location=[37.5502, 126.982], zoom_start=11, 
                 tiles='Stamen Toner')

map.choropleth(geo_data = geo_data,
               data = tmp_criminal,
               columns = [crime.index, tmp_criminal],
               fill_color = 'PuRd', #PuRd, YlGnBu
               key_on = 'feature.id')
map


 
map = folium.Map(location=[37.5502, 126.982], zoom_start=11, 
                 tiles='Stamen Toner')

map.choropleth(geo_data = geo_data,
               data = crime_norm['검거'],
               columns = [crime_norm.index, crime_norm['검거']],
               fill_color = 'YlGnBu', #PuRd, YlGnBu
               key_on = 'feature.id')
map

# 9. 경찰서별 검거 현황과 구별 범죄발생 현황을 동시에 시각화

# 검거를 따로 모으고, 경찰서별 위도경도 정보를 얻어와서 crime_raw에 저장 
crime_raw['lat'] = station_lat
crime_raw['lng'] = station_lng

col = ['살인 검거','강도 검거', '강간 검거','절도 검거','폭력 검거']
tmp = crime_raw[col] /crime_raw[col].max()
crime_raw['검거'] = np.sum(tmp, axis = 1)
crime_raw.head()

map = folium.Map(location=[37.5502, 126.982], zoom_start=11)

for n in crime_raw.index:
    folium.Marker([crime_raw['lat'][n], 
                   crime_raw['lng'][n]]).add_to(map)
    
map

# 검거에 10을 곱해서 원의 넓이를 정하고, 검거율은 원의 넓이로 표현! 
# -> 각 위치에서 원이 클수록 검거율이 높은 것! 
# CircleMarker쓸때 fill = True 꼭 써주기 
map = folium.Map(location=[37.5502, 126.982], zoom_start=11)

for n in crime_raw.index:
    folium.CircleMarker([crime_raw['lat'][n], crime_raw['lng'][n]], 
                        radius = crime_raw['검거'][n]*10, 
                        color='#3186cc', fill_color='#3186cc', fill=True).add_to(map)
    
map

map = folium.Map(location=[37.5502, 126.982], zoom_start=11)

map.choropleth(geo_data = geo_data,
               data = crime_norm['범죄'],
               columns = [crime_norm.index, crime_norm['범죄']],
               fill_color = 'PuRd', #PuRd, YlGnBu
               key_on = 'feature.id')

for n in crime_raw.index:
    folium.CircleMarker([crime_raw['lat'][n], crime_raw['lng'][n]], 
                        radius = crime_raw['검거'][n]*10, 
                        color='#3186cc', fill_color='#3186cc', fill=True).add_to(map)
    
map


#%% Ch3) 시카고 샌드위치 맛집 소개 사이트에 접근하기
* goo.gl/wAtv1s

# 0. 준비 
from bs4 import BeautifulSoup   # import library
from urllib.request import urlopen  # url에서 바로 접근  

# 주소가 길때 나눠서 표현 
url_base = 'http://www.chicagomag.com'
url_sub = '/Chicago-Magazine/November-2012/Best-Sandwiches-Chicago/'
url = url_base + url_sub

html = urlopen(url)
soup = BeautifulSoup(html, "html.parser")

soup

# 1. 필요한 태그 찾기 

# 첫번째 샌드위치 집 
'''
<div class="sammy" style="position: relative;">
<div class="sammyRank">1</div>

<div class="sammyListing"><a href="/Chicago-Magazine/November-2012/Best-Sandwiches-in-Chicago-Old-Oak-Tap-BLT/"><b>BLT</b><br>
Old Oak Tap<br>
<em>Read more</em> </a></div>
</div>
'''
# -> soup.find_all('div, 'sammy') 로 찾으면 50 개 전체가 나옴! 

print(soup.find_all('div', 'sammy'))

len(soup.find_all('div', 'sammy'))  # 개수 

print(soup.find_all('div', 'sammy')[0])  # 첫번째 샌드위치 집 



# 2. 접근한 웹 페이지에서 원하는 데이터 추출하고 정리하기
tmp_one = soup.find_all('div', 'sammy')[0]
type(tmp_one) 
#>>> bs4.element.Tag -> 이러한 형태는 find(_all)이 가능 

tmp_one.find(class_='sammyRank') # 한번더 찾은결과 

tmp_one.find(class_='sammyRank').get_text()  #텍스트만 얻고 -> 이게 rank

tmp_one.find(class_='sammyListing')  #메뉴이름과 가게이름 얻기 

tmp_one.find(class_='sammyListing').get_text()  # 메뉴이름과 가게이름 텍스트 얻고 

# a태그의 href : 연결되는 url정보  -> 상대경로 
tmp_one.find('a')['href']  # 가게 웹페이지 주소를 찾기위해 

tmp_string = tmp_one.find(class_='sammyListing').get_text()
print(tmp_string)

import re # 정규식 regular expression / split 사용하기 위해 
# split : 특정 패턴과 일치하면 분리 
# re.split(pattern, string)string을 pattern 기준으로 나눔 
# \n or \r\n   : 둘다 공백을 의미하고, 둘중 하나니까 결과가 2개 나옴 
tmp_string = tmp_one.find(class_='sammyListing').get_text()
re.split(('\n|\r\n'), tmp_string)   
print(re.split(('\n|\r\n'), tmp_string)[0])  # -> [0] : 메뉴이름
print(re.split(('\n|\r\n'), tmp_string)[1])  # ->  [1]: 가게이름 

# 절대 경로 url은 그대로, 상대 경로를 절대경로로 변경  (href는 상대경로 )
from urllib.parse import urljoin

rank = []   
main_menu = []  
cafe_name = []
url_add = []

list_soup = soup.find_all('div', 'sammy')

for item in list_soup:
    rank.append(item.find(class_='sammyRank').get_text())
    
    tmp_string = item.find(class_='sammyListing').get_text()

    main_menu.append(re.split(('\n|\r\n'), tmp_string)[0])
    cafe_name.append(re.split(('\n|\r\n'), tmp_string)[1])
    
    url_add.append(urljoin(url_base, item.find('a')['href']))
    

# pip install tqdm  -> 상태 진행바를 쉽게 만들어주는 모듈 
from tqdm import tqdm_notebook
import time

from urllib.parse import urljoin 

rank = []
main_menu = []
cafe_name = []
url_add = []

list_soup = soup.find_all('div', 'sammy')
bar_total = tqdm_notebook(list_soup)

for item in bar_total:
    rank.append(item.find(class_='sammyRank').get_text())
    
    tmp_string = item.find(class_='sammyListing').get_text()

    main_menu.append(re.split(('\n|\r\n'), tmp_string)[0])
    cafe_name.append(re.split(('\n|\r\n'), tmp_string)[1])
    
    url_add.append(urljoin(url_base, item.find('a')['href']))
    
    time.sleep(0.05)  # 0.05간격으로 처리하기 

rank[:5]

main_menu[:5]

cafe_name[:5]

url_add[:5]

len(rank), len(main_menu), len(cafe_name), len(url_add)


# 3. 리스트 형태를 pd로 데이터 프레임으로 변경 
import pandas as pd

data = {'Rank':rank, 'Menu':main_menu, 'Cafe':cafe_name, 'URL':url_add}
df = pd.DataFrame(data)
df.head()

df = pd.DataFrame(data, columns=['Rank','Cafe','Menu','URL'])  #컬럼 정리 
df.head(5)

df.to_csv('C:/Users/serah/ds/data/03. best_sandwiches_list_chicago.csv', sep=',', 
          encoding='UTF-8')



# 4. 다수의 웹 페이지에 자동으로 접근해서 원하는 정보 가져오기
# 페이지에서 가게이름 클릭 -> 가격, 주소가 나와있음 
# 개발자 코드로 클래스와 이름 찾기  -> p 태그에 addy 클래스 

from bs4 import BeautifulSoup 
from urllib.request import urlopen

import pandas as pd

df = pd.read_csv('C:/Users/serah/ds/data/03. best_sandwiches_list_chicago.csv', index_col=0)
df.head()

df['URL'][0]

html = urlopen(df['URL'][0]) 
soup_tmp = BeautifulSoup(html, "html.parser")
soup_tmp

print(soup_tmp.find('p', 'addy'))

price_tmp = soup_tmp.find('p', 'addy').get_text() # 텍스트로 먼저 가져오고 
price_tmp

price_tmp.split()  # 나눠서 리스트 형태로 저장 

price_tmp.split()[0]

price_tmp.split()[0][:-1] # 가격 가져오기 

# 주소 가져오기 (주소는 나눈 리스트에서 [1:-2] )   
# ' '.join( list ) : 리스트에서 문자열으로 바꿔주기 
' '.join(price_tmp.split()[1:-2]) 

price = []
address = []

for n in df.index[:3]:
    html = urlopen(df['URL'][n])
    soup_tmp = BeautifulSoup(html, 'lxml')
    
    gettings = soup_tmp.find('p', 'addy').get_text()
    
    price.append(gettings.split()[0][:-1])
    address.append(' '.join(gettings.split()[1:-2]))

price

address



## Jupyter Notebook에서 상태 진행바를 쉽게 만들어주는 tqdm 모듈
## 상태 진행바까지 적용하고 다시 샌드위치페이지 50개에 접근하기
from tqdm import tqdm_notebook

price = []
address = []

for n in tqdm_notebook(df.index):
    html = urlopen(df['URL'][n])
    soup_tmp = BeautifulSoup(html, 'lxml')
    
    gettings = soup_tmp.find('p', 'addy').get_text()
    
    price.append(gettings.split()[0][:-1])
    address.append(' '.join(gettings.split()[1:-2]))



# 5. 50개 웹 페이지에 대한 정보 가져오기
print(price)
print(address)

print(df.head(10))
print(len(price),len(address), len(df))  #>>> 50 50 50 잘 가져옴! 

# df에 가격과 주소 추가 
df['Price'] = price
df['Address'] = address

df = df.loc[:, ['Rank', 'Cafe', 'Menu', 'Price', 'Address']]
df.set_index('Rank', inplace=True)
df.head()

# 중간중간 저장 
df.to_csv('C:/Users/serah/ds/data/03. best_sandwiches_list_chicago2.csv', sep=',', 
          encoding='UTF-8')



# 6. 맛집 위치를 지도에 표기하기
import folium
import pandas as pd
import googlemaps
import numpy as np

df = pd.read_csv('C:/Users/serah/ds/data/03. best_sandwiches_list_chicago2.csv', index_col=0)
df.head(5)

gmaps = googlemaps.Client(key=' ')

lat = []
lng = []

for n in tqdm_notebook(df.index):
    if df['Address'][n] != 'Multiple':
        target_name = df['Address'][n]+', '+'Chicago'
        gmaps_output = gmaps.geocode(target_name)
        location_output = gmaps_output[0].get('geometry')
        lat.append(location_output['location']['lat'])
        lng.append(location_output['location']['lng'])
        
    else:
        lat.append(np.nan)
        lng.append(np.nan)

len(lat), len(lng)

df['lat'] = lat
df['lng'] = lng
df

# 주소에 multiple이 나타나는 경우 ->  주소를 위도,경도 평균값으로 대체 
mapping = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], 
                                      zoom_start=11)
folium.Marker([df['lat'].mean(), df['lng'].mean()], 
                                      popup='center').add_to(mapping)
mapping

mapping = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], 
                     zoom_start=11)

for n in df.index:
    if df['Address'][n] != 'Multiple':
        folium.Marker([df['lat'][n], df['lng'][n]], 
                                      popup=df['Cafe'][n]).add_to(mapping)

mapping

# circle marker 적용할때, fill=True 





# 네이버 평점 변화 확인하기 
# goo.gl/f5cHRG  - 2017년 8월 6일


### 네이버 영화 평점 기준 영화의 평점변화 확인하기

from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlopen

url_base = "https://movie.naver.com/"
url_sub = "movie/sdb/rank/rmovie.nhn?sel=cur&date=20170806"
url = url_base+url_sub  #sub, sub로 사용하기 (둘다 되긴 됨) 

html = urlopen(url)
soup = BeautifulSoup(html, "html.parser")

soup



# <div class="tit5">
# <a href="/movie/bi/mi/basic.nhn?code=62586" title="다크 나이트">다크 나이트</a>
# </div>


soup.find_all('div', class_ ='tit5')


len(soup.find_all('div', class_ ='tit5'))

# 제목찾기   <div class="tit5">
soup.find_all('div','tit5')[0].a.string

# 평점 찾기  <td class="point">9.32</td>
soup.find_all('td','point')[0].string

moviename = [soup.find_all('div','tit5')[n].a.string for n in range(0,47)]
moviename

# 날짜를 5/1~100일간 정의, 해당 영화정보 찾기
date = pd.date_range('2017-05-01', periods = 100, freq = 'D')
date

import urllib
from tqdm import tqdm_notebook

movie_date = []
movie_name = []
movie_point = []

for today in tqdm_notebook(date) :
    html = "https://movie.naver.com/" + "movie/sdb/rank/rmovie.nhn?sel=cur&date={date}"
    response = urlopen(html.format(date=urllib.parse.quote(today.strftime('%Y%m%d'))))
    soup = BeautifulSoup(response, "html.parser")
    
    end = len(soup.find_all('td','point'))
    
    movie_date.extend([today for n in range(0,end)])
    movie_name.extend([soup.find_all('div','tit5')[n].a.string for n in range(0,end)])
    movie_point.extend([soup.find_all('td', 'point')[n].string for n in range(0,end)])
    
    
# for문 html에서 마지막에 {date}로 지정한 것은 
# 밑에 response에서 {date}를 변수로 취급하고 내용을 바꾸려고! 

# extent는 리스트형태로 리스트에 집어넣고 append는 value별로 각각 리스트에 넣음 

len(movie_name), len(movie_date), len(movie_point)

import pandas as pd
movie = pd.DataFrame({'date':movie_date, 'name':movie_name, 'point':movie_point})
movie['point'] = movie['point'].astype(float)   # point를 float로 바꿔주기 
movie.head()

movie.info()

# 참고) 날짜가 아니라 점수의 합산으로 보려면 pivot_table
import numpy as np
movie_unique = pd.pivot_table(movie, index = ['name'], aggfunc = np.sum)
movie_best = movie_unique.sort_values(by = 'point', ascending = False)
movie_best.head() 

tmp = movie.query('name ==["노무현입니다"]')
type(tmp)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize = (12,8))

plt.plot(tmp['date'], tmp['point'])
plt.legend(loc = 'best')
plt.grid()
plt.show()

### 영화/날짜별 평점 변화 확인하기 

movie_p = pd.pivot_table(movie, index=['date'], columns = ['name'], values = ['point'] )
movie_p.head()

movie_p.columns = movie_p.columns.droplevel()  # 레벨수준을 낮추는것 (위에 point삭제 )
movie_p.head()


# 그래프 한글문제 해결 
import platform
from matplotlib import font_manager, rc     
plt.rcParams['axes.unicode_minus'] = False 

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~') 

movie_p.plot(y = ['히든 피겨스','사운드 오브 뮤직', '시네마 천국'],
             figsize=(20,10))
plt.legend(loc = 'best')
plt.grid()
plt.show()


#%% Ch4) 셀프 주유소는 정말 저렴할까? 
#  goo.gl/VH1A5t : 오피넷(주유소 정보)

# tab 치면 뒤에 필요한거 나옴 

# 0. 준비 
from selenium import webdriver
driver= webdriver.Chrome('C:/Users/serah/ds/driver/chromedriver.exe')
driver.get('http://www.opinet.co.kr/searRgSelect.do')

# 1. 서울시 구별 주유소 가격 정보 얻기 
# 지역별 주유소/충전소 찾기 누르고 
# 개발자도구 - 동작구에 올려놓고 copy xpath ->  //*[@id="SIGUNGU_NM0"]
# select밑에 option태그에 구 이름 저장되어있음 
gu_list_raw = driver.find_element_by_xpath('''//*[@id="SIGUNGU_NM0"]''')
gu_list= gu_list_raw.find_elements_by_tag_name('option') 
print(gu_list)

gu_names = [option.get_attribute('value') 
            for option in gu_list]
gu_names.remove('')
gu_names

# gu_names의 첫번째[0]인 강남구-> 테스트 
element = driver.find_element_by_id('SIGUNGU_NM0')
element.send_keys(gu_names[0])
#>>> 웹페이지 에서 강남구로 선택되어서 바뀜 

# 조회 버튼 누르기 위해 xpath copy
# copy하고 뒤에 /span은 떼기
xpath= '''//*[@id="searRgSelect"]'''
element_get_excel = driver.find_element_by_xpath(xpath).click()

# 강남구만 ->  엑셀저장
# xpath = '''//*[@id="glopopd_excel"]'''
element_get_excel = driver.find_element_by_xpath(xpath).click()

# 전체 서울 25개 구의 엑셀파일 얻기 위한 for문
import time
from tqdm import tqdm_notebook   #중간중간 쉬었다가 해

for gu in tqdm_notebook(gu_names) :
    element = driver.find_element_by_id('SIGUNGU_NM0')
    element.send_keys(gu)
    
    time.sleep(2)
    
    xpath = '''//*[@id="searRgSelect"]'''
    element_sel_gu =driver.find_element_by_xpath(xpath).click()
    time.sleep(1)
    
    xpath = '''//*[@id="glopopd_excel"]'''
    element_get_excel = driver.find_element_by_xpath(xpath).click()
    
    time.sleep(1)
    
#>>> C:/다운로드 파일에 저장됨 

driver.close()

# 2. 구별 주유 가격 데이터 정리
# 파일 -> C:/Users/serah/ds/data 로 옮기기 

import pandas as pd
from glob import glob
glob('C:/Users/serah/ds/data/지역*.xls')

station_files = glob('C:/Users/serah/ds/data/지역*.xls')
station_files

tmp_raw =[]

for file_name in station_files :
    tmp = pd.read_excel(file_name, header = 2)
    tmp_raw.append(tmp)
    
station_raw = pd.concat(tmp_raw)  #한번에 다 합치기 

station_raw.info() # 가격정보가 object 나중에 int나 float로 바꿀 예정 

station_raw.head()

stations= pd.DataFrame({'oil_store':station_raw['상호'],
                        '주소':station_raw['주소'],
                        '가격':station_raw['휘발유'],
                        '셀프':station_raw['셀프여부'],
                        '상표':station_raw['상표'] })
stations.head()

# stations 일단 저장
import xlwt
stations.to_excel('C:/Users/serah/ds/output/stations_xls.xls')

stations.info()

# '구'컬럼 생성  
stations['구'] = [eachaddress.split()[1] for eachaddress in stations['주소']]
stations.head()

# 500개 전체다 볼수 없으니 .unique검사 수행
stations['구'].unique()

# >>>  '특별시' 가 들어있음 

stations[stations['구']=='서울특별시'] #>>>주소에 1이 포함되어있어서 혼자 서울특별시

stations[stations['구']=='특별시'] #>>> 서울 특별시로 되어있어서 특별시가 됨! 

stations.loc[stations['구']=='서울특별시', '구'] = '성동구'
stations.loc[stations['구']=='특별시','구'] = '도봉구'

stations['구'].unique()  # 예외처리하고, 결과 한번더 확인! 

# 가격이 object인 이유는 가격정보 없는 주유소에 '-'가 기입되어있음 
stations[stations['가격']=='-']

# 가격정보 없으면 예외처리하기
stations = stations[stations['가격'] != '-']
stations.head()

stations.info()  # 4개 삭제됨! 

# 가격 float로 변경
# stations['가격'] = stations['가격'].astype(float) 로 해도됨 
stations['가격'] = [float(i) for i in stations['가격'] ]
stations.info()

# 중복된 인덱스 제거하기 위해 .reset_index(inplace=True)로 새로 인덱스 부여하고 지우기
stations.reset_index(inplace = True)

stations.head()

del stations['index']
stations.head()

# 3. 박스 플롯으로 확인하기 (셀프 주유소가 저렴한지!?) 

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import platform

from matplotlib import font_manager, rc      
plt.rcParams['axes.unicode_minus'] = False    

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~') 


# 셀프 y or n
stations.boxplot(column='가격', by='셀프', figsize=(12,8))

# 주유소 브랜드/셀프유무 별 
plt.figure(figsize=(12,8))
sns.boxplot(x='상표', y = '가격', hue= '셀프', data = stations, palette = 'Blues')
plt.show()

# 데이터 분포 포함한 브랜드별 
plt.figure(figsize = (12,8))
sns.boxplot(x = '상표', y = '가격', data = stations, palette = 'Set3')
sns.swarmplot(x = '상표', y = '가격', data = stations, color = ".6")
plt.show()

# 4. 서울시 구별 주유 가격 확인하기
import json
import folium
import googlemaps
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

stations.sort_values(by = '가격', ascending = False).head()

stations.sort_values(by = '가격', ascending = True).head()

# 구별 가격 평균을 구하기 위해 
import numpy as np
gu_data = pd.pivot_table(stations, index = ['구'], values=['가격'], aggfunc=np.mean)
gu_data.head()

geo_path = 'C:/Users/serah/ds/data/02. skorea_municipalities_geo_simple.json'
geo_data =json.load(open(geo_path, encoding ='utf-8'))

map = folium.Map(location =[37.5502, 126.982], zoom_start=10.5, tiles='Stamen Toner')
map.choropleth(geo_data=geo_data,
               data = gu_data,
               columns =[gu_data.index,'가격'],
               fill_color ='Blues',   #PuRd, YlGnBu
               key_on='feature.id')
map

# 5. 서울시 주유 가격 상하위 10개를 지도에 표시하기
oil_price_top10= stations.sort_values(by = '가격', ascending = False).head(10)
print(oil_price_top10)
oil_price_bottom10 = stations.sort_values(by ='가격', ascending = True).head(10)
print(oil_price_bottom10)

gmaps = googlemaps.Client(key = ' ')

# 위도 경도 얻어오기, 찾을 수 없는 경우를 대비해 try, except구문 사용 (에러나면 nan처리)
from tqdm import tqdm_notebook
lat = []
lng = []

for i in tqdm_notebook(oil_price_top10.index):
    try:
        tmp_add = str(oil_price_top10['주소'][i]).split('(')[0]
        tmp_map = gmaps.geocode(tmp_add)
        
        tmp_loc = tmp_map[0].get('geometry')
        lat.append(tmp_loc['location']['lat'])
        lng.append(tmp_loc['location']['lng'])
        
    except :
        lat.append(np.nan)
        lng.append(np.nan)
        print('here is nan!')
        
oil_price_top10['lat'] = lat
oil_price_top10['lng'] = lng
oil_price_top10

# oil_price_bottom10 도 같은 작업 
from tqdm import tqdm_notebook
lat = []
lng = []

for i in tqdm_notebook(oil_price_bottom10.index):
    try:
        tmp_add = str(oil_price_bottom10['주소'][i]).split('(')[0]
        tmp_map = gmaps.geocode(tmp_add)
        
        tmp_loc = tmp_map[0].get('geometry')
        lat.append(tmp_loc['location']['lat'])
        lng.append(tmp_loc['location']['lng'])
        
    except :
        lat.append(np.nan)
        lng.append(np.nan)
        print('here is nan!')
        
oil_price_bottom10['lat'] = lat
oil_price_bottom10['lng'] = lng
oil_price_bottom10

oil_price_top10.info()

# pd.notnull()로 nan이 아닌 것만 표시 
map = folium.Map(location = [37.5502, 126.982], zoom_start=10.5)
for i in oil_price_top10.index:
    if pd.notnull(oil_price_top10['lat'][i]):
        folium.CircleMarker([oil_price_top10['lat'][i], oil_price_top10['lng'][i]],
                            radius =20,
                            color = '#CD3181',
                            fill_color= '#CD3181').add_to(map)
    for i in oil_price_bottom10.index:
        if pd.notnull(oil_price_bottom10['lat'][i]):
            folium.CircleMarker([oil_price_bottom10['lat'][i], oil_price_bottom10['lng'][i]],
                                radius =20,
                                color = '#3186cc',
                                fill_color= '#3186cc').add_to(map)   
map


#%% Ch 5) 우리나라 인구 소멸 위기지역 분석
# kosis.kr/index.index.jsp

# 0. lib, data준비 
import pandas as pd
import numpy as np

import platform
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

path = "c:/Windows/Fonts/malgun.ttf"
from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')    

plt.rcParams['axes.unicode_minus'] = False

population = pd.read_excel('C:/Users/serah/ds/data/05. population_raw_data.xlsx', header=1)
population.fillna(method='pad',inplace =True)
population.head()

# .fillna(method = 'pad', inplace =True) : nan처리하지 않고 그 앞내용으로 채우기

# 1. 데이터 전처리
# 변수명 rename
population.rename(columns = {'행정구역(동읍면)별(1)':'광역시도',
                     '행정구역(동읍면)별(2)' : '시도',
                     '계' : '인구수'}, inplace =True)
population.head(100)

# '행정구역(동읍면)별(2)' : '시도' 에 '소계'라는 항목 삭제 -> 제외하고 불러오기 
# 846개 -> 792개
population  = population[(population['시도'] != '소계')]  
population.head()

population.info()  

# 원래 '항목' 컬럼 -> '구분'으로 rename
population.rename(columns = {'항목':'구분'}, inplace=True)

# value의 값을 바꾸려면 df.loc로 [df['col'] =='old value','col'] ='new value'
population.loc[population['구분'] == '총인구수 (명)', '구분'] = '합계'
population.loc[population['구분'] == '남자인구수 (명)', '구분'] = '남자'
population.loc[population['구분'] == '여자인구수 (명)', '구분'] = '여자'

population

# '20-39세', '65세이상' 컬럼을 새로 만들기
population['20-39세'] = population['20 - 24세'] + population['25 - 29세'] +                         population['30 - 34세'] + population['35 - 39세']
    
population['65세이상'] = population['65 - 69세'] + population['70 - 74세'] +                         population['75 - 79세'] + population['80 - 84세'] +                         population['85 - 89세'] + population['90 - 94세'] +                         population['95 - 99세'] + population['100+']
            
population.head(10)

# 일부만 선택하기 위해 pd.pivot_table 사용 
pop = pd.pivot_table(population, 
                     index = ['광역시도', '시도'], 
                     columns = ['구분'],
                     values = ['인구수', '20-39세', '65세이상'])
pop

# 인구 소멸 비율 계산해서 새로운 col로 넣기 
# pivot_table은 df['큰틀','중간틀','작은틀'] 로 필요한 부분 가져오기 
pop['소멸비율'] = pop['20-39세','여자'] / (pop['65세이상','합계'] / 2)
pop.head()

# 비율이  1보다 작으면 소멸위기! -> 새로운 col지정 
pop['소멸위기지역'] = pop['소멸비율'] < 1.0
pop.head()

#소멸위기지역 리스트 얻기 
pop[pop['소멸위기지역']==True].index.get_level_values(1)

pop['소멸위기지역']==True

# pivot-> 다단 인덱스 지우고 새로 0부터 부여 (인덱스 초기화)
pop.reset_index(inplace=True) 
pop.head()

# pivot으로 인해 다단으로 된 컬럼을 하나로 합치기 - for문으로 
print(pop.info())
print(pop.columns)

tmp_coloumns = [pop.columns.get_level_values(0)[n] + pop.columns.get_level_values(1)[n] 
                for n in range(0,len(pop.columns.get_level_values(0)))]

pop.columns = tmp_coloumns

pop.head()

pop.info()

# 2. 대한민국 지도 그리기 - folium 사용 , 대한민국 경계선 그려진 json 파일 활용

# 지도 시각화를 위한 고유 ID만들기 - json파일 사용하기 위해 맞춰줌
pop['시도'].unique()

# 광역시의 구(=자치구), 시에 있는 구(=행정구)  -> 광역시도 +시도를 합치면 될듯하지만
# 경기도 동안구가 되므로 dict 활용하여 구분필요 

si_name = [None] * len(pop)
print(si_name)  #>>> None으로 pop의 개수만큼 빈 리스트 생성

# 시 인데 구를 가지는 경우 
tmp_gu_dict = {'수원':['장안구', '권선구', '팔달구', '영통구'], 
                       '성남':['수정구', '중원구', '분당구'], 
                       '안양':['만안구', '동안구'], 
                       '안산':['상록구', '단원구'], 
                       '고양':['덕양구', '일산동구', '일산서구'], 
                       '용인':['처인구', '기흥구', '수지구'], 
                       '청주':['상당구', '서원구', '흥덕구', '청원구'], 
                       '천안':['동남구', '서북구'], 
                       '전주':['완산구', '덕진구'], 
                       '포항':['남구', '북구'], 
                       '창원':['의창구', '성산구', '진해구', '마산합포구', '마산회원구'], 
                       '부천':['오정구', '원미구', '소사구']}
print(tmp_gu_dict) #dict형태

# 광역시도:[광역시,특별시,자치시] 가 아니라면 일반 시/군 으로 보고
# 동일한 이름인 강원도 고성군, 경상남도 고성군 처리
# 세종특별자치시는 세종으로 처리
# 광역시도에서 앞 두글자와 시도에서 두글자인 경우 모두선택, 아니면 앞 두글자만 선택

for n in pop.index:
    if pop['광역시도'][n][-3:] not in ['광역시', '특별시', '자치시']:
        if pop['시도'][n][:-1]=='고성' and pop['광역시도'][n]=='강원도':
            si_name[n] = '고성(강원)'
        elif pop['시도'][n][:-1]=='고성' and pop['광역시도'][n]=='경상남도':
            si_name[n] = '고성(경남)'
        else:
             si_name[n] = pop['시도'][n][:-1]
                
        for keys, values in tmp_gu_dict.items():
            if pop['시도'][n] in values:
                if len(pop['시도'][n])==2:
                    si_name[n] = keys + ' ' + pop['시도'][n]
                elif pop['시도'][n] in ['마산합포구','마산회원구']:
                    si_name[n] = keys + ' ' + pop['시도'][n][2:-1]
                else:
                    si_name[n] = keys + ' ' + pop['시도'][n][:-1]
        
    elif pop['광역시도'][n] == '세종특별자치시':
        si_name[n] = '세종'
        
    else:
        if len(pop['시도'][n])==2:
            si_name[n] = pop['광역시도'][n][:2] + ' ' + pop['시도'][n]
        else:
            si_name[n] = pop['광역시도'][n][:2] + ' ' + pop['시도'][n][:-1]

si_name

pop['ID'] = si_name

# 필요없는 컬럼 삭제 del (행은 drop으로 삭제)
del pop['20-39세남자']
del pop['65세이상남자']
del pop['65세이상여자']

pop.head()

# Cartogram으로 우리나라 지도 만들기  - 저자가 엑셀로 한국지도 그려놓음 

draw_korea_raw = pd.read_excel('C:/Users/serah/ds/data/05. draw_korea_raw.xlsx', encoding = 'EUC-KR')
draw_korea_raw

# 그림의 모양이 아니라 지역별 (x,y)좌표 필요
# 행정구역의 화면상 좌표를 얻기위해 .stack()명령 사용 - pivot의 반대개념 

# stack()으로 풀고, 인덱스 재설정(reset_index), colname변경

draw_korea_raw_stacked = pd.DataFrame(draw_korea_raw.stack())
draw_korea_raw_stacked.reset_index(inplace =True)
draw_korea_raw_stacked.rename (columns ={'level_0':'y', 'level_1':'x', 0:'ID'},
                                        inplace = True)
draw_korea_raw_stacked

draw_korea = draw_korea_raw_stacked # 변수이름 변경 

# 광역시도를 구분하는 경계선 입력 

BORDER_LINES =[
            [(5, 1), (5,2), (7,2), (7,3), (11,3), (11,0)], # 인천
            [(5,4), (5,5), (2,5), (2,7), (4,7), (4,9), (7,9), 
             (7,7), (9,7), (9,5), (10,5), (10,4), (5,4)], # 서울
            [(1,7), (1,8), (3,8), (3,10), (10,10), (10,7), 
             (12,7), (12,6), (11,6), (11,5), (12, 5), (12,4), 
             (11,4), (11,3)], # 경기도
            [(8,10), (8,11), (6,11), (6,12)], # 강원도
            [(12,5), (13,5), (13,4), (14,4), (14,5), (15,5), 
             (15,4), (16,4), (16,2)], # 충청북도
            [(16,4), (17,4), (17,5), (16,5), (16,6), (19,6), 
             (19,5), (20,5), (20,4), (21,4), (21,3), (19,3), (19,1)], # 전라북도
            [(13,5), (13,6), (16,6)], # 대전시
            [(13,5), (14,5)], #세종시
            [(21,2), (21,3), (22,3), (22,4), (24,4), (24,2), (21,2)], #광주
            [(20,5), (21,5), (21,6), (23,6)], #전라남도
            [(10,8), (12,8), (12,9), (14,9), (14,8), (16,8), (16,6)], #충청북도
            [(14,9), (14,11), (14,12), (13,12), (13,13)], #경상북도
            [(15,8), (17,8), (17,10), (16,10), (16,11), (14,11)], #대구
            [(17,9), (18,9), (18,8), (19,8), (19,9), (20,9), (20,10), (21,10)], #부산
            [(16,11), (16,13)], #울산
        #     [(9,14), (9,15)], 
            [(27,5), (27,6), (25,6)],
]

# 지역이름 표시를 위해 만들어진 코드 가져옴

plt.figure(figsize=(8, 11))

# 지역 이름 표시
for idx, row in draw_korea.iterrows():
    
    # 광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시한다. 
    # (중구, 서구)
    if len(row['ID'].split())==2:
        dispname = '{}\n{}'.format(row['ID'].split()[0], row['ID'].split()[1])
    elif row['ID'][:2]=='고성':
        dispname = '고성'
    else:
        dispname = row['ID']

    # 서대문구, 서귀포시 같이 이름이 3자 이상인 경우에 작은 글자로 표시한다.
    if len(dispname.splitlines()[-1]) >= 3:
        fontsize, linespacing = 9.5, 1.5
    else:
        fontsize, linespacing = 11, 1.2

    plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',
                 fontsize=fontsize, ha='center', va='center', 
                 linespacing=linespacing)
    
# 시도 경계 그린다.
for path in BORDER_LINES:
    ys, xs = zip(*path)
    plt.plot(xs, ys, c='black', lw=1.5)


# y축이 엑셀에선 0부터 작, matplotlib이 0이라고 인식하는 좌표가 반대이기때문에 사용     
plt.gca().invert_yaxis()  
#plt.gca().set_aspect(1)

plt.axis('off')

plt.tight_layout()
plt.show()

# 인구분석결과인 pop과 지도를 그리기 위한 draw_korea를 합칠때 필요한 key인 ID에 문제가 없는지 확인
set(draw_korea['ID'].unique())-set(pop['ID'].unique())

# 반대로도 확인 
set(pop['ID'].unique()) -set(draw_korea['ID'].unique()) 

# pop 행정구를 가진 데이터가 더 있지만 지도에 표현하지 못하니 삭제! 
tmp_list = list(set(pop['ID'].unique())- set(draw_korea['ID'].unique()))
for tmp in tmp_list :
    pop = pop.drop(pop[pop['ID']==tmp].index)
    
print(set(pop['ID'].unique())- set(draw_korea['ID'].unique()))

pop.head()

# pop과 draw_korea merge key는 ID
pop = pd.merge(pop, draw_korea, how ='left', on = ['ID'])
pop.head()

# 인구수 합계 지도 그리기
mapdata =pop.pivot_table(index = 'y', columns ='x', values = '인구수합계')
masked_mapdata =np.ma.masked_where(np.isnan(mapdata), mapdata)

mapdata

masked_mapdata

# 인구수합계로 지도 그리기 (만들어져있던 코드 사용 )
def drawKorea(targetData, blockedMap, cmapname):
    gamma = 0.75

    whitelabelmin = (max(blockedMap[targetData]) - 
                                     min(blockedMap[targetData]))*0.25 + \
                                                                min(blockedMap[targetData])

    datalabel = targetData

    vmin = min(blockedMap[targetData])
    vmax = max(blockedMap[targetData])

    mapdata = blockedMap.pivot_table(index='y', columns='x', values=targetData)
    masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)
    
    plt.figure(figsize=(9, 11))
    plt.pcolor(masked_mapdata, vmin=vmin, vmax=vmax, cmap=cmapname, 
               edgecolor='#aaaaaa', linewidth=0.5)

    # 지역 이름 표시
    for idx, row in blockedMap.iterrows():
        # 광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시한다. 
        #(중구, 서구)
        if len(row['ID'].split())==2:
            dispname = '{}\n{}'.format(row['ID'].split()[0], row['ID'].split()[1])
        elif row['ID'][:2]=='고성':
            dispname = '고성'
        else:
            dispname = row['ID']

        # 서대문구, 서귀포시 같이 이름이 3자 이상인 경우에 작은 글자로 표시한다.
        if len(dispname.splitlines()[-1]) >= 3:
            fontsize, linespacing = 10.0, 1.1
        else:
            fontsize, linespacing = 11, 1.

        annocolor = 'white' if row[targetData] > whitelabelmin else 'black'
        plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',
                     fontsize=fontsize, ha='center', va='center', color=annocolor,
                     linespacing=linespacing)

    # 시도 경계 그린다.
    for path in BORDER_LINES:
        ys, xs = zip(*path)
        plt.plot(xs, ys, c='black', lw=2)

    plt.gca().invert_yaxis()

    plt.axis('off')

    cb = plt.colorbar(shrink=.1, aspect=10)
    cb.set_label(datalabel)

    plt.tight_layout()
    plt.show()

# 7. 인구현황 및 인구 소멸지역 확인하기
drawKorea('인구수합계', pop, 'Blues')

# 인구 소멸위기 지역을 표시 
# bool형인 것을 1과 0으로 바꾸고 표시(T or F)
pop['소멸위기지역'] = [1 if con else 0 for con in pop['소멸위기지역']]
drawKorea('소멸위기지역',pop, 'Reds')

# 8. 인구현황에서 여성인구비율 확인하기
# 위의 함수 살짝 수정필요 - 데이터에 음(-)값의 여부에 따라 일부설정 바뀌어야함
def drawKorea(targetData, blockedMap, cmapname):
    gamma = 0.75

    whitelabelmin = 20.

    datalabel = targetData

    tmp_max = max([ np.abs(min(blockedMap[targetData])), 
                                  np.abs(max(blockedMap[targetData]))])
    vmin, vmax = -tmp_max, tmp_max

    mapdata = blockedMap.pivot_table(index='y', columns='x', values=targetData)
    masked_mapdata = np.ma.masked_where(np.isnan(mapdata), mapdata)
    
    plt.figure(figsize=(9, 11))
    plt.pcolor(masked_mapdata, vmin=vmin, vmax=vmax, cmap=cmapname, 
               edgecolor='#aaaaaa', linewidth=0.5)

    # 지역 이름 표시
    for idx, row in blockedMap.iterrows():
        # 광역시는 구 이름이 겹치는 경우가 많아서 시단위 이름도 같이 표시한다. 
        #(중구, 서구)
        if len(row['ID'].split())==2:
            dispname = '{}\n{}'.format(row['ID'].split()[0], row['ID'].split()[1])
        elif row['ID'][:2]=='고성':
            dispname = '고성'
        else:
            dispname = row['ID']

        # 서대문구, 서귀포시 같이 이름이 3자 이상인 경우에 작은 글자로 표시한다.
        if len(dispname.splitlines()[-1]) >= 3:
            fontsize, linespacing = 10.0, 1.1
        else:
            fontsize, linespacing = 11, 1.

        annocolor = 'white' if np.abs(row[targetData]) > whitelabelmin else 'black'
        plt.annotate(dispname, (row['x']+0.5, row['y']+0.5), weight='bold',
                     fontsize=fontsize, ha='center', va='center', color=annocolor,
                     linespacing=linespacing)

    # 시도 경계 그린다.
    for path in BORDER_LINES:
        ys, xs = zip(*path)
        plt.plot(xs, ys, c='black', lw=2)

    plt.gca().invert_yaxis()

    plt.axis('off')

    cb = plt.colorbar(shrink=.1, aspect=10)
    cb.set_label(datalabel)

    plt.tight_layout()
    plt.show()

pop.head()

# 인구 전체에 대한 여성비
# 0.5를 빼서 -> 결과가 0이면 여성비율이 50퍼센트 인것! 즉 파란색일수록 여성비가 높, 빨간색이면 낮  
pop['여성비'] = (pop['인구수여자']/pop['인구수합계']-0.5)*100
drawKorea('여성비', pop, 'RdBu')

# 2030 여성비
pop['2030여성비'] = (pop['20-39세여자']/pop['20-39세합계']-0.5) *100
drawKorea('2030여성비', pop, 'RdBu')

# 8. folium에서 인구 소멸위기지역 표현하기
pop_folium = pop.set_index('ID') # -> folium 에서 쉽게 인식가능
pop_folium.head()

import folium
import json
import warnings 
warnings.simplefilter(action = 'ignore', category =FutureWarning)

geo_path = 'C:/Users/serah/ds/data/05. skorea_municipalities_geo_simple.json'
geo_data=json.load(open(geo_path, encoding ='utf-8'))

map = folium.Map(location= [36.2002, 127.054], zoom_start=7)
map.choropleth(geo_data =geo_data,
              data = pop_folium['인구수합계'],
              columns = [pop_folium.index, pop_folium['인구수합계']],
              fill_color = 'YlGnBu', #PuRd, YlGnBu
              key_on = 'feature.id')
map

map = folium.Map(location =[36.2002, 127.054], zoom_start=7)
map.choropleth(geo_data=geo_data,
               data = pop_folium['소멸위기지역'],
               columns = [pop_folium.index, pop_folium['소멸위기지역']],
               fill_color = 'Blues',
               key_on = 'feature.id')
map

draw_korea.to_csv('C:/Users/serah/ds/data/05. draw_korea.csv', encoding ='utf-8', sep=',')


#%% ch6) 19대 대선 자료 분석하기




#%% ch7) 시계열 데이터를 다뤄보자

## 7-1. Numpy의 polyfit으로 회귀(regression) 분석하기
* visual studio 2015 build tools 
* pip install prophet / pandas_datareader / pystan
* conda install -c conda-forge fbprophet

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from fbprophet import Prophet
from datetime import datetime

path = "c:/Windows/Fonts/malgun.ttf"
import platform
from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')    

plt.rcParams['axes.unicode_minus'] = False

pinkwink_web = pd.read_csv('C:/Users/serah/ds/data/08. PinkWink Web Traffic.csv', 
                                          encoding='utf-8', thousands=',',
                                          names = ['date','hit'], index_col=0)
pinkwink_web = pinkwink_web[pinkwink_web['hit'].notnull()]
pinkwink_web.head()

pinkwink_web['hit'].plot(figsize=(12,4), grid=True);

# 데이터를 설명할 간단한 함수를 찾으려고 함 -> 회귀(regression)
# 시간축 time 만들고 hit를 traffic 변수에 저장
time = np.arange(0,len(pinkwink_web))
traffic = pinkwink_web['hit'].values

# 시작점과 끝점을 균일간격으로 나눈 점들을 생성해주는 linspace() 
fx = np.linspace(0, time[-1], 1000)

print(traffic)

print(fx)

# 최소제곱법 SSE 근사적으로 구하려는 해와 실제 해의 오차의 제곱의 합이 최소가 되는 해를 구하는 방법 
# 참 값과 비교해서 error를 계산해야함 함수 정의
def error(f, x, y):
    return np.sqrt(np.mean((f(x)-y)**2))

# polyfit : reg선 그려줌
# 입력값(x), 출력값(y)으로 다항식의 계수 a,b,c..를 찾아줌
# p= polyfit(x,y,n) n은 차수  -> p는 차수에 따른 다항식의 계수값 
fp1 = np.polyfit(time, traffic, 1) #1차 함수
f1 = np.poly1d(fp1)

f2p = np.polyfit(time, traffic, 2)
f2 = np.poly1d(f2p)

f3p = np.polyfit(time, traffic, 3)
f3 = np.poly1d(f3p)

f15p = np.polyfit(time, traffic, 15) # 15차 함수 
f15 = np.poly1d(f15p)

print(error(f1, time, traffic))
print(error(f2, time, traffic))
print(error(f3, time, traffic))
print(error(f15, time, traffic)) # 오차는 작지만, 과적합일 수 있다. 

# overfitting 과한정보 -> training 시 몇개의 뉴런은 쉬게 하고 나머지만 가지고 융통성있게 훈련 
#>>> 오차값들이 비슷하니 간단한 1차로 하는게 좋을 듯

plt.figure(figsize=(10,6))
plt.scatter(time, traffic, s=10)

plt.plot(fx, f1(fx), lw=4, label='f1')
plt.plot(fx, f2(fx), lw=4, label='f2')
plt.plot(fx, f3(fx), lw=4, label='f3')
plt.plot(fx, f15(fx), lw=4, label='f15')

plt.grid(True, linestyle='-', color='0.75')

plt.legend(loc=2)
plt.show()



## 7-2. Prophet 모듈을 이용한 forecast 예측
* fb에서 발표한 시계열 데이터 기반의 예측 lib 
* 통계적 지식이 없어도 파라미터 조정을 통해 모형조정, 내부가 어떻게 동작하는지 고민할 필요 없음 

# pinkwink_web 데이터에서 날짜(index)와 방문수(hit)만 따로 저장
df = pd.DataFrame({'ds':pinkwink_web.index, 'y':pinkwink_web['hit']})
df.reset_index(inplace=True)

# pd의 to_datetime함수로 날짜 선언
df['ds'] =  pd.to_datetime(df['ds'], format="%y. %m. %d.")
del df['date']

# prophet 사용할때 연단위 주기 있다고 알려주기
m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False) 
m.fit(df);

# periods 기간을 설정, 향후 60일간을 예측
future = m.make_future_dataframe(periods=60)
future.tail()

# prophet object의 predict method 사용 
# yhat: 예측값, yhat_lower/upper:예측의 최소/최대값 
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot(forecast);

m.plot_components(forecast);

## 7-3. Seasonal 시계열 분석으로 주식 데이터 분석하기

# from pandas_datareader import data
# import yfinance as yf
# yf.pdr_override()
# start_date = '2012-1-1' 
# end_date = '2017-6-30' 
# KIA = data.get_data_yahoo('000270.KS', start_date, end_date)
    
# KIA = web.DataReader('KRX:000270','google',start,end) # 구글용... 동작이 안됨
# KIA = web.DataReader('000270.KS','yahoo',start,end) # 구글용... 동작이 안됨


import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
from fbprophet import Prophet
from datetime import datetime

path = "c:/Windows/Fonts/malgun.ttf"
import platform
from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')    

plt.rcParams['axes.unicode_minus'] = False

# yahoo kospi지수 
start =datetime(2017,1,1)
end = datetime(2017,4,30)

df_kospi = web.get_data_yahoo('^KS11', start, end)
# df_kospi = web.DataReader('^KS11','yahoo', start, end)
df_kospi.head()

# 삼성
# start =datetime(2018,11,1)  # 0붙이면 안됨
# end = datetime(2018,11,30)
# df_s =web.get_data_yahoo('005930.KS', start, end)
df_s = web.get_data_yahoo('005930.KS','2018-11-01','2018-11-30')

df_s

df_s['Close'].plot()

# 기아
# start =datetime(2009,7,1)
# end = datetime(2019,7,31)

# 2009년도 나올때 까지 run
df_kia = web.get_data_yahoo('000270.KS', '2009-07-01', '2019-07-31')
df_kia

df_kia['Close'].plot()

print(df_kia.describe())

# 일부 데이터를 잘라서 forecast수행 (최근 2개월 제외)
# 실제와 비슷한지 밑에서 비교하기 위해 
kia_trunc=df_kia[:'2019-05-31']
kia_trunc['Close'].plot(figsize=(12,4), grid =True)

df = pd.DataFrame({'ds':kia_trunc.index, 'y':kia_trunc['Close']})
df.reset_index(inplace =True)
del df['Date']
df.head()

m = Prophet(yearly_seasonality=True, weekly_seasonality=True,  daily_seasonality= True)
m.fit(df)

future = m.make_future_dataframe(periods =365)
future.tail()

forecast =m.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail()

m.plot(forecast)

m.plot_components(forecast)

# 실제값과 비교 
plt.figure(figsize=(12,6))
plt.plot(df_kia.index, df_kia['Close'], label='real')
plt.plot(forecast['ds'],forecast['yhat'],label='forecast')
plt.grid()
plt.legend()
plt.title('real vs forecast')
plt.show()

## 7-4 Growth Model과 Holiday Forecast

df = pd.read_csv('C:/Users/serah/ds/data/08. example_wp_R.csv')
df['y'] = np.log(df['y'])

df['cap'] = 8.5

m = Prophet(growth='logistic', daily_seasonality=True)
m.fit(df)

future = m.make_future_dataframe(periods=1826)
future['cap'] = 8.5
fcst = m.predict(future)
m.plot(fcst);

forecast = m.predict(future)
m.plot_components(forecast);

## holiday

df = pd.read_csv('C:/Users/serah/ds/data/08. example_wp_peyton_manning.csv')
df['y'] = np.log(df['y'])
m = Prophet(daily_seasonality=True)
m.fit(df)
future = m.make_future_dataframe(periods=366)

df.y.plot(figsize=(12,6), grid=True);

playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))

m = Prophet(holidays=holidays, daily_seasonality=True)
forecast = m.fit(df).predict(future)

forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
        ['ds', 'playoff', 'superbowl']][-10:]

m.plot(forecast);

m.plot_components(forecast);




#%% ch8) 자연어 처리
'''
* 자연어(natural language): 일상에서 사용하는 언어 -> 분석하여 컴퓨터가 처리할 수 있도록 함
* 요구사항 분석, 텍스트 수집/저장,전처리/분석/분석서비스
* 토큰화(단어로 분리), 불용어제거(전치사,관사등 제거), 어간추출(단어의 기본형태 추출), 문서표현(문서나 문장을 하나의 벡터로 표현, 단어를 인덱싱, 빈도수 추출하여 표현)

##  한글 자연어 처리 기초 - KoNLPy 및 필요 모듈의 설치
* 설치 목록
    * JDK (Java SE Downloads) : 파이썬 버전 맞춰서
    * JAVA_HOME 설정 : 자바 환경변수 설정
    * pip install JPype1 /KoNLPy /Word Cloud/gensim   * 이후 Jupyter Notebook 재실행 필요
    * prompt)python>import nltk> nltk.download() > stopwords,punkt 다운
        
* nltk(english)
* gensim(topic modeling, 다양한 언어)

* jpype 에러나면 converStrings=False 사용
C:\Users\serah\Anaconda3\lib\site-packages\jpype\_core.py:210: UserWarning: 
Deprecated: convertStrings was not specified when starting the JVM. The default
behavior in JPype will be False starting in JPype 0.8. The recommended setting
for new code is convertStrings=False.  The legacy value of True was assumed for
this session. If you are a user of an application that reported this warning,
please file a ticket with the developer.
* 0.7.0하다가 에러나면 0.6.x버전 깔기
'''
## 2. 한글 자연어 처리 기초
# * konlpy - 꼬꼬마, 한나눔 엔진있음
# * http://kkma.snu.ac.kr/documents/index.jsp

import numpy as np
from konlpy.tag import Kkma
kkma = Kkma()

# 문장 분석(마침표 없어도 문장으로 인식)
kkma.sentences('한국어 분석을 시작합니다 재미있어요~~')

# 명사 분석
kkma.nouns('한국어 분석을 시작합니다 재미있어요~~')

# 형태소 분석 - part od speech 
kkma.pos('한국어 분석을 시작합니다 재미있어요~~')

# 한나눔  
from konlpy.tag import Hannanum
hannanum = Hannanum()

hannanum.nouns('한국어 분석을 시작합니다 재미있어요~~')

# 말뭉치
hannanum.morphs('한국어 분석을 시작합니다 재미있어요~~')

hannanum.pos('한국어 분석을 시작합니다 재미있어요~~')

# Okt() <- 구 Twitter
from konlpy.tag import Okt  #Twitter가 Okt로 바뀜 
t = Okt()

t.nouns('한국어 분석을 시작합니다 재미있어요~~')

t.morphs('한국어 분석을 시작합니다 재미있어요~~')

t.phrases('한국어 분석을 시작합니다 재미있어요~~')

t.pos('한국어 분석을 시작합니다 재미있어요~~')

# stem: True 이면 stem tokens
# norm : True이면 normalize tokens
# 매개변수 pos(phrase, norm =False, stem= False)
print(t.pos('이것도 재미 있습니당 ㅋㅋㅋㅋ'))
print(t.pos('이것도 재미 있습니당 ㅋㅋㅋㅋ', norm =True))
print(t.pos('이것도 재미 있습니당 ㅋㅋㅋㅋ', norm =True, stem =True))

## 3. 워드 클라우드

* WordCloud 설치 : **pip install wordcloud**

from wordcloud import WordCloud, STOPWORDS

import numpy as np
from PIL import Image

# 앨리스 영문판
text = open('C:/Users/serah/ds/data/09. alice.txt').read()
alice_mask = np.array(Image.open('C:/Users/serah/ds/data/09. alice_mask.png'))

stopwords = set(STOPWORDS)
stopwords.add("said")  #카운트에서 제외(said라는 단어가 많이 나와서)

import matplotlib.pyplot as plt
import platform

path = "c:/Windows/Fonts/malgun.ttf"
from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
   rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
   print('Unknown system... sorry~~~~') 
    
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(8,8))
plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis('off')
plt.show()

# WordCloud 자체적으로 단어카운트하는 기능 있음
wc = WordCloud(background_color='white', max_words=2000, 
               mask=alice_mask, stopwords = stopwords)
wc = wc.generate(text)
wc.words_

plt.figure(figsize=(12,12))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# 스타워즈 텍스트
text = open('C:/Users/serah/ds/data/09. a_new_hope.txt').read()

text = text.replace('HAN', 'Han') #HAN을 Han으로 replace
text = text.replace("LUKE'S", 'Luke')

mask = np.array(Image.open('C:/Users/serah/ds/data/09. stormtrooper_mask.png'))

# 제외할 단어 설정- 불용어처리
stopwords = set(STOPWORDS)
stopwords.add("int")
stopwords.add("ext")

# 워드 클라우드 설정 
wc = WordCloud(max_words=1000, mask=mask, stopwords=stopwords, 
               margin=10, random_state=1).generate(text)

default_colors = wc.to_array()

# 전체적으로 회색으로 처리하기 위해 함수하나 만들기
import random
def grey_color_func(word, font_size, position, orientation, 
                    random_state=None, **kwargs):
    return 'hsl(0, 0%%, %d%%)' % random.randint(60,100)

# hsl- 색상, 명도, 채도값

plt.figure(figsize=(12,12))
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
          interpolation='bilinear')
plt.axis('off')
plt.show()

## 육아휴직관련 법안 1809890호 

help(nltk)

import nltk

# konlpy에 내장된 법률 문서 여러개 있음 그중 하나

from konlpy.corpus import kobill
#files_ko = kobill.fileids()
doc_ko = kobill.open('1809890.txt').read()

doc_ko

#Okt(구 twitter분석기) 로 명사분석 
from konlpy.tag import Okt
t = Okt()
tokens_ko = t.nouns(doc_ko)
tokens_ko

# 수집된 단어의 횟수(len(ko.tokens))
# 고유한 횟수(len(set(ko.tokens))) 확인 
ko = nltk.Text(tokens_ko)

print(len(ko.tokens))           # returns number of tokens (document length)
print(len(set(ko.tokens)))   # returns number of unique tokens

# 딕셔너리 형태로 '단어':발생횟수 반환
ko.vocab()                        # returns frequency distribution

plt.figure(figsize=(12,6))
ko.plot(50)
plt.show()

# 한글은 stopwords 지정하기가 영어보다 까다로워서 for문으로 
stop_words = ['.', '(', ')', ',', "'", '%', '-', 'X', ').', '×',
              '의','자','에','안','번','호','을','이','다','만',
              '로','가','를']

# ko의 단어가 stop_words 에 해당하지 않으면 ko에 저장 
ko = [i for i in ko if i not in stop_words]

ko

ko = nltk.Text(ko)

plt.figure(figsize=(12,6))
ko.plot(50)     # Plot sorted frequency of top 50 tokens
plt.show()

ko.count('초등학교')

# 문서내 몇번 언급되었는지 확인
plt.figure(figsize=(12,6))
ko.dispersion_plot(['육아휴직', '초등학교', '공무원'])

# 특정 단어의 문서내 위치를 개략적으로 분량과 함께 알수 있음
ko.concordance('초등학교') 

# 문장 내에서 연어(collocation)로 사용되었는지 확인가능 
# 연어: 어떤 언어 내에서 특정한 뜻을 나타낼 때 흔히 함께 쓰이는 단어들의 결합 (연이어 나타나는)
# 원하는 단어의 주변부 단어까지 확인 

ko.collocations()


ko.collocation_list()

data = ko.vocab().most_common(150)

wordcloud = WordCloud(font_path='c:/Windows/Fonts/malgun.ttf',
                      relative_scaling = 0.1,
                      background_color='white'
                      ).generate_from_frequencies(dict(data))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()





## 5. Naive Bayes Classifier의 이해 - 영문

# 심플한 분류기 - 지도학습(SL- 정답 정해져있음)의 한종류, 두사건은 서로 독립이고 조건부확률 활용
# 지도학습의 특성 -pos긍정, neg부정 태그 존재
# 문장이 많을 수록 신뢰성이 올라감! 
from nltk.tokenize import word_tokenize
import nltk

# i like you -> pos   / you like me -> neg 
train = [('i like you', 'pos'), 
         ('i hate you', 'neg'), 
         ('you like me', 'neg'),
         ('i like her', 'pos')]
train

# train에서 사용된 전체 단어 찾기 , i.lower() 소문자
all_words = set(i.lower() for sentence in train
                for i in word_tokenize(sentence[0]))
all_words
#>>> {'hate', 'her', 'i', 'like', 'me', 'you'} : 말뭉치

# 하나의 문장만 먼저 테스트
a='i like you'
t =({i: (i in word_tokenize(a)) for i in all_words}, a)
t

# train의 문장별로 말뭉치(all_words)에 포함되어있는지 T/F로 나타내기
t = [({i: (i in word_tokenize(x[0])) for i in all_words}, x[1]) for x in train]
t

#>>> outcome 처음을 보면, i like you가 말뭉치 단어들에 있는지 기록됨 
# 이를 이용해서 NaiveBayes분류기 동작시키기 

classifier = nltk.NaiveBayesClassifier.train(t)
classifier.show_most_informative_features()

# t에 있는 긍정/부정 태그를 이용해서 분류한 결과
# hate:False일 때, 문장이 긍정일 비율이 1.7:1
# like는 총 3번 사용되었고, 2개가 긍정 -> like:True 일때 문장이 긍정일 비율은 1.7:1

# test_sentence로 분류기 통과해보기
test_sentence = 'i like MeRui'
test_sent_features = {
    i.lower(): (i in word_tokenize(test_sentence.lower()))
    for i in all_words
}
test_sent_features

classifier.classify(test_sent_features)  

t_s='you don\'t like me and you hate me'
t_s_f ={i.lower():
           (i in word_tokenize(t_s.lower()))
           for i in all_words}
t_s_f

classifier.classify(t_s_f)

## 8-6. Naive Bayes Classifier의 이해 - 한글

from konlpy.tag import Okt

pos_tagger = Okt()

train = [('메리가 좋아', 'pos'), 
         ('고양이도 좋아', 'pos'),
         ('난 수업이 지루해', 'neg'),
         ('메리는 이쁜 고양이야', 'pos'),
         ('난 마치고 메리랑 놀거야', 'pos')]
train

# 말뭉치 만들기 - 모든 단어로 쪼개기
# 조사별로도 다 나눔 -> 메리가, 메리는,메리랑
all_words = set(i.lower() for sentence in train
                for i in word_tokenize(sentence[0]))
all_words

t = [({i: (i in word_tokenize(x[0])) for i in all_words}, x[1]) for x in train]
t

classifier = nltk.NaiveBayesClassifier.train(t)
classifier.show_most_informative_features()

# 분류기 테스트 
test_sentence = '난 수업이 마치면 메리랑 놀거야'

test_sent_features = {word.lower():
                          (word in word_tokenize(test_sentence.lower()))
                          for word in all_words}
test_sent_features

classifier.classify(test_sent_features)

#>>> neg 긍정이 나올줄 알았는데 neg가 나옴! (조사가 다 붙어있어서 그런듯)

# 그래서 한글을 다룰때는 형태소 분석이 필요!
# 'Lucy Park'님의 코드 -> 태그 붙여주기 
def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

# 위의 tokenize 함수를 이용해서 train 분석
train_docs = [(tokenize(row[0]), row[1]) for row in train]
train_docs

# 함수를 사용해서 만든 새로운 말뭉치!
tokens = [t for d in train_docs for t in d[0]]
tokens

# 말뭉치에 있는 단어가 있는지 아닌지를 구분하는 함수 만들고, train에 적용
# tokenize함수에 /형태 구분이 되어있어서 판독하기 편함
def term_exists(doc):
    return {word: (word in set(doc)) for word in tokens}

train_xy = [(term_exists(d), c) for d,c in train_docs]
train_xy

classifier = nltk.NaiveBayesClassifier.train(train_xy)
classifier.show_most_informative_features()

test_sentence = [("난 수업이 마치면 메리랑 놀거야")]

# 테스트 문장의 형태소 분석을 먼저하고
test_docs = pos_tagger.pos(test_sentence[0])
test_docs

test_sent_features = {word: (word in tokens) for word in test_docs}
test_sent_features

classifier.classify(test_sent_features) #>>> 의도한대로 pos 나옴 

## 8-7. 문장의 유사도 측정
### 많은 문장/문서에서 유사한 문장을 찾아내는 방법 
### -> 어떤 문장을 벡터로 표현할 수 있다면, 벡터간의 거리를 구하는 방법으로 해결
'''
    * one hot encoding
    * 텍스트/범주형 데이터 -> 수치형 데이터 (벡터에 해당되는 하나의 데이터만 1, 나머지는 0으로 변경)
    * 단어들끼리 유사도 계산불가
* embedding :  Word2Vec
    * 단어들의 유사도 계산 
    * encoding) king =[1,0,0,0], man =[0,1,0,0] 
    * embedding) king=[1,2], man=[1,3]
'''
# 0. 기본 
# sklearn에서 텍스트의 특징(feature)을 추출하는 모듈에서 countvectorizer를 import
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df = 1) # 0과1사이의 실수 (min_df=1이 default)
vectorizer

# 연습용 문장 
contents = ['메리랑 놀러가고 싶지만 바쁜데 어떻하죠?',
            '메리는 공원에서 산책하고 노는 것을 싫어해요',
            '메리는 공원에서 노는 것도 싫어해요. 이상해요.',
            '먼 곳으로 여행을 떠나고 싶은데 너무 바빠서 그러질 못하고 있어요']

# 단어들을 feature로 잡고 벡터로 변환 vectorizer 사용 
X = vectorizer.fit_transform(contents)
vectorizer.get_feature_names()

# one hot encoding 문장별로 나옴! 
X.toarray()

# X.toarray().transpose() <- vectorizer.get_feature_names()와 같은 순서대로 나옴 
X.toarray().transpose()

# (1,3)에 1 == '것도'
# (8,1)의 1 == '놀러가고'

num_samples, num_features =X.shape
num_samples, num_features

new_post =['메리랑 공원에서 산책하고 놀고 싶어요']
new_post_vec =vectorizer.transform(new_post)
new_post_vec.toarray()

import scipy as sp

def dist_raw(v1,v2):
    delta = v1-v2
    return sp.linalg.norm(delta.toarray())   

# sp의 선형대수의 norm
# norm : 벡터간의 거리를 구해서 절대값으로 표현해줌 
# L1 norm: 절대값을 씌우고 다 더해줌(길이나 사이즈 측정가능) 
# L2 norm :평균이나 주변값들과의 거리(나중에 배움)

best_doc = None
best_dist =65535
best_i = None

for i in range(0, num_samples) :
    post_vec =X.getrow(i)
    d = dist_raw(post_vec, new_post_vec)
    
    
    print("== Post %i with dist=%.2f   : %s" %(i,d,contents[i]))
    
    if d<best_dist:
        best_dist = d
        best_i = i

print('best post is %i, dist = %.3f ' % (best_i, best_dist))
print('-->', new_post)
print('--->', contents[best_i])

### 조금더 합리적인 한글 문장 형태소 분석 및 벡터화 


# 1. Okt()으로 형태소 분석, 벡터화 

from konlpy.tag import Okt
t = Okt()

# Okt()이용해서 형태소 분석한 결과 -> contents_tokens
contents_tokens = [t.morphs(row) for row in contents]
contents_tokens

# >>> 메리는 메리끼리, 더 디테일하게 쪼갬

# vectorize하기 쉽게 띄어쓰기로 형태소를 구분하고, 하나의 문장으로 만들기
contents_for_vectorize = []

for content in contents_tokens:
    sentence = ''
    for word in content:
        sentence = sentence + ' ' + word
        
    contents_for_vectorize.append(sentence)
    
contents_for_vectorize

# vectorizer 사용하고 feature(특성)찾기
# -> 디테일하게 형태소 분석을 하고 사용하면 랑,는..은 사라짐
X = vectorizer.fit_transform(contents_for_vectorize)
num_samples, num_features = X.shape
num_samples, num_features

# feature들 확인 
vectorizer.get_feature_names()

# 벡터화
X.toarray().transpose()

# 새로운 문장을 위와 동일한 작업(형태소로 자르고 띄어쓰기 구분하고 한문장으로 만든다음) 
new_post0 = ['메리랑 공원에서 산책하고 놀고 싶어요']
new_post_tokens0= [t.morphs(row) for row in  new_post0]

new_post_for_vectorize0 =[]

for content in new_post_tokens0:
    sentence =''
    for word in content:
        sentence=sentence+ ' '+word
    
    new_post_for_vectorize0.append(sentence)

    
new_post_for_vectorize0

# new_post_for_vectorize를 벡터화 하기 -> new_post_vec
new_post_vec0 = vectorizer.transform(new_post_for_vectorize0)
new_post_vec0.toarray()

# 새로운 문장new_post_vec을 비교해야할 문장(contents)들과 각각 거리를 구하면 됨
# post_vec이 기존 contents
import scipy as sp

# 벡터의 차를 구하는 함수
def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

# 벡터의 차를 정규화 norm하는 함수
best_doc = None
best_dist = 65535
best_i = None

for i in range(0, num_samples):
    post_vec = X.getrow(i)
    d = dist_raw(post_vec, new_post_vec0)
    
    print("== Post %i with dist=%.2f   : %s" %(i,d,contents[i]))
    
    if d<best_dist:
        best_dist = d
        best_i = i

#>>> == Post 1 with dist=1.00   : 메리는 공원에서 산책하고 노는 것을 싫어해요
# 문장의 의미는 반대지만 가장 벡터의 거리가 가까운문장(유사단어 많)

# 위에 content와 새로운 문장을 형태소분석+벡터화
for i in range(0,len(contents)):
    print(X.getrow(i).toarray())
    
print('-------------------')
print(new_post_vec.toarray())

# 이제 거리구하기 
# 함수 생성 (각 벡터를 norm으로 나눠준 후, 벡터의 거리를 구하는 함수 )
def dist_norm(v1,v2):
    v1_normalized = v1 /sp.linalg.norm(v1.toarray())
    v2_normalized = v2 /sp.linalg.norm(v2.toarray())
    
    delta =v1_normalized - v2_normalized
    
    return sp.linalg.norm(delta.toarray())
    

best_doc = None
best_dist =65535
best_i = None

for i in range(0, num_samples ) :  #num_samples : 샘플의 갯수 = len(contents)와 동일 
    post_vec = X.getrow(i)
    d = dist_norm(post_vec, new_post_vec0)
    
    
    print('== post %i with dist = %.3f : %s' %(i,d,contents[i]))
    if d < best_dist:
        best_dist =d
        best_i =i


print('best post is %i, dist = %.3f ' % (best_i, best_dist))
print('-->', new_post0)
print('--->', contents[best_i])

# 2. tfidf(=scikit의 TfidfVectorizer) 이용, 가중치부과하고 벡터화 

# tfidf : 텍스트 마이닝에서 사용하는, 단어별 가중치부과 
# tf : term frequency   - 자주나타나는 단어는 중요도가 높다고 판단
# idf : inverse document frequency - 모든문서에 같은 단어가 있다면 핵심어휘일지는 몰라도 문서간 비교에서는 중요한 단어가 아니라고 봄 
# 이 원리로 tfidf함수 만들기

def tfidf(t,d,D) :
    tf = float(d.count(t)) / sum(d.count(w) for w in set(d))
    idf = sp.log( float(len(D)) / (len([doc for doc in D if t in doc])) )
    return tf, idf

a, abb, abc = ['a'], ['a','b','b'],['a','b','c']
D = [a,abb,abc]

print(tfidf('a',a,D))   # 모든문장에 a가 있으니 idf =0.0
print(tfidf('b',abb,D))
print(tfidf('a',abc,D))
print(tfidf('b',abc,D))
print(tfidf('c',abc,D))


# 위에서 만든 tfidf 함수 -> return tf*idf 로 수정해서 사용하면 되지만
# 여기서는 scikit-learn의 TfidfVectorizer를 import 해서 사용
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer =TfidfVectorizer(min_df=1, decode_error='ignore')

# sklearn에서 텍스트의 특징(feature)을 추출하는 모듈에서 countvectorizer를 import
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df = 1)

contents_tokens=[t.morphs(row) for row in contents]

contents_for_vectorize = []

for content in contents_tokens:
    sentence =''
    for word in content:
        sentence = sentence+ ' ' + word
        
    contents_for_vectorize.append(sentence)


X =vectorizer.fit_transform(contents_for_vectorize)
num_samples, num_features = X.shape
num_samples, num_features

# contents 문장들 다듬기 
vectorizer.get_feature_names()

# 테스트문장 비교 
new_post1 = ['근처 공원에 메리랑 놀러가고 싶네요']  #new_post랑 다른문장
new_post_tokens1= [t.morphs(row) for row in  new_post1]

new_post_for_vectorize1 =[]

for content in new_post_tokens1:
    sentence =''
    for word in content:
        sentence=sentence+ ' '+word
    
    new_post_for_vectorize1.append(sentence)

    
new_post_for_vectorize1

new_post_vec1= vectorizer.transform(new_post_for_vectorize1)
new_post_vec1

# 다른 결과와 비교해보기 위해 실행 
best_doc = None
best_dist =65535
best_i = None

for i in range(0, num_samples ) :  #num_samples : 샘플의 갯수 = len(contents)와 동일 
    post_vec = X.getrow(i)
    d = dist_norm(post_vec, new_post_vec1)
    
    
    print('== post %i with dist = %.3f : %s' %(i,d,contents[i]))
    if d < best_dist:
        best_dist =d
        best_i =i


print('best post is %i, dist = %.3f ' % (best_i, best_dist))
print('-->', new_post1)
print('--->', contents[best_i])



## 8-8. 여자 친구 선물 고르기
* word2vec 사용

import pandas as pd
import numpy as np

import platform
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

path = "c:/Windows/Fonts/malgun.ttf"
from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')    

plt.rcParams['axes.unicode_minus'] = False

from bs4 import BeautifulSoup 
from urllib.request import urlopen
import urllib
import time

tmp1 = 'https://search.naver.com/search.naver?where=kin'
html = tmp1 + '&sm=tab_jum&ie=utf8&query={key_word}&start={num}'

response = urlopen(html.format(num=1, key_word=urllib.parse.quote('여친 선물')))

soup = BeautifulSoup(response, "html.parser")

tmp = soup.find_all('dl')

tmp_list = []
for line in tmp:
    tmp_list.append(line.text)
    
tmp_list

from tqdm import tqdm_notebook

present_candi_text = []

for n in tqdm_notebook(range(1, 1000, 10)):
    response = urlopen(html.format(num=n, key_word=urllib.parse.quote('여자 친구 선물')))

    soup = BeautifulSoup(response, "html.parser")

    tmp = soup.find_all('dl')

    for line in tmp:
        present_candi_text.append(line.text)
        
    time.sleep(0.5)

present_candi_text

len(present_candi_text)

import nltk
from konlpy.tag import Okt 
t = Okt()

present_text = ''

for each_line in present_candi_text[:10000]:
    present_text = present_text + each_line + '\n'

tokens_ko = t.morphs(present_text)
tokens_ko

ko = nltk.Text(tokens_ko, name='여자 친구 선물')
print(len(ko.tokens))
print(len(set(ko.tokens)))

ko = nltk.Text(tokens_ko, name='여자 친구 선물')
ko.vocab().most_common(100)

ko.similar('여자친구')

stop_words = ['.','가','요','답변','...','을','수','에','질문','제','를','이','도',
                      '좋','1','는','로','으로','2','것','은','다',',','니다','대','들',
                      '2017','들','데','..','의','때','겠','고','게','네요','한','일','할',
                      '10','?','하는','06','주','려고','인데','거','좀','는데','~','ㅎㅎ',
                      '하나','이상','20','뭐','까','있는','잘','습니다','다면','했','주려',
                      '지','있','못','후','중','줄','6','과','어떤','기본','!!',
                      '단어','선물해','라고','중요한','합','가요','....','보이','네','무지']

tokens_ko = [each_word for each_word in tokens_ko 
                                                         if each_word not in stop_words]

ko = nltk.Text(tokens_ko, name='여자 친구 선물')
ko.vocab().most_common(50)

plt.figure(figsize=(15,6))
ko.plot(50) 
plt.show()

from wordcloud import WordCloud, STOPWORDS
from PIL import Image

data = ko.vocab().most_common(300)


wordcloud = WordCloud(font_path='c:/Windows/Fonts/malgun.ttf',
                      relative_scaling = 0.2,
                      #stopwords=STOPWORDS,
                      background_color='white',
                      ).generate_from_frequencies(dict(data))
plt.figure(figsize=(16,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

mask = np.array(Image.open('C:/Users/serah/ds/data/09. heart.jpg'))

from wordcloud import ImageColorGenerator

image_colors = ImageColorGenerator(mask)

data = ko.vocab().most_common(200)


wordcloud = WordCloud(font_path='c:/Windows/Fonts/malgun.ttf',
               relative_scaling = 0.1, mask=mask,
               background_color = 'white',
               min_font_size=1,
               max_font_size=100).generate_from_frequencies(dict(data))

default_colors = wordcloud.to_array()

plt.figure(figsize=(12,12))
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
plt.axis('off')
plt.show()

* gensim install : **pip install gensim**

import gensim
from gensim.models import word2vec

t = Okt()
results = []
lines = present_candi_text

for line in lines:
    malist = t.pos(line, norm=True, stem=True)
    r= []
    
    for word in malist:
        if not word[1] in ["Josa", "Eomi", "Punctuation"]:
            r.append(word[0])
            
    r1 = (" ".join(r)).strip()
    results.append(r1)
    print(r1)

data_file = 'pres_girl.data'
with open(data_file, 'w', encoding='utf-8') as fp:
    fp.write("\n".join(results))

data = word2vec.LineSentence(data_file)
model = word2vec.Word2Vec(data, size=200, window=10, hs=1, 
                                                                        min_count=2, sg=1)
model.save('pres_girl.model')

model = word2vec.Word2Vec.load("pres_girl.model")

model.most_similar(positive=['선물'])

model.most_similar(positive=['여자친구'])

model.most_similar(positive=['디바이스'])
