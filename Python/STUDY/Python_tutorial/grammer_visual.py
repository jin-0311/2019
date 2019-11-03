# 파이썬 문법, 시각화 정리


#%% <<<ch1 데이터 전처리>>>

#%% ch 1-1) 데이터 불러오기, 저장하기
#  불러오기  (csv/tsv, excel, pickle)
import pandas as pd 

df =pd.read_csv('C:/파일.확장자(csv,tsv가능), index_col = , sep='\t',engine ='python', encoding = 'utf-8' or 'euc-kr')

df = pd.read_excel('파일.확장자', parse_cols ='원하는 컬럼들', header = , encoding ='')

df = pd.read_pickle('파일.확장자')
 

# 저장하기 - 피클, csv/tsv
df.to_pickle('C:/df.pickle')
df.to_csv('파일.csv', index =T/F)
df.to_csv('파일.tsv, sep='\t')   # sep필요 -> 컬럼별로 띄어쓰기 자동으로 

# 저장하기 - 엑셀 
# 시리즈는 df 변환 : 시리즈.to_frame() 
import xlwt
df.to_excel('파일.xls')
import openpyxl
df.to_excel('파일.xlsx')



#%% ch 1-2) 데이터 특징 확인 
import pandas as pd
df = pd.read_csv('C:/Users/serah/ds/data/gapminder.tsv', sep='\t')   
df
print(df.head())     #  default : 5줄만 조회 
df.head(10) 

print(type(df))      #>>>  <class 'pandas.core.frame.DataFrame'>
print(df.shape)      # 목록의 형태(크기)확인
print(df.columns)    # 가져온 데이터 목록의 열 정보 확인
print(df.dtypes)     # 변수명, 변수타입 
print(df.info())     # 가져온 목록의 전체정보 확인
print(df.describe()) # 기초통계량  통계적 개요 
print(df.index)

#%% ch1-3) 데이터 만들기 - 시리즈, 데이터프레임(딕셔너리 기반) 
#%% 시리즈 Series = 데이터 프레임의 컬럼은 모두 시리즈 
# 인덱스는 자동으로 0,1,..으로 생성
import pandas as pd
s= pd.Series(['apple',33])   
print(s)

s1= pd.Series(['jane','student'])
s1

# 시리즈 생성시 문자열을 인덱스로 지정
s2= pd.Series(['jane','student'], index=['person','job'])
print(s2)


#%% 데이터 프레임 DataFrame
# 딕셔너리(파이썬 기본 자료구조)로 생성하기 {key:value} / unorderd data
import pandas as pd
sc= pd.DataFrame({
    'name' :['rosa franklin', 'will gosset'],
    'occupation':['chemist', 'statistician'],
    'born':['1920-07-25', '1876-06-13'],
    'died':['1958-04-16','1937-10-16'],
    'age':[37,61]})
sc

# index, columns(시리즈 순서) 지정
sc1= pd.DataFrame(
    data={'occupation':['chemist', 'statistician'],
          'born':['1920-07-25', '1876-06-13'],
          'died':['1958-04-16','1937-10-16'],
          'age':[37,61]},
    index=['rosa franklin', 'will gosset'],
    columns=['occupation','born','age','died']
)
sc1

# 순서가 보장된 딕셔너리 전달하기 - OrderedDict 클래스 사용
# 괄호랑 , 주의해서 작성 
from collections import OrderedDict
sc2 = pd.DataFrame(OrderedDict([
    ('name' ,['rosa franklin', 'will gosset']),
    ('occupation',['chemist', 'statistician']),
    ('born',['1920-07-25', '1876-06-13']),
    ('died',['1958-04-16','1937-10-16']),
    ('age',[37,61])
])
)
sc2

sc1= pd.DataFrame(
    data={'occupation':['chemist', 'statistician'],
          'born':['1920-07-25', '1876-06-13'],
          'died':['1958-04-16','1937-10-16'],
          'age':[37,61]},
    index=['rosa franklin', 'will gosset'],
    columns=['occupation','born','age','died']
)
print(sc1)

#%% 시리즈, 데이터프레임 활용 

# 1. 필요한 시리즈 선택하기 
first_row = sc1.loc['will gosset']
type(first_row)     # >>> pandas.core.series.Series
first_row


# 2. index, value속성과 keys method 사용하기 

# index 속성 사용하기
print(first_row.index)  #>>>Index(['occupation', 'born', 'age', 'died'], dtype='object')

# value 속성 사용하기 
print(first_row.values) #>>> ['statistician' '1876-06-13' 61 '1937-10-16']

# keys method == 인덱스 속성
print(first_row.keys()) #>>> Index(['occupation', 'born', 'age', 'died'], dtype='object')


ages=sc0['Age']
print(ages.max())
print(ages.min())
print(ages.mean())
print(ages.std())

# index 속성 응용
print(first_row.index[0]) #>>>occupation

# 시리즈의 [mean,min,max,std] method 사용하기
ages= sc1['age']
print(ages)
print(ages.mean())
print(ages.min())
print(ages.max())
print(ages.std())

sc0 = pd.read_csv('C:/Users/serah/ds/data/scientists.csv')
sc0.head()

# 불린 추출 
print(ages[ages>ages.mean()])  # 평균보다 큰 '나이'를 출력
print(ages>ages.mean())        # 평균보다 나이가 크면 '(T)'출력 -> bool 자료형 

bool_value = [True,True,False,False,True,True,False,True]
print(ages[bool_value]) #bool_value를 ages에 전달하여 T만 출력 


# 3. 시리즈와 데이터 프레임의 데이터 처리하기

print(sc0)
print(sc0['Born'].dtype)
print(sc0['Died'].dtype)
# 둘다 object(문자열)

# 날짜 계산을 위해 'Born','Died'를 datetime형태로 데이터 타입 변경 
born_datetime=pd.to_datetime(sc0['Born'],format = '%Y-%m-%d')
print(born_datetime)
died_datetime=pd.to_datetime(sc0['Died'], format= '%Y-%m-%d')
print(died_datetime)
#sc0에 위의 2개 변수 추가 
sc0['born_dt'],sc0['died_dt'] = (born_datetime,died_datetime)
sc0.head()
print(sc0.shape) #8행으로 바뀜 

# died-born 구하기 
sc0['age_days_dt'] = (sc0['died_dt']-sc0['born_dt'])
sc0

print(sc0['Age'])
print(sc0['age_days_dt'])

# 4. 시리즈, 데이터프레임의 데이터 섞기

import random              # 난수발생 
random.seed(42)            # seed 42로 고정 
random.shuffle(sc0['Age'])  #왜 섞어? 정확도를 위해 다양한 데이터로 learning하기위해 
print(sc0['Age'])

# 5. 데이터 프레임의 열 삭제하기
print(sc0.columns)
sc0_dropped = sc0.drop(['Age'], axis=1)  #axis0 = 열, axis=1 행  -확인하기 
print(sc0_dropped.columns)


# 6. 실습
import pandas as pd
import numpy as np

s= pd.Series([1,3,5,np.nan,6,8])
s
dates= pd.date_range('20130101', periods = 6)  # date_range -> 판다스의 날짜형 데이터, periods =6일간 
dates

df = pd.DataFrame(np.random.randn(6,4),    # 6행4열의 난수 생성
                 index = dates,            # 인덱스는 dates로 
                 columns = ['A','B','C','D'])
print(df)

df.sort_values(by = 'B', ascending = False) 
df

#%% ch 1-4) 데이터 추출 및 정제

import pandas as pd
import numpy as np
dates= pd.date_range('20130101', periods = 6)
dates
df = pd.DataFrame(np.random.randn(6,4),    # 6행4열의 난수 생성
                 index = dates,            # 인덱스는 dates로 
                 columns = ['A','B','C','D'])
print(df)
df.sort_values(by = 'B', ascending = False) 
df
#%% 데이터 추출(loc/iloc, 중복데이터, 컬럼추가 rename, value변경)
# - https://3months.tistory.com/292 참조 
# 1) 슬라이싱 기법 

# 필요한 행/열 추출하기 df['col'] or df['row'] : 이름이나 인덱스로 가능 
print(df['A'])
print(df[['A','B']])   # 2개 이상 가져올땐, [[]] 두개로 묶어주기  
print(df[0:3])
print(df['20130102':'20130104'])

# loc(location) ->  :사용하면 전체, [:] 범위 - 문자열 리스트로불러오기 가능, 숫자 가능 

print(df.loc[dates[0]])
print(df.loc[:,['A','B']])
print(df.loc['20130102' : '20130104', ['A','B']])
print(df.loc['20130102', ['A','B']])
print(df.loc[dates[0], 'A'])

      
# iloc(index location)  번호만을 이용해서 데이터에 바로 접근  - 음수 가능 
print(df.iloc[3])   #(0부터세고)3행 추출 
print(df.iloc[:,3]) # 3열 추출 
print(df.iloc[-1])  #맨 마지막 행 추출
print(df.iloc[:,-1]) 


print(df.iloc[3:5, 0:2])

print(df.iloc[[1,2,4],0:2]) # 열에 순서로 쓸꺼면 []없이
print(df.iloc[[1,2,4],[0,2]])  # 열에 필요한것만 뽑을거면 [,]

print(df.iloc[1:3,:])
print(df.iloc[:, 1:3])


# loc[숫자] 사용
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(6,4),    # 6행4열의 난수 생성
                 columns = ['A','B','C','D'])
print(df)
df.sort_values(by = 'B', ascending = False) 
df
df.loc[0] = 10+ df.loc[1]
df


# shape활용 마지막행 구하기 
df.shape # >>> (6,4) 로 (m x n) 으로 행렬 알려줌
num_rows=df.shape[0]
last_row=num_rows -1
print(last_row)
print(df.iloc[last_row]) 



# 2) range method - 범위를 미리지정, 
small_range =list(range(2))
small_range1=list(range(1,2))
print(small_range)
print(small_range1)

df.iloc[:,small_range]
print(subset.head())


# 3) 조건에 맞는 데이터만 추출 (결과에 맞지 않으면 NAN)
df[df.A>0]
df[df > 0]
df>0 #>>> TorF 불린 형태만 나옴 



#%% 기타 데이터 처리 

# 중복데이터 확인 및 처리하기
#  -> https://rfriend.tistory.com/266?category=675917
#  옵션: keep ='first'/'last'/False
df.duplicated(['A'])  #>>> 불형으로 알려줌 


#  행 삭제 drop, 열 삭제 del
a= pd.DataFrame({'A': [1,2,3,4],
                 'B': [5,6,7,8]})
a
a.drop([1], inplace = True)
a

del df['A'] ; df  #>>> 자동으로 df에 적용됨 
df


# df2= df.copy()를 사용:  위치와 내용도 복사 - > 원본데이터 따로 존재 
# df2 =df 만 사용하면 데이터 위치만 복사, 
df2 = df.copy()


# 새로운 컬럼 추가 df['컬럼이름'] = [values] / value값 있는지 물어보기
df2['E'] = ['one','one','two','three','four','three']
df2

df2['E'].isin(['two','four'])   #isin 있니? - T or F 로 반환 


# rename  변수명 변경 
import pandas as pd
a = pd.DataFrame({
    'name' :['apple','banana','cap','drug'],
    'price':[500,600,700,200],
    'prefer' :[1,4,2,3]
})
a

a.rename(columns ={a.columns[0]:'이름',
                   a.columns[1]:'가격',
                   a.columns[2]:'선호도'},
                   inplace = True)
a

# value 의 값 변경시 df.loc로 사용하기
# df.loc[df['col'] =='old value','col'] ='new value'
a.loc[a['이름']=='drug','이름'] ='drum'
a


#%%  벡터와 스칼라로 브로드캐스팅 (시리즈, 데이터 프레임을 한 번에 계산)
# 벡터: 시리즈처럼 여러개의 값을 가진 데이터
# 스칼라: 단순크기를 나타내는 데이터
import pandas as pd
data = pd.read_csv('C:/Users/serah/ds/data/scientists.csv')
data

ages = data['Age']
ages

# 벡터와 벡터 
print(ages+ages)
print(ages*ages)

# 벡터와 스칼라 (를 합치는게 브로드캐스팅 )
print(ages + 100)
print(ages*2)

a = pd.Series([1,100])   #a는 2개의 value
print(a)
print(ages+a)            #ages는 8개의 value 있음  나머지(인덱스가 2~7인 값)는 NaN 

rev_ages = ages.sort_index(ascending = False)
print(rev_ages)
print(ages)
print(ages+rev_ages)   # 벡터와 벡터의 연산은 일치하는 인덱스 값끼리 수행 

# 불린 추출과 브로드캐스팅
print(data['Age'].mean())
print(data[data['Age']> data['Age'].mean()])  # print()안하면 output표처럼 나옴
print(data.loc[[True,True,False,True]])     #loc:index값 찾아가기 - [2]가 F로 되어있으니 0,1,3만 나옴

# 데이터 프레임에 스칼라 연산을 하면  ->  숫자는 연산 / 문자열은 반복
data*2    #>>> ... 74 ChemistChemist 



#%%  .apply 함수 적용하기 - 한번에 함수 적용 

# 함수 적용하기 df.apply(함수이름, 옵션 )
import pandas as pd
import numpy as np
dates= pd.date_range('20130101', periods = 6)
dates
df = pd.DataFrame(np.random.randn(6,4),    # 6행4열의 난수 생성
                 index = dates,            # 인덱스는 dates로 
                 columns = ['A','B','C','D'])
# np의 cumsum 누적합 구하기 
print(df)
print(df.apply(np.cumsum))             # column(세로) 별로 누적 합계
print(df.apply(np.cumsum, axis =0))    # column(세로) 별로 누적 합계
print(df.apply(np.cumsum, axis =1))    # 행별 누적합계 

# 최대값-최소값의 차이(거리)를 알고 싶으면 one-line함수인 lambda (점프투 참조)
df.apply(lambda x : x.max()- x.min())


#%% ch 1-5) 데이터 concat/merge

#%% Data 연결하기 concat

# 1) 기초 
import pandas as pd
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])
print(df1)
print(df2)
print(df3)

# 데이터프레임에 시리즈 연결하기
new_row_series = pd.Series(['n1', 'n2', 'n3', 'n4'])
print(pd.concat([df1, new_row_series]))

# 행 1개로 구성된 데이터프레임 생성하여 연결하기
new_row_df = pd.DataFrame([['n1', 'n2', 'n3', 'n4']], columns=['A', 'B', 'C', 'D']) 
print(new_row_df)

# 한번에 2개이상 df연결 가능 
print(pd.concat([df1, new_row_df])) 

#연결할 데이터 프레임이 1개라면 append 메서드를 사용
print(df1.append(new_row_df)) 

#데이터 프레임의 인덱스를 0부터 다시 지정함
data_dict = {'A': 'n1', 'B': 'n2', 'C': 'n3', 'D': 'n4'}
print(df1.append(data_dict, ignore_index=True))   




# 2)  pd.concat([df1,df2]) : 단순하게 밑으로 합치기

# 기본
result = pd.concat([df1,df2,df3])   # default axis = 0, (열방향,세로에) 밑으로 붙는다. 
print(result)
result0 = pd.concat([df1, df2, df3], axis = 1)  # axis = 1 , (행방향,가로에) 옆으로 붙는다. -> -> 한개라도 겹치는 값이 없으면 nan 
print(result0)

# 다중 index 지정 -> level 형성
result = pd.concat([df1, df2, df3], keys=['x', 'y', 'z'])   # default)axis=0,(열방향,세로)밑으로  
print(result)
print(result.index)      # 인덱스가 multi(여러개) - levels, labels로
print(result.index.get_level_values(0))  # 그중 levels만
print(result.index.get_level_values(1))  # labels 만 

# ignore_index 인자 사용 - 열 이름을 다시 지정
ignore_index= pd.concat([df1, df2, df3], ignore_index=True) 
print(ignore_index)



# 실습 
df4 = pd.DataFrame({'B': ['B2', 'B3', 'B6', 'B7'], 
                    'D': ['D2', 'D3', 'D6', 'D7'],
                    'F': ['F2', 'F3', 'F6', 'F7']},
                   index=[2, 3, 6, 7])

print(df1)
print(df4)
result = pd.concat([df1, df4], axis=1)  # axis =1 옆으로 그냥붙임 
print(result)

print(df1)  # 인덱스:0123 , ABCD
print(df4)  # 인덱스:2367, BDF


#inner/outer join을 사용한 concat

# 3-1) inner join : 양쪽에 겹치는 값들만 가져옴 (인덱스 2,3만! 열 모두)
result1 = pd.concat([df1, df4], axis = 1, join = 'inner')
print(result1)

# join에 index지정, df1의 인덱스를 그대로 가져옴 (df4엔 0,1없으니 -> nan/ df4의 6,7,인덱스는 사라짐) 
result2 = pd.concat([df1, df4], axis = 1, join_axes=[df1.index])
print(result2)

# 3-2) outer join 
result3 = pd.concat([df1, df4], axis = 1, join= 'outer')  # axis =1 , 옆으로 그대로 다붙임 
print(result3)
result4 = pd.concat([df1,df4], ignore_index = True)  
print(result4)

# result4 : axis =0(default)-> 아래로 합쳐짐  
# 인덱스 무시 -> 순차적으로 0~7까지 생성(=range index붙게됨)
# 같은 칼럼은 같이, 없는 칼럼 생성(F) 





#%%  데이터 연결하기 – merge
# merge 기본적으로 내부 조인을 실행

# 메서드를 사용한 데이터 프레임(site)을 왼쪽으로 지정,
# 첫 번째 인자값으로 지정한 데이터프레임(visited_subset)을 오른쪽에 지정
# 열이름 일치-왼쪽 프레임을 기준으로 연결

left = pd.DataFrame({'key': ['K0', 'K4', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
print(left)
print(right)

# left에는 key가 0234, right에는 0123 

#  1) 단순 merge - 겹치는 key 기준  (아래의  how ='inner'과 동일) 
print(pd.merge(left, right, on = 'key'))  # 겹치는 키만(023)

#  2) left/right merge - left/right 키값을 기준으로 합쳐! 
print(pd.merge(left, right, how = 'left', on = 'key'))  #왼쪽키 0234 만
print(pd.merge(left, right, how= 'right', on='key'))    #오른쪽키 0123  만 

#  3) inner/outer merge 
print(pd.merge(left, right, how='outer', on='key'))  # 전체다 (없으면 nan) - 합집합 
print(pd.merge(left, right, how='inner', on='key'))  # 겹치는 것만! (inner가 defualt) - 교집합


# 다른 데이터로 merge
import pandas as pd

person = pd.read_csv('C:/Users/serah/ds/data/survey_person.csv')  
site = pd.read_csv('C:/Users/serah/ds/data/survey_site.csv') 
survey = pd.read_csv('C:/Users/serah/ds/data/survey_survey.csv') 
visited = pd.read_csv('C:/Users/serah/ds/data/survey_visited.csv')

print(person)
print(site)
print(survey)
print(visited)

visited_subset = visited.loc[[0, 2, 6], ]
print(visited_subset)
print(site)

o2o_merge = site.merge(visited_subset, 
                       left_on='name', right_on='site') 
print(o2o_merge)

m2o_merge = site.merge(visited, 
                       left_on='name', right_on='site') 
print(site)
print(visited)
print(m2o_merge)

ps = person.merge(survey, 
                  left_on='ident', right_on='person') 
vs = visited.merge(survey, 
                   left_on='ident', right_on='taken')

print(person)
print(survey)
print(ps)
print(vs)



# ps 데이터 프레임의 ident, taken, quant, reading 열의 값과 
# vs 데이터프레임의 person, ident, quant, reading 열의 값을 이용하여 ps와 vs 데이터 프레임을 연결

ps_vs = ps.merge(vs, 
                 left_on=['ident', 'taken', 'quant', 'reading'], 
                 right_on=['person', 'ident', 'quant', 'reading'])
print(ps_vs.head())

 

#%% ch 1-5) 데이터의 누락값 처리
# NAN 누락값  NAN=Not A Number =NAN, NaN, nan
# 누락값 사용하려면 numpy 필요(수학/과학연산을 위한 library)

from numpy import NaN,NAN,nan

# 누락값은 비교할 자체가 없어서 다 F
print(NaN == True)
print(NaN == 0)
print(NaN == '')

import pandas as pd
print(pd.isnull(NaN))
print(pd.isnull(NAN))
print(pd.isnull(nan))

print(pd.notnull(NaN))
print(pd.notnull(NAN))
print(pd.notnull(nan))

print(pd.notnull(42))
print(pd.notnull('missing'))

#%% 누락값이 생기는 이유 

# 1)na가 있는 데이터집합을 연결할때 
import pandas as pd

visited = pd.read_csv('C:/Users/serah/ds/data/survey_visited.csv')
survey = pd.read_csv('C:/Users/serah/ds/data/survey_survey.csv') 

print(visited)
print(survey)

vs = visited.merge(survey, 
                   left_on = 'ident',
                   right_on = 'taken'
                  )
vs

# 2) 데이터를 입력할때 누락값이 생기는 경우 
num_legs = pd.Series({'goat': 4, 'amoeba':nan})
print(num_legs)
print(type(num_legs))


sc = pd.DataFrame({
    'name' :['rosa', 'will'],
    'occupation' :['chemist', 'statistician'],
    'born':['1920-07-25', '1876-06-13'],
    'died' :['1958-04-16','1937-10-16'],
    'missing' :[NaN,nan],
})
print(sc)
print(type(sc))

# 3) 범위를 지정하여 데이터를 추출할때 누락값이 생기는 경우
gap = pd.read_csv('C:/Users/serah/ds/data/gapminder.tsv', sep ='\t')
gap.head()


life_exp = gap.groupby(['year'])['lifeExp'].mean()
print(life_exp)
print(life_exp.loc[range(2000,2010),])  # 리스트에서는 , 안붙여도 상관 없지만 튜플은 꼭
# df에 존재하지 않는 데이터를 추출시 NAN 발생 

y2000 = life_exp[life_exp.index>2000]   # 불린 추출 -> 누락값 빼고 추출 
print(y2000)

yy2000 = life_exp.index >2000    # T or F 확인 가능 
print(yy2000)

#%%  누락값 처리 

# 0. 누락값 파악 

#누락값 개수 구하기
ebola = pd.read_csv('C:/Users/serah/ds/data/country_timeseries.csv')
print(ebola.count())  # 누락값이 아닌 개수 구하기 
print(ebola.shape)   
# 누락값 개수 구하기
num_rows = ebola.shape[0]
print(num_rows)
num_missing = num_rows - ebola.count()
print(num_missing)
 
#     
import numpy as np
print(np.count_nonzero(ebola.isnull()))
print(np.count_nonzero(ebola['Cases_Guinea'].isnull()))

# value_couns <- 지정한 열의 빈도를 구하는 method
print(ebola.Cases_Guinea.value_counts(dropna=False).head())

# 1. 누락값 변경하기

# 1)fillna메서드에 0을 대입 -> 누락값을 0으로 변경 
print(ebola.fillna(0)).iloc[0:10, 0:5]

#2) -forward(이전값으로 대체)값으로 채워짐 6번 값을 5번값으로 -> 맨앞이 nan이면 그다음도 nan
print(ebola.fillna(method='ffill').iloc[0:10, 0:5]) 

#3) backword(이후 값으로 대체) -6번값을 7번값으로 
print(ebola.fillna(method = 'bfill').iloc[0:10, 0:5])

# 4) 앞뒤의 평균값으로 
ebola.interpolate().iloc[0:10,0:5]

# 데이터 주무르기 
# ebola.fillna(method = 'pad', inplace = T/F)  -> 앞의 내용으로 채우기 

# 2. 누락값 삭제하기
print(ebola.shape) # (행,열)
ebola_dropna = ebola.dropna()
print(ebola_dropna.shape)
ebola_dropna

# 3. 누락값이 포함된 데이터 계산하기 (na 있으면 결과도 na)
ebola['Case_multiple'] = ebola['Cases_Guinea'] + ebola['Cases_Liberia']+ ebola['Cases_SierraLeone']
ebola.head()
e_subset = ebola.loc[:, ['Cases_Guinea', 'Cases_liberia', 'Cases_Sierrsleone','Cases_multiple']]
e_subset.head(10)

# 누락값을 무시하고 계산 skipna = True  -> 열 계산
print(ebola.Cases_Guinea.sum(skipna = True))
print(ebola.Cases_Guinea.sum(skipna = False))

# NAN 무시하고 행별 계산 
ebola['Case_multiple'] = ebola['Cases_Guinea'] + ebola['Cases_Liberia']+ ebola['Cases_SierraLeone']
x = ebola['Case_multiple']
x.dropna()

#%% <<<ch 2 다양한 library() >>>

#%% glob

# fhv_tripdata 실습 : 공통파일명* 붙이면 다 열림
import glob
nyc_taxi = glob.glob('C:/Users/serah/ds/data/fhv_*')
nyc_taxi


t1 = pd.read_csv(nyc_taxi[0])
t2 = pd.read_csv(nyc_taxi[1])
t3 = pd.read_csv(nyc_taxi[2])
t4 = pd.read_csv(nyc_taxi[3])
t5 = pd.read_csv(nyc_taxi[4])

print(t1.head())
print(t2.head())
print(t3.head())
print(t4.head())
print(t5.head())

print(t1.shape)
print(t2.shape)
print(t3.shape)
print(t4.shape)
print(t5.shape)


taxi = pd.concat([t1,t2,t3,t4,t5], join= 'inner')
print(taxi.shape)



list_taxi_df = []  # list 준비
for csv_filename in nyc_taxi :
    df = pd.read_csv(csv_filename)
    list_taxi_df.append(df)
print(len(list_taxi_df))


print(list_taxi_df[0].head())

taxi_loop_concat = pd.concat(list_taxi_df)
print(taxi_loop_concat.shape)





#%% Pivot

# 0. 준비 
import pandas as pd
import numpy as np
df = pd.read_excel('C:/Users/serah/ds/data/02. sales-funnel.xlsx')
df.head()
# gr) pd.pivot_table(df, index =[''], values = [''], aggfunc = )

print(df)
print(df.index)

# 1. index = ['name'] 기준 -> 중복 이름을 하나로 표현해서 결과를 표현(숫자형 데이터는 평균값) 
print(pd.pivot_table(df, index = ['Name'])) 

# 인덱스 여러개 지정가능 
print(pd.pivot_table(df, index = ['Name', 'Rep','Manager']))


# 2. 특정 values =['col'] 지정가능  - 기본적으로 평균값 반환 
print(pd.pivot_table(df, index = ['Manager','Rep'], values = ['Price']))   

# 합계를 표현하려면 aggfunc=[np.sum] 사용 / values만 지정시 평균값
print(pd.pivot_table(df, index = ['Manager','Rep'], values = ['Price'], aggfunc = [np.sum])) 

# 평균과 len:데이터의 개수 구하기 
pd.pivot_table(df, index = ['Manager','Rep'], values = ['Price'], aggfunc = [np.mean, len])

# value가 나오는 부분에 columns = ['Product'] 추가로 각 제품을 표현가능 
pd.pivot_table(df, index = ['Manager','Rep'], values = ['Price'], columns = ['Product'], 
               aggfunc = [np.sum])

# fill_value =0 으로 nan을 0으로 표시 
pd.pivot_table(df, index = ['Manager', 'Rep'], values = ['Price'], columns = ['Product'], 
               aggfunc = [np.sum], 
               fill_value = 0)

# value(보여줄 값)을 2개 이상, column 설정하면 따로 나옴
pd.pivot_table(df, index = ['Manager', 'Rep'], values = ['Price', 'Quantity'], 
               columns = ['Product'], aggfunc = [np.sum], fill_value = 0)

# product를 컬럼으로 지정하지 않고 인덱스로 넣으면  rep별 prod별 sum과 mean 나옴 
# margins = True -> 소수점 다 표현 
pd.pivot_table(df, index = ['Manager','Rep','Product'], values = ['Price', 'Quantity'],
               aggfunc = [np.sum, np.mean], fill_value =0, margins = True)

# margin = False -> 소수점 간결, 합계 안나타냄  / 한개의 컬럼에도 margins적용 가능 
pd.pivot_table(df, index = ['Manager','Rep','Product'], values = ['Price', 'Quantity'],
               aggfunc = [np.sum, np.mean], fill_value =0, margins = False)

# 각각 value에 aggfunc 적용하기 (price처럼 2개의 func 도 가능 )
pd.pivot_table(df, index = ['Manager','Rep'], columns = ['Product'],
              values = ['Quantity','Price'],
              aggfunc ={'Quantity': len, 'Price': [np.sum, np.mean]},
              fill_value = 0)

eg= pd.pivot_table(df, index = ['Manager','Rep'], columns = ['Product'],
              values = ['Quantity','Price'],
              aggfunc ={'Quantity': len, 'Price': [np.sum, np.mean]},
              fill_value = 0)
eg

# pivot_table은 df['큰틀','중간틀','작은틀'] 로 필요한 부분만 가져오기 
eg['Price','mean','CPU']


'''
#%% Beautiful soup - 웹 데이터 가져오기 
* beautiful soup
* html 코드를 파이썬이 이해하는 객체구조로 변환하는 parsing 을 맡고 있음 - > 의미있는 정보 추출 
* html - 페이지 소스보기 
    * prompt) pip install bs4
    * import HTMLPaser 
    
* http://zeroplus1.zc.bz/jh/web/main.php?id=132&category=ETC

# <!DOCTYPE html> : 문서가 HTML5 문서임을 나타냄
# <html> :HTML 웹페이지의 가장 근본적인 성분
# <head> :문서에 대한 메타 정보를 포함  
# <title> :문서의 제목을 나타냄
# <body> : 페이지에서 보여질 컨텐츠를 포함
# <h1> : 큰 제목을 나타냄  <h1>제목</h1>
# <p> : 하나의 문단을 나타냄 <p> 문단 </p>
# <div>  : 부분을 나눌때 사용함
'''
from bs4 import BeautifulSoup
page = open("C:/Users/serah/ds/data/03. test_first.html",'r').read()   # 'r' 읽기모드 
soup = BeautifulSoup(page, 'html.parser')  # 전체 html 코드를 soup에 저장 
print(soup.prettify())   # 전체 다 읽어오기 

list(soup.children)  # soup 한단계 아래의 태그 데려오기! 

html = list(soup.children)[2] # 리스트중 2번째꺼 == <html>  가져오기
html

list(html.children)  # <html>의 한단계 아래 데려오기 
list(html.parent)    # <html>의 한단계 위 데려오기 
body = list(html.children)[3] # <body>부분 가져오기 
body

soup.body

soup.head

list(soup.children)

list(body.children)

soup.find_all('p')   # 필요한 태그 찾기!  모두!

soup.find('p')  # 맨처음 나오는 p태그 찾기 

soup.find_all('p', class_='outer-text')  # p 태그의 class 가 'outer-text'인것만 찾기 

soup.find_all(class_="outer-text")    # class로 찾을 수도 있음 

soup.find_all(id="first")

soup.head

soup.head.next_sibling # soup의 head다음에 \n이 있으니 그거 찾아줌 

soup.head.previous_sibling

soup.head.next_sibling.next_sibling # head 다음다음 

body.p

body.p.next_sibling.next_sibling

 # .get_text() : 태그 안의 텍스트만 출력하는 함수 
for each_tag in soup.find_all('p'):
    print(each_tag.get_text()) 

body.get_text()  #태그가 있던 자리 -> \n 표시하고,  전체 텍스트를 보여줌 

links = soup.find_all('a')  # 'a' -> 클릭가능한 링크를 의미 
links

# href속성 -> 링크 주소 얻기 
for each in links:
    href = each['href'] 
    text = each.string
    print(text + ' -> ' + href)

# 크롬 개발자 도구를 이용해서 원하는 태그 찾기

from urllib.request import urlopen  #  url로 접근하는 경우 import 
# 홈페이지 - 더보기 - 개발자도구 - 화살표 모양 누르고 원하는 부분 클릭 -> 태그와 클래스 얻을 수 있음 

# 네이버 주식
url = "http://finance.naver.com/marketindex/"
page = urlopen(url)
soup = BeautifulSoup(page, "html.parser")
print(soup.prettify())

soup.find_all('span', 'value')[0].string  # find_all로 찾고, 첫번째 span태그의 value얻기 

# 다음 
url = 'https://www.daum.net/'
page = urlopen(url)
soup = BeautifulSoup(page, "html.parser")
print(soup.prettify())

# <strong class="date_today">07. 31. (수)</strong> # 다음에서 오늘 날짜
soup.find_all('strong','date_today')[0].string

#  <strong class="ico_ws ico_wm03">구름많음</strong> # 다음에서 오늘 날씨 
soup.find_all('strong','ico_ws ico_wm03')[0].string

# 샌드위치 집  goo.gl/wAtv1s
print(soup.find_all('div', 'sammy')) #>>> 전체 50개 태그들 나옴 
len(soup.fins_all('div','sammy'))  #>>> 50 



#%% Selenium
* 오피넷(주유소 정보) : www.opinet.co.kr
* 다른 것을 눌러도 url변경이 안되는 경우 selenium 을 사용
* https://chromedriver.chromium.org/downloads 에서 웹드라이버 다운 필요 
    * 스크립트 명령에 따라 액션 실행, 크롬버전 확인해야함(도움말 정보)

# prompt) pip install selenium
# python) import selenium
# print(selenium.__version__) #>>> 3.141.0

# 테스트 
from selenium import webdriver
driver = webdriver.Chrome('C:/Users/serah/ds/driver/chromedriver.exe')
driver.get('http://naver.com')  #>>> 생성된 브라우저는 접근 ㄴㄴ

# screenshot
driver.save_screenshot('C:/Users/serah/ds/images/001.jpg')

# 아이디, 비밀번호 집어넣기 - 이거 하면 막힘
elem_login = driver.find_element_by_id('id')
elem_login.clear()  # 입력되어있는 내용 clear 
elem_login.send_keys('id')

elem_login = driver.find_element_by_id('pw')
elem_login.clear()
elem_login.send_keys('')

# 로그인 버튼 누르기 -> 개발자도구에서 copy xpath 선택, 복붙 
xpath = 
driver.find_element_by_xpath(xpath).click()
# 메일 제목 가져오기
driver.get("http://mail.naver.com")

from bs4 import BeautifulSoup
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

raw_list = soup.find_all('div','name __ccr(lst.from)')
raw_list

send_list = [raw_list[n].find('a').get_text() for n in range(0, len(raw_list))]
send_list

# 항상 사용하고 닫아주기! 
driver.close()

#%% <<< ch 3 시각화 >>>


#%%  시각화 Tutorial

# 그래프 한글작업 
import matplotlib.pyplot as plt
# %matplotlib inline  <- for jupyter
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


# lib : plt,  data: gapminder data
import pandas as pd
df = pd.read_csv('C:/Users/serah/ds/data/gapminder.tsv', sep='\t')   # sep: 구분자  / engine='python'
df

# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  

global_life_exp = df.groupby('year')['lifeExp'].mean() #lifeexp를 연도별로 그룹화 하여 평균 구하기
print(global_life_exp)
print(df.groupby('year'))
global_life_exp.plot()

# lib: plt, data: sns의 anscombe
import seaborn as sns
anscombe = sns.load_dataset("anscombe")
print(anscombe)

print(type(anscombe))  # dataset 열이 데이터 그룹을 구분
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

#matplotlib.org/ 에서 그래프 옵션 체크!
dataset1 = anscombe[anscombe['dataset'] =='I']  # 첫번째 그룹만 
plt.plot(dataset1['x'],dataset1['y'])
plt.plot(dataset1['x'],dataset1['y'],'^')



#%% Matplotlib(as plt)  - 튜토리얼 

# matplotlib 로 그래프 그리기 

# 1. 전체 그래프가 위치할 기본틀을 만들고
# 2. 그래프를 그려넣을 격자
# 3. 격자에 그래프를 하나씩 추가, 순서는 왼쪽 -> 오른쪽
# 4. 1행이 차면 2행을 그려 넣는다.

# 0. 데이터 및 lib준비 
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline  # 그래프 결과를 out에 나타나게!')

import seaborn as sns
anscombe = sns.load_dataset("anscombe")


# 1. 데이터 준비 - 기본틀
dataset1 = anscombe[anscombe['dataset'] =='I']
dataset2 = anscombe[anscombe['dataset'] == 'II']
dataset3 = anscombe[anscombe['dataset'] == 'III']
dataset4 = anscombe[anscombe['dataset'] == 'IV']

# 2. 격자 
fig = plt.figure()
axes1 =fig.add_subplot(2,2,1)  # 행크기, 열크기
axes2 =fig.add_subplot(2,2,2)  
axes3 =fig.add_subplot(2,2,3)  
axes4 =fig.add_subplot(2,2,4)  


# 3. 그래프 추가
axes1.plot(dataset1['x'],dataset1['y'],'o')  #점으로 표현
axes2.plot(dataset2['x'],dataset2['y'],'o')
axes3.plot(dataset3['x'],dataset3['y'],'o')
axes4.plot(dataset4['x'],dataset4['y'],'o')
fig # 그래프 확인하기 위해 


# 4. 타이틀 추가
# 전체
fig.suptitle("Anscombe Data")  
# 그래프 별 
axes1.set_title("dataset1")
axes2.set_title("dataset2")
axes3.set_title("dataset3")
axes4.set_title("dataset4")

#겹치는 그래프 간격 넣어주기
fig.tight_layout()               


# 데이터 프레임과 시리즈로 그래프 그리기 
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
tips = sns.load_dataset("tips")


fig = plt.figure()
plt.plot([1,2,3,4,5,6,7,8,9,8,7,6,5,4,3,2,1,0])
plt.show()

# sns의 tips 데이터 
fig, ax = plt.subplots()
ax = tips['total_bill'].plot.hist() 

fig, ax = plt.subplots() 
ax = tips[['total_bill', 'tip']].plot.hist(alpha=0.5, bins=20, ax=ax) 

fig, ax = plt.subplots() 
ax = tips['tip'].plot.kde() 

fig, ax = plt.subplots() 
ax = tips.plot.scatter(x='total_bill', y='tip', ax=ax) 

fig, ax = plt.subplots() 
ax = tips.plot.hexbin(x='total_bill', y='tip', ax=ax) 

fig, ax = plt.subplots() 
ax = tips.plot.hexbin(x='total_bill', y='tip', gridsize=10, ax=ax) 

fig, ax = plt.subplots() 
ax = tips.plot.box(ax=ax) 








#%% Matplotlib (as plt) - 기초 그래프(히스토그램/산점도/박스/다변량)

# 0. lib/데이터 준비 
import pandas as pd
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
tips = sns.load_dataset("tips")
tips.head()

# 1. fig기본틀, axes1 그래프 격자 
fig = plt.figure()
axes1=fig.add_subplot(1,1,1)

# 1.히스토그램: df의 열 데이터 분포와 빈도를 살펴보는 용도로
# - 변수1개: 일변량 그래프
axes1.hist(tips['total_bill'], bins=10) #bins=10: 10단위로 x축의 간격 조정
axes1.set_title('Histogram of total bill')
axes1.set_xlabel('Freq')
axes1.set_ylabel('Total bill')
fig

# 2. 산점도 : 변수2개, 이변량 그래프 
# total_bill 열에 따른 tip열의 분포를 나타낸 산점도 그래프
scatter_plot = plt.figure()
axes1= scatter_plot.add_subplot(1,1,1)
axes1.scatter(tips['total_bill'],tips['tip'])
axes1.set_title('Scatterplot of Total bill vs tip')
axes1.set_xlabel('total bill')
axes1.set_ylabel('tip')
                   

# 3. 박스 그래프 
# - 이산형 변수(범주형): 명확하게 구분되는 값을 의미 남/녀
# - 연속형 변수 : 명확하게 셀수 없는 값들 tips

boxplot = plt.figure()
axes1 = boxplot.add_subplot(1,1,1)
axes1.boxplot(
    [tips[tips['sex'] == 'Female']['tip'],
     tips[tips['sex'] == 'Male']['tip']],
    labels=['Female','Male'])        # for 성별구분 , label 없으면 1,2로 나옴 

axes1.set_title('Boxplot of Tips by Sex')
axes1.set_xlabel('Sex')
axes1.set_ylabel('Tip')



# 4. 다변량 그래프 그리기 - 3개 이상의 변수
# 성별을 새 변수로 추가 (문자열은 산점도 그래프의 색상을 지정)
def recode_sex(sex):
    if sex =='Female':
        return 0
    else:
        return 1
    
tips['sex_color'] = tips['sex'].apply(recode_sex)  # apply : 브로드캐스팅! 한번에 함수적용

scatter_plot =plt.figure()
axes1=scatter_plot.add_subplot(1,1,1)
axes1.scatter(
    x = tips['total_bill'],
    y = tips['tip'],
    s = tips['size'] * 10,
    c = tips['sex_color'],
    alpha = 0.5)  # alpha 채도 변경 
axes1.set_title('Total bill vs Tip Colored by sex')
axes1.set_xlabel('Total bill')
axes1.set_ylabel('Tip')


#%% Matplotlib (as plt) - 파이썬 주무르기

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

x = [0,1,2,3,4,5,6,7,8,9,10]
y = [0,1,2,3,4,5,4,3,2,1,0]
plt.plot(x,y)
plt.show()

plt.figure(figsize=(10,6))
plt.plot(x,y)
plt.grid()
plt.show()

# numpy 활용 
import numpy as np
t = np.arange(0,12,0.01)  # 0부터 12까지 0.01간격으로 만듦
y = np.sin(t)

plt.figure(figsize = (10,6))
plt.plot(t,y)
plt.grid()  # 그리드 추가 
plt.xlabel('time')       # x축 라벨 적용하기
plt.ylabel('Amplitude')  # y축 라벨 적용하기
plt.title('Example of sinewave')  # title 라벨 적용하기 
plt.show()

# sin, cos함수 
plt.figure(figsize=(10,6))
plt.plot(t, np.sin(t), lw = 3,label='sin')
plt.plot(t, np.cos(t), 'r', label='cos')
plt.grid()
plt.legend()     # 자동으로 위치 지정됨 
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.show()

# 위 그래프 확대 xlim, ylim 지정 
plt.figure(figsize=(10,6))
plt.plot(t, np.sin(t), lw=3, label='sin')
plt.plot(t, np.cos(t), 'r', label='cos')
plt.grid()
plt.legend()
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.title('Example of sinewave')
plt.ylim(-1.2, 1.2)
plt.xlim(0, np.pi)
plt.show()

import numpy as np
t =np.arange(0,2*np.pi, 0.01)
# sin(t) graph
plt.figure(figsize =(10,6))
plt.plot(t, np.sin(t))
plt.grid()
plt.title('sin')
plt.xlabel('sec')
plt.ylabel('sin')
plt.show()

# 옵션) lw = 3 : 선의 굵기(default 보다 굵) /ls = " " : 선 스타일 / 'r' : red , 'b' : blue

# 그래프 다양한 옵션 
# g1
t = np.arange(0, 5, 0.5)
plt.figure(figsize=(10,6))
plt.plot(t, t, 'r--')
plt.plot(t, t**2, 'bs')
plt.plot(t, t**3, 'g^')
plt.title('g1')
plt.show()

#g2
t = np.arange(0, 5, 0.5)
plt.figure(figsize=(10,6))
plt.plot(t, t**2, 'bs')
plt.title('g2')
plt.show()

# g3
plt.figure(figsize=(10,6))
plt.plot(t, t**3, 'g^')
plt.title('g3')
plt.show()


# g4
t = [0, 1, 2, 3, 4, 5, 6] 
y = [1, 4, 5, 8, 9, 5, 3]
plt.figure(figsize=(10,6))
plt.plot(t, y, color='green')
plt.title('g4')
plt.show()

# g5
plt.figure(figsize=(10,6))
plt.plot(t, y, color='green', linestyle='dashed')
plt.title('g5')
plt.show()

#g6
plt.figure(figsize=(10,6))
plt.plot(t, y, color='green', linestyle='dashed', marker='o')
plt.title('g6')
plt.show()

#g7
plt.figure(figsize=(10,6))
plt.plot(t, y, color='green', linestyle='dashed', marker='o',
        markerfacecolor = 'blue')
plt.title('g7')
plt.show()

#g8
plt.figure(figsize=(10,6))
plt.plot(t, y, color='green', linestyle='dashed', marker='o',
        markerfacecolor = 'blue', markersize=12)
plt.xlim([-0.5, 6.5])
plt.ylim([0.5, 9.5])
plt.title('g8')
plt.show()

#g9
t = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,9,8,3,2,4,3,4])
plt.figure(figsize=(10,6))
plt.scatter(t,y)
plt.title('g9')
plt.show()

#g10
plt.figure(figsize=(10,6))
plt.scatter(t,y, marker='>')
plt.title('g10')
plt.show()

#g11 - x축에 따라 색상을 바꾸는 
colormap = t
plt.figure(figsize=(10,6))
plt.scatter(t,y, s = 50, c = colormap, marker='>')
plt.colorbar()   # 오른쪽에 색상표생성 
plt.title('g11')
plt.show()

#g12
colormap = t
plt.figure(figsize=(10,6))
plt.scatter(t,y, s = 50, c = colormap, marker='>')
plt.colorbar()
plt.title('g12')
plt.show()

#g13
# np의 랜덤변수 함수를 이용해 데이터3개 만들기
# loc = 평균값, scale = 표준편차값
s1 = np.random.normal(loc=0, scale=1, size=1000)
s2 = np.random.normal(loc=5, scale=0.5, size=1000)
s3 = np.random.normal(loc=10, scale=2, size=1000)
plt.figure(figsize=(10,6))
plt.plot(s1, label='s1')
plt.plot(s2, label='s2')
plt.plot(s3, label='s3')
plt.legend()
plt.title('g13')
plt.show()


#g14
plt.figure(figsize=(10,6))
plt.boxplot((s1, s2, s3))
plt.grid()
plt.title('g14')
plt.show()







#%% seaborn (as sns) - 기초
# seaborn library는 matplotlib lib을 기반으로 만들어짐 
# seaborn(sns)로 히스토그램을 그리려면 
    # subplots, distplot(:기본 틀 만듦) method 사용 
    # total_bill method : total_bill 열 데이터를 전달 

import seaborn as sns

plt.figure(figsize=(10,6))
plt.subplot(221)
plt.subplot(222)
plt.subplot(212)
plt.show()


plt.figure(figsize=(10,6))

plt.subplot(411)
plt.subplot(423)
plt.subplot(424)
plt.subplot(413)
plt.subplot(414)

plt.show()

t = np.arange(0,5,0.01)

plt.figure(figsize=(10,12))

plt.subplot(411)
plt.plot(t,np.sqrt(t))
plt.grid()

plt.subplot(423)
plt.plot(t,t**2)
plt.grid()

plt.subplot(424)
plt.plot(t,t**3)
plt.grid()

plt.subplot(413)
plt.plot(t,np.sin(t))
plt.grid()

plt.subplot(414)
plt.plot(t,np.cos(t))
plt.grid()

plt.show()


#%% Seaborn(as sns) - 기초 그래프 ~ 시각화 


# 0. 데이터 준비 
import seaborn as sns
tips = sns.load_dataset('tips')

# 1. 히스토그램 / 밀집도 / 러그 

# 히스토그램 + 밀집도 
ax = plt.subplots()
ax = sns.distplot(tips['total_bill'])
ax.set_title('Total bill Histogram with Densiry Plot')

# 밀집도(정규화 시켜 넓이가 1이 되도록 그린 선 그래프) 제외하기
ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], kde =False)
ax.set_title('Total bill Histogram')
ax.set_xlabel('Total bill')
ax.set_ylabel('Frequency')

# 히스토그램을 제외하기
ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], hist =False)
ax.set_title('Total bill Density')
ax.set_xlabel('Total bill')
ax.set_ylabel('Unit probability')

# rug 추가 : 그래프의 축에 동일한 길이의 직선을 붙여 데이터의 밀집도(빈도수)를 표현
ax = plt.subplots()
ax = sns.distplot(tips['total_bill'], rug = True)
ax.set_title('Total bill Histogram with Densiry Plot and rug')
ax.set_xlabel('Total bill')

# 2. 카운트 그래프 count graph  - 이산값을 나타낸 그래프 

# 기본 1 - count of days
ax= plt.subplots()
ax= sns.countplot('day', data = tips)
ax.set_title('Count of Days')
ax.set_xlabel('Days of Week')
ax.set_ylabel('Frequency')

# 기본 2 - count of sex
print(tips.head())
ax= plt.subplots()
ax= sns.countplot('sex', data = tips)
ax.set_title('Count of sex')
ax.set_xlabel('sex')
ax.set_ylabel('Frequency')


# 3-1) 산점도 
# .regplot method사용하여 산점도와 회귀선 그릴 수 있음

# only scatterplot 
ax = plt.subplots()
ax = sns.regplot(x = 'total_bill', y = 'tip', data = tips, fit_reg =False)
ax.set_title('Scatterplot of total bill and tip')
ax.set_xlabel('Total bill')
ax.set_ylabel('tip')

# scatterplot and regression plot
ax = plt.subplots()
ax = sns.regplot(x = 'total_bill', y = 'tip', data = tips)
ax.set_title('Scatterplot and regplot of total bill and tip')
ax.set_xlabel('Total bill')
ax.set_ylabel('tip')




# 3-2) 히스토그램과 산점도 (+ hexbin)

# histgram & scatterplot
joint = sns.jointplot(x = 'total_bill', y = 'tip', data = tips)
joint.set_axis_labels ( xlabel = 'total bill', ylabel = 'tip')
joint.fig.suptitle('Joint plot', fontsize = 10, y=1.03)  # title y 위치

# histgram & scatterplot with hexbin
joint = sns.jointplot(x = 'total_bill', y = 'tip', data = tips, kind = 'hex')
joint.set_axis_labels ( xlabel = 'total bill', ylabel = 'tip')
joint.fig.suptitle('Joint plot', fontsize = 10, y=1.03)  


# 4. 이차원 밀집도 그리기
# kde는 그래프 그려주는 것 -> kde, ax=plt.subplots()  해도 나옴 

ax = plt.subplots()
ax = sns.kdeplot(data = tips['total_bill'], data2 = tips['tip'], shade = True)  #shade 음영의 차이 
ax.set_title('Kernel Density plot of total bill and tip')
ax.set_xlabel('total bill')
ax.set_ylabel('tip')


# 5. 바 그래프 그리기

# 1)
ax = plt.subplots()
ax = sns.barplot(x = 'time', y= 'total_bill', data= tips)
ax.set_title('bar plot of average total bill for time of day')
ax.set_xlabel('Time of day')
ax.set_ylabel('Average total bill')
print(tips.head())

# 2)
ax = plt.subplots()
ax = sns.barplot(x = 'day', y= 'total_bill', data= tips)
ax.set_title('bar plot of average total bill of days')
ax.set_xlabel('week')
ax.set_ylabel('average total bill')

# 6. 박스 그래프 그리기 

# 6-1) 기본 

ax = plt.subplots()
ax = sns.boxplot(x = 'time', y = 'total_bill', data = tips)
ax.set_title('Boxplot of total bill by time of day')
ax.set_xlabel('time of day')
ax.set_ylabel('total bill')


# 날짜별 total_bill  + hue속성: smoker별로 나누기
ax = plt.subplots()
ax = sns.boxplot(x = 'day', y = 'total_bill', hue = 'smoker', data = tips)
ax.set_title('Boxplot of total bill of days')
ax.set_xlabel('week')
ax.set_ylabel('total bill')


# 6-2) 바이올린 박스플롯 -  violin shaped boxplot == violinplot
ax = plt.subplots()
ax = sns.violinplot(x = 'time', y = 'total_bill', data= tips)
ax.set_title('violin plot of total bill by time of day')
ax.set_xlabel('time of day')
ax.set_ylabel('total bill')

# violinplot + hue 추가 
violin, ax = plt.subplots()
ax = sns.violinplot(x = 'time', y = 'total_bill', hue = 'sex', data= tips, split= True)  # split 안쓰면 4개나옴 각각따로 
plt.show()

# violinplot에 스타일 적용하기 
fig, ax = plt.subplots() 
ax = sns.violinplot(x='time', y='total_bill', hue='sex', data=tips, split=True) 

sns.set_style('whitegrid') 
fig, ax = plt.subplots() 
ax = sns.violinplot(x='time', y='total_bill', hue='sex', data=tips, split=True) 


fig = plt.figure() 
seaborn_styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks'] 
for idx, style in enumerate(seaborn_styles):
    plot_position = idx + 1
    with sns.axes_style(style):
        ax = fig.add_subplot(2, 3, plot_position)
        violin = sns.violinplot(x='time', y='total_bill', data=tips, ax=ax)
        violin.set_title(style)         
fig.tight_layout() 

# 7. 관계 그래프 그리기
fig = sns.pairplot(tips)


pair_grid = sns.PairGrid(tips)
pair_grid = pair_grid.map_upper(sns.regplot)
pair_grid = pair_grid.map_lower(sns.kdeplot)
pair_grid = pair_grid.map_diag(sns.distplot, rug = True)
plt.show()

# 8. 여러가지 다변량 그래프 

# 8-1) 산점도 관계 그래프 - 색상추가 
scatter = sns.lmplot ( x = 'total_bill', y = 'tip', data= tips,  hue = 'sex', fit_reg = True)
plt.show()

# 8-2) 관계 그래프 hue 추가 
fig = sns.pairplot(tips, hue = 'sex') #pa 왜 붙어있었지?


# 8-3) 산점도 그래프의 크기과 모양 조절하기 
# 동그라미 & 사이즈 조절 (scatter_kws={'s': ) 
scatter = sns.lmplot(x='total_bill', y='tip', data=tips, fit_reg=False, hue='sex', scatter_kws={'s': tips['size']*10}) 
plt.show()

# 동그라미 엑스 - > 마커 스타일 검색하면 많음(공유폴더 확인)
scatter = sns.lmplot(x='total_bill', y='tip', data=tips, fit_reg=False, hue='sex', markers=['o', 'x'], scatter_kws={'s': tips['size']*10}) 
plt.show()

# implot method로 4개의 데이터 그룹의 그래프 한번에 그리기 
anscombe_plot = sns.lmplot(x='x', y='y', data=anscombe, fit_reg=False)

anscombe_plot = sns.lmplot(x='x', y='y', data=anscombe, fit_reg=False, col='dataset', col_wrap=2)

# facegrid 클래스로 그룹별 그래프 그릴수 있음 
facet = sns.FacetGrid(tips, col='time') 
facet.map(sns.distplot, 'total_bill', rug=True) 

facet = sns.FacetGrid(tips, col='day', hue='sex') 
facet = facet.map(plt.scatter, 'total_bill', 'tip') 
facet = facet.add_legend() 

facet = sns.FacetGrid(tips, col='time', row='smoker', hue='sex') 
facet.map(plt.scatter, 'total_bill', 'tip') 











#%%  Google maps

# prompt 에서 pip install googlemaps
import googlemaps
gmaps = googlemaps.Client(key = '')

gmaps.geocode('서울중부경찰서', language= 'ko')
# 나머지는 예제 참조 

# Folium

# prompt) pip install folium
import folium

map_osm = folium.Map(location = [45.5236, -122.6750],
                     zoom_start = 13)
map_osm
map_osm.save('map_osm.html')  # 스파이더에서 folium 보고 싶을때 , 주피터 권장 Users/serah에 저장 

stamen = folium.Map(location = [45.5236, -122.6750],
                    tiles = 'Stamen Toner',  # Stamen Terrain 
                    zoom_start = 13)
stamen

stamen = folium.Map(location = [45.5236, -122.6750],
                   tiles = 'Stamen Terrain', zoom_start=13)
stamen

# 위도 latitude 가로 / 경도 longitude 세로 

# 기존 map에 
# folium.Marker([lat,lon], 
#               popup ='쓸내용',
#               icon = folium.Icon(icon = '모양')).add_to(map) 
# folium.CircleMarker([lat, lon], 
#                    radius = 50, 
#                     popup = 'Park', 
#                     color = '#3186cc', 
#                     fill_color= '#3186cc').add_to(map)
# folium.RegularPolygonMarker([lat,lon], 
#                             popup='Broadway Bridge', 
#                             fill_color='#769d96', 
#                             number_of_sides=8, radius=10).add_to(map)


map_1 = stamen = folium.Map(location = [45.5236, -122.6750],
                   tiles = 'Stamen Terrain', zoom_start=12)
folium.Marker([45.3288, -121.6625], popup = 'Mt. Hood Meadows', icon = folium.Icon(icon = 'cloud')).add_to(map_1)
folium.Marker([45.3288, -121.7113], popup = 'Timberline Lodge', icon = folium.Icon(icon = 'cloud')).add_to(map_1)
map_1
map_1.save('map1.html')

map_2 = stamen = folium.Map(location = [45.5236, -122.6750],
                   tiles = 'Stamen Terrain', zoom_start=12)
folium.Marker([45.3288, -121.6625], 
              popup = 'Mt. Hood Meadows', 
              icon = folium.Icon(icon = 'cloud')).add_to(map_2)
folium.Marker([45.3311, -121.7113], 
              popup = 'Timberline Lodge', 
              icon = folium.Icon(color = 'green')).add_to(map_2)
folium.Marker([45.3300, -121.6823], 
              popup = 'Some other Loc.', 
              icon = folium.Icon(color = 'red', icon= 'info-sign')).add_to(map_2)
map_2


map_3 = stamen = folium.Map(location = [45.5236, -122.6750],
                   tiles = 'Stamen Terrain', zoom_start=12)
folium.Marker([45.5244, -121.6699], 
              popup = 'The Waterfront').add_to(map_3)
folium.CircleMarker([45.5215, -122.6261], 
                    radius = 50, popup = 'Park', 
                    color = '#3186cc', fill_color= '#3186cc').add_to(map_3)
map_3

map_4 = folium.Map(location=[45.5236, -122.6750], zoom_start=13)
folium.RegularPolygonMarker([45.5012, -122.6655], 
                            popup='Ross Island Bridge', fill_color='#132b5e', 
                            number_of_sides=3, radius=10).add_to(map_4)
folium.RegularPolygonMarker([45.5132, -122.6708], 
                            popup='Hawthorne Bridge', fill_color='#45647d', 
                            number_of_sides=4, radius=10).add_to(map_4)
folium.RegularPolygonMarker([45.5275, -122.6692], 
                            popup='Steel Bridge', fill_color='#769d96', 
                            number_of_sides=6, radius=10).add_to(map_4)
folium.RegularPolygonMarker([45.5318, -122.6745], 
                            popup='Broadway Bridge', fill_color='#769d96', 
                            number_of_sides=8, radius=10).add_to(map_4)
map_4


# 미국 주별 실업률 
import folium
import pandas as pd
state_unemployment = 'C:/Users/serah/ds/data/02. folium_US_Unemployment_Oct2012.csv'

state_data = pd.read_csv(state_unemployment)
state_data.head()

state_geo = 'C:/Users/serah/ds/data/02. folium_us-states.json'  #json파일경로를 담고 

map = folium.Map(location=[40, -98], zoom_start=4)
map.choropleth(geo_data=state_geo, 
               data=state_data,
               columns=['State', 'Unemployment'],
               key_on='feature.id',
               fill_color='Blues',  #YlGn
               legend_name='Unemployment Rate (%)')
map

#%% Seaborn(as sns) - 파이썬 주무르기

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns

x = np.linspace(0, 14, 100)
y1 = np.sin(x)
y2 = 2*np.sin(x+0.5)
y3 = 3*np.sin(x+1.0)
y4 = 4*np.sin(x+1.5)
plt.figure(figsize=(10,6))
plt.plot(x,y1, x,y2, x,y3, x,y4)
plt.title('g1 : sin graphs')
plt.show()


sns.set_style("white")   # white, whitegrid, dark, darkgrid
plt.figure(figsize=(10,6))
plt.plot(x,y1, x,y2, x,y3, x,y4)
plt.title('g2 :white graph')
plt.show()

sns.set_style("darkgrid")   
plt.figure(figsize=(10,6))
plt.plot(x,y1, x,y2, x,y3, x,y4)
plt.title('g3 :darkgrid, despine graph')
sns.despine(offset =10 )  # 뭘까 이건 
plt.show()


# 준비
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')

tips = sns.load_dataset("tips")
tips.head(5)

# 내장 데이터 tips 사용 
# boxplot
sns.set_style("whitegrid")
plt.figure(figsize=(8,6))
sns.boxplot(y=tips["total_bill"])  # x= 으로 바꾸면 옆으로 누움 
plt.title('g1:boxplot only totalbill')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x="day", y="total_bill", data=tips)
plt.title('g2:boxplot totalbill/day')
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, palette="Set3")
plt.title('g3:boxplot totalbill/day /smoker')
plt.show()


# swarmplot 상자그림 모양의 산점도 
plt.figure(figsize=(8,6))
sns.swarmplot(x="day", y="total_bill", data=tips, color=".5")
plt.title('g1:swarmplot')
plt.show()

# swarmplot + boxplot
plt.figure(figsize=(8,6))
sns.boxplot(x="day", y="total_bill", data=tips)
sns.swarmplot(x="day", y="total_bill", data=tips, color=".25")
plt.title('g2:swarmplot + boxplot')
plt.show()

# lmplot - scatter, regression linear line, ci(=유효범위) 
sns.set_style("darkgrid")
sns.lmplot(x="total_bill", y="tip", data=tips, size=7)
plt.title('g1 : tip/ totalbill  lmplot')
plt.show()

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, size=7)
plt.title('g2 : tip/ totalbill /smoker lmplot')
plt.show()

sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, palette="Set1", size=7)
plt.title('g3 : tip/ totalbill /smoker with palette lmplot')
plt.show()

# 내장 데이터 : 연도 및 월별 항공기 승객수 데이터
flights = sns.load_dataset("flights")
print(flights.head(5))

flights = flights.pivot("month", "year", "passengers")  # 피봇으로 정리 
print(flights.head(5))

# heatmap 히트맵 
plt.figure(figsize=(10,8))
sns.heatmap(flights)
plt.title('g1 : simple heatmap')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(flights, annot=True, fmt="d")   #fmt없으면 e로 나옴 
plt.title('g2 : annot true, fmt=d heatmap')
plt.show()


# 히트맵 난수 예제 
uniform_data = np.random.rand(10, 12)
print(uniform_data)

sns.heatmap(uniform_data)
plt.title('g3: random heatmap1')
plt.show()

sns.heatmap(uniform_data, vmin=0, vmax=2)  # 범위지정 
plt.title('g4: random heatmap2')
plt.show()

# 내장데이터 : 아이리스 꽃 데이터  , sns.pairplot 
sns.set(style="ticks")
iris = sns.load_dataset("iris")
print(iris.head(10))

sns.pairplot(iris)
plt.title('g1: simple pairplot')
plt.show()

sns.pairplot(iris, hue="species")
plt.title('g2: hue=species pairplot')
plt.show()

sns.pairplot(iris, vars=["sepal_width", "sepal_length"])
plt.title('g3: sepal width/length pairplot')
plt.show()

sns.pairplot(iris, x_vars=["sepal_width", "sepal_length"], 
             y_vars=["petal_width", "petal_length"])
plt.title('g4: sepal width/length and petal width/length')
plt.show()

# anscombe 데이터 활용 실습 
anscombe = sns.load_dataset("anscombe")
anscombe.head(5)

sns.set_style("darkgrid")
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),  ci=None, size=7)
plt.title('g1')
plt.show()

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'I'"),
           ci=None, scatter_kws={"s": 80}, size=7)
plt.title('g2')
plt.show()

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           order=1, ci=None, scatter_kws={"s": 80}, size=7)
plt.title('g3')
plt.show()

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           order=2, ci=None, scatter_kws={"s": 80}, size=7)
plt.title('g4')
plt.show()

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           ci=None, scatter_kws={"s": 80}, size=7)
plt.title('g5')
plt.show()

sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'III'"),
           robust=True, ci=None, scatter_kws={"s": 80}, size=7)
plt.title('g6')
plt.show()


#%% 파이차트
import pandas as pd
rawdata=pd.read_excel('C:/Users/serah/ds/data/Report.xls', header =1)
rawdata

col_names =['스트레스','스트레스남학생','스트레스여학생','우울감경험률','우울남학생','우울여학생','자살생각율','자살남학생','자살여학생']
data= pd.read_excel('C:/Users/serah/ds/data/Report.xls', header =1, usecols='C:K', names =col_names)
data

data.loc[1] = 100.-data.loc[0]
data

data['응답'] =['그렇다','아니다']
data

data.set_index('응답',drop=True, inplace =True)
data

# 그래프 한글작업 
import matplotlib.pyplot as plt
# %matplotlib inline
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

# 
    
data['스트레스'].plot.pie()

data['스트레스'].plot.pie(explode =[0,0.02])  # 구분하는 곳 간격 

# 세개 그리기 
f, ax =plt.subplots(1,3, figsize=(16,8))

data['스트레스'].plot.pie(explode=[0,0.02],
                     ax =ax[0], 
                     autopct = '%1.1f%%')
ax[0].set_title('스트레스를 받은 적이 있다.')
ax[0].set_ylabel('')


data['우울감경험률'].plot.pie(explode=[0,0.02],
                     ax =ax[1], 
                     autopct = '%1.1f%%')
ax[1].set_title('우울감을 경험한 적이 있다.')
ax[1].set_ylabel('')


data['자살생각율'].plot.pie(explode=[0,0.02],
                     ax =ax[2], 
                     autopct = '%1.1f%%')
ax[2].set_title('자살을 고민한 적이 있다.')
ax[2].set_ylabel('')

plt.show()




















