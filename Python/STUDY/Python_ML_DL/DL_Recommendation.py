'''
contents
0. 전자상거래 추천 시스템 - Tensorflow        # 출처 : 실전활용! 텐서플로 딥러닝 프로젝트


'''

# < 0. 추천 시스템 구현 
# 출처 : 실전활용! 텐서플로 딥러닝 프로젝트
# https://wikibook.co.kr/tensorflow-projects/

import tensorflow as tf
import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


df= pd.read_excel('C:/dataset/Online Retail.xlsx')  # utf8, euc-kr, cp949,ISO-8859-1
df.head()
df.info()

# 시간 오래걸림 다음에 파일 불러 올때를 대비해서 pickle로 저장하기
import pickle
with open('df_retail.bin','wb') as f_out:
    pickle.dump(df, f_out)

# 피클 열기
with open('df_retail.bin','rb') as f_in:
    df=pickle.load(f_in)

df.columns =df.columns.str.lower()

df.invoiceno.unique()
df.sample(10)
df = df[~df.invoiceno.astype('str').str.startswith('C')].reset_index(drop=True)


# str.startswith(), str.endswith() 처음이나 마지막에 있는 텍스트 매칭
str1 = 'this is an apple! wow'
str1.startswith('this')
str1.endswith('w')
# 찾고 싶은게 여러개라면 만약 this 와 thi 일경우 tuple(리스트 or set)로 묶어줘야함
# str1.startswith(tuple(['thi','this']))

# 물결 표시(~)는 판다스를 색인할 때 True를 False로, False를 True로 뒤집어줌
a = pd.DataFrame({'name': ['a', 'b', 'c', 'd', 'e'],
                'age': [1, 2, 3, 4, 5],
                'score': [80, np.nan, 80, np.nan, 100]})
a
# index로 뽑아낼 수 있음
b = ~(a['score'] == 80)
b
# 해당하는 인덱스의 값 가져오기
a[b]

df.customerid = df.customerid.fillna(-1).astype('int32')  # customer id 없는 사람 -1로 fill
df.sample(10)

# stockcode 정수로 인코딩 - 고유 인덱스 번호로 mapping
stock_value = df.stockcode.astype('str')
stock_value
stockcodes = sorted(set(stock_value)) # 중복 제거 하고 정렬
stockcodes = {c: i for (i, c) in enumerate(stockcodes)}
stockcodes # 0 ~
len(stockcodes) # 4059개
df_stockcode = stock_value.map(stockcodes).astype('int32')

# train: 11.10.09 이전의 10월치, valid : 그 다음1개월, test : 11.11.09 이후 1개월치
df_train = df[df.invoicedate <'2011-10-09']
df_valid = df[(df.invoicedate >= '2011-10-09') & (df.invoicedate <= '2011-11-09')]  # 조건 걸려면 조건 별로 하나씩 괄호안에 넣어주기
df_test = df[df.invoicedate >= '2011-11-09']


'''
# 단순한 추천 시스템 
5개의 추천 보여줌
추천시스템의 평가 : 정밀도 precision,    성공한 추천수(실제로 구매한 항목수) / 총 추천수 

다른 정밀도 지표 
MAP (Mean Average Precision)
NDCG(Normalized Discounted Cumulativa Gain) 

'''

# 기본 기준선 : 항목별로 얼마나 구매했는지 계산, 자주 구매된 top5 가져와 모든 사용자에게 추천 가능
top=df_train.stockcode.value_counts().head(5).index.values

