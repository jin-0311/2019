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

df = pd.read_excel('C:/dataset/Online Retail.xlsx')  # utf8, euc-kr, cp949,ISO-8859-1
df.head()
df.info()

# 시간 오래걸림 다음에 파일 불러 올때를 대비해서 pickle로 저장하기
import pickle

with open('df_retail.bin', 'wb') as f_out:
    pickle.dump(df, f_out)

# 피클 열기
with open('df_retail.bin', 'rb') as f_in:
    df = pickle.load(f_in)

df.columns = df.columns.str.lower()

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
stockcodes = sorted(set(stock_value))  # 중복 제거 하고 정렬
stockcodes = {c: i for (i, c) in enumerate(stockcodes)}
stockcodes  # 0 ~
len(stockcodes)  # 4059개
df.stockcode = stock_value.map(stockcodes).astype('int32')  # df.stockcode로 해야 적용됨

# train: 11.10.09 이전의 10월치, valid : 그 다음1개월, test : 11.11.09 이후 1개월치
df_train = df[df.invoicedate < '2011-10-09']
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
top = df_train.stockcode.value_counts().head(5).index.values
top  # Out[31]: array([3527, 3506, 1347, 2730,  180], dtype=int64)

num_groups = len(df_valid.invoiceno.drop_duplicates())  # top 배열 반복, 이를 추천항목으로 사용
num_groups  # 2435
baseline = np.tile(top, num_groups).reshape(-1, 5)  # np.tile : 배열을 불러들여 num_group만큼 반복
baseline

# 정밀도 계산전 할일
# invoiceno 단위로 groupby, 각 거래마다 추천 생성, 그룹별 정확한 예측 수 기록, 전체 정밀도 계산  -> 하지만 느려짐 (pd의 groupby속도가 느리기 때문)

# 데이터가 잘 저장되어 있으므로
'''
invoiceno은 특정 시간(invoicedate)에 주문한 것끼리 다 같음

i: 어떠한 거래가 특정 행번호인 i에서 시작
k: 거래 항목의 개수
즉 i 와 i+k사이의 모든 행은 동일한 invoiceid에 해당 

따라서 각 거래이 시작과 끝을 알아야 함. 이를 위해서 n+1의 특별한 행렬 유지 (n:데이터셋에 있는 그룹(거래)의 개수)
이 배열을 indptr 이라고 할때, 각 거래인 t에 대해
indptr[t] : 거래 시작 df의 행번호 반환
indptr[t+1] : 거래 끝난 df의 행번호 반환 

# 다양한 길이의 그룹을 표현하는 위의 방식은 CRS or CSR(


'''
a = df.head(20)
print(a)