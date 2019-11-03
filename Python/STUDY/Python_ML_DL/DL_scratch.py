# 출처 : 밑바닥부터 시작하는 딥러닝
# http://www.hanbit.co.kr/media/books/book_view.html?p_code=B8475831198
# coding='utf8'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


'''
ch1 python introduction
ch2 perceptron : AND/NAND/OR/XOR 논리 회로, 다층 퍼셉트론
ch3 신경망 : 활성화함수(시그모이드/렐루/비선형함수), 다차원배열 계산, 3층 신경망 구현, 출력층설계(항등/소프트맥스함수), 손글씨 숫자인식
ch4 신경망 학습 : 손실함수(평균제곱오차, 교차 엔트로피오차,미니배치학습), 수치미분, 기울기, 학습알고리즘 구현
ch5 오차역전파법 : 계산 그래프, 연쇄법칙, 역전파, 단순한 계층구현, 활성화함수 계층구현, affine/softmax계층구현, 오차역전파법 구현
ch6 학습관련 기술: 매개변수 갱신, 가중치초기값, 배치 정규화, 바른학습을 위해, 하이퍼파라미터 최적화
ch7 합성곱 신경망 CNN : 전체구조, 합성곱계층 ,풀링 계층 , 합성곱/풀링 구현, cnn구현,시각화
ch8 딥러닝: 더 깊은 신경망, 딥러닝 역사, 딥러닝 고속화, 활용, 미래
'''
#
# chapter 1 intro
# < class 생성
class man :
    def __init__(self, name):
        self.name=name
        print('Initialized!')
    def hello(self):
        print('hello '+self.name+'!!')
    def goodbye(self):
        print('goodbye '+self.name +'!!!')
m=man('jin')
m.hello()
m.goodbye()

# < numpy broadcast
import numpy as np
A = np.array([[1,2],[3,4]])
B = np.array([10,20])
A*B

X= np.array([[51,55],[14,19],[0,4]])
X[0]
X[0][1]
for row in X:
    print(row)
X=X.flatten()
X
X[np.array([0,2,4])] # 인덱싱
X>15
X[X>15]



# chapter 2 perceptron
'''
# 단어 
x1,x2:input
w1,w2:weight
theta: 임계값 -> 임계값을 넘으면 y=1, 그렇지 않으면 y=0

# 단순 논리회로 
AND gate: 두 입력이 모두 1일때만 y=1
NAND gate: 두 입력이 모두 1일때만 y=0 
OR gate : 둘중 하나가 1이면 y=1

퍼셉트론의 구조는 모두 같고 매개변수(w1,w2,theta)만 다름
'''

# < 단순 퍼셉트론
def AND(x1,x2):
    w1, w2, theta =0.5,0.5,0.7
    tmp=x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
AND(0,0)
AND(0,1)
AND(1,0)
AND(1,1)

# 가중치w, 편향b 도입 퍼셉트론
# theta를 -b로 치환  xn*wn+bn >0 , y=1
import numpy as np
x=np.array([0,1])
w=np.array([0.5,0.5])
b=-0.7
w*x
np.sum(w*x)
np.sum(w*x)+b

# 가중치w, 편향b 구현 퍼셉트론
# AND gate
def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else :
        return 1
AND(1,1)

# NAND gate : w,b만 다름
def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else :
        return 1
NAND(1,1)

# OR gate : w,b만 다름
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else :
        return 1
OR(0,0)

# < 퍼셉트론의 한계 XOR gate
'''
XOR gate : 배타적 논리합
x1,x2중 하나만 1일때 y=1  

비선형: 곡선or 2개 이상의 직선으로 이루어진
선형: 1개의 직선으로 이루어진 

다층 퍼셉트론 MLP Multi-Layor Perceptron : 층을 쌓아서 비선형을 표현하기 가능 
NAND의 결과인 s1, OR의 결과인 s2 -> AND로 y출력 
'''
# 기존 게이트를 조합하여 구현
# (0,0),(1,1)  /  (1,0),(0,1) 두개의 그룹으로 나눌 때 선형함수로 불가
def XOR (x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y
XOR(0,0)
XOR(1,0)
XOR(0,1)
XOR(1,1)

# chapter 3 신경망 Neural Network
'''
신경망은 가중치 매개변수를 데이터로 부터 자동으로 학습할 수 있음
입력층-> 은닉층 -> 출력층

# 퍼셉트론 복습
퍼셉트론은 수동으로 가중치 설정
y = 0, (b+w1x1+w2x2 <= 0)
    1, (b+w1x1+w2x2 > 0)

b는 x1,x2위에 입력이 1인 뉴런 (항상 입력1)
y = h(b+w1x1+w2x2)
h(x) = 0, x<=0
       1, x>0
h(x) : 활성화 함수 activation function, (단순)퍼셉트론에서는 계단함수가 활성화 함수
'''

# < 활성화 함수 -  퍼셉트론의 계단 함수
# np 지원이 안되는 계단 함수 구현
def step_func1(x):
    if x>0 :
        return 1
    else:
        return 0
# np 가능 계단함수
def step_func2(x):
    y=x>0
    return y.astype(np.int)

x=np.array([-1.0,1.0,2.0])
x
y=x>0
y  # bool
y=y.astype(np.int)
y   # np.int  True:1, False:0

# 그래프
import numpy as np
import matplotlib.pyplot as plt
def step_function(x):
    return np.array(x>0, dtype=np.int)
x=np.arange(-5.0,5.0,0.1)
y=step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.xlim(-6,6)
plt.close('all')


# < 활성화 함수 - 신경망의 시그모이드 함수 (sigmoid function)
# h(X) = 1 / (1 + exp**(-x))
import math
math.exp(1)
# 시그모이드 구현
def sigmoid(x):
    return 1 / (1 +np.exp(-x))

x=np.array([-1.0, 1.0, 2.0])
sigmoid(x)

t=np.array([1,2,3])
1.0 + t
1.0/t
# 그래프
x =np.arange(-5,5,0.1)
y=sigmoid(x)
plt.xlim(-6,6)
plt.ylim(-0.1,1.1)
plt.plot(x,y)
plt.close()


# < 활성화 함수 - 신경망의 ReLU 함수 Rectified Linear Unit 렐루
'''
h(x) = x , x>0
       0, x<=0
입력이 0을 넘으면 그대로 출력, 입력이 0이하면 0으로 출력 
'''
def relu(x):
    return np.maximum(0,x)

x =np.arange(-5,5,0.1)
y=relu(x)
plt.xlim(-6,6)
plt.ylim(-0.1,1.1)
plt.plot(x,y)
plt.close()


# < 다차원 배열의 계산
import numpy as np
A = np.array([1,2,3,4])
print(A)
np.ndim(A)  # 차원 반환  >>>  1           # 1차원의 원소4개
A.shape     # 튜플로 (n,m) 반환  >>> (4,)
A.shape[0]  # >>> 4
A.shape[1]  # >>> error

B = np.array([[1,2],[3,4],[5,6]])
print(B)
np.ndim(B)  # 2차원
B.shape   # (3,2) 3행 2열

c = np.array([[1,2,7],[3,4,8],[5,6,9]])
np.ndim(c)   # 2차원
c.shape
# 3차원 배열은 다른 함수 사용

# 행렬의 곱
A=np.array([[1,2],[3,4]])
B=np.array([[5,6],[7,8]])
np.dot(A,B)

A=np.array([[1,2],[3,4],[5,6]])
B=np.array([5,6])
np.dot(A,B)

# < 신경망에서의 행렬곱
X =np.array([1,2])
X.shape
W=np.array([[1,3,5],[2,4,6]])
Y=np.dot(X,W)
print(Y)

# < 신경망 구현하기 3층 신경망
'''
p85 참조 
W12**(1) : 다음층의 1번째 뉴런(a1)의 1, 앞층의 두번째 뉴런(x2)의 2, (1):1층의 가중치 
A**(1) = X * W**(1) + B**(1)

X= np.array([1,0.5]) 는 (1,2)행렬 이지만 X.shape는 (2,)로 나옴! 헷갈리지 않기
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]]) -> (2,3)행렬, W1.shape=(2,3)로 그대로 나옴 
'''

X= np.array([1,0.5])
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

print( X.shape,W1.shape, B1.shape)
A1= np.dot(X,W1)+B1
print(A1)

Z1=sigmoid(A1)  #활성화 함수 적용
print(Z1)  # 0층에서 1층으로 나온 출력

# 1층에서 2층가기
W2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
B2=np.array([0.1,0.2])
print(Z1.shape, W2.shape, B2.shape)
A2=np.dot(Z1,W2) +B2
Z2=sigmoid(A2)
print(Z2)

# 2층에서 마지막 3층(출력층)
# 항등함수(입력을 그대로 출력), 굳이 정의할필요는 없다.   # 회귀)항등함수, 2클래스분류)시그모이드, 다중클래스분류)소프트맥스함수 일반적
def identity_function(x):
    return x

W3=np.array([[0.1,0.3],[0.2,0.4]])
B3=np.array([0.1,0.2])

A3=np.dot(Z2,W3)+B3
Y= identity_function(A3)  # or Y=A3 써도 똑같음
print(Y)

# < 신경망 구현 정리
def init_network():
    network ={}
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1,W2,W3=network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']

    a1=np.dot(x,W1) +b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2) +b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3) +b3
    y=identity_function(a3)

    return y

network = init_network()
x=np.array([1,0.5])
y=forward(network,x)
print(y)

# << 출력층 설계하기 >
# < 항등함수와 소프트맥스 함수 구현하기

# <항등함수 identity function : 입력을 그대로 출력
# a1 입력시 y1=a1


# 소프트맥스(다중클래스 분류에서 사용) softmax function
'''
yk= exp(ak)/i=1~n까지 합(exp(ai))
n: 출력층의 뉴런의 개수
yk: k번째 출력
ak:  입력신호 ak

출력층의 각 뉴런이 모든 입력신호에서 영향을 받음 (분모)
'''
a=np.array([0.3,2.9,4])
exp_a=np.exp(a)
sum_exp_a =np.sum(exp_a)

print(a,exp_a, sum_exp_a)
y=exp_a/sum_exp_a
print(y)

# < 소프트맥스 함수 구현
def softmax1(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y

# < 소프트 맥스 함수 - 오버플로 주의 (너무 큰수에 대해 처리하지 못함 e**1000 = inf)
# 분자 분모에 C를 곱해주고, exp()안으로 넣으면 logC로 변경, locC=C'로 바꿈     보통 C를 입력신호중 최대값으로, a에서 빼줌
a= np.array([1010,1000,990])
np.exp(a)/np.sum(np.exp(a))

c= np.max(a)
print(a-c)

np.exp(a-c)/np.sum(np.exp(a-c))

def softmax(a):
    c=np.max(a)
    exp_a= np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y

# 소프트 맥스 함수 특징
# a=np.array([0.3,2.9,4]) 로 지정하고 y=softmax(a)로 해도 됨
y = softmax([0.3,2.9,4])
print(y)
np.sum(y)
'''
반환되는 출력값는 0~1사이의 수  -> 확률로 해석가능
다 더하면 1
y[2] =0.73659691 : 73.6%의 확률로 2번째 클래스이다.

* 주의할 점은 소프트맥스 함수를 적용해도 각 원소들의 대소관계는 변하지 않음
y=exp(x)는 단조증가함수(a<=b 일때 f(a)<= f(b)인 함수) 이기 때문에 a에서 2번째가 제일 크면 y도 2번째가 제일 큼

* 신경망으로 분류할때는 출력층의 소프트맥스 함수 생략해도 됨 

'''

# < 출력층의 뉴런수 정하기>
# 뉴런수는 풀려는 문제에 맞게 적절히 정해야함
# 입력 이미지를 숫자 0부터 9까지중 하나로 분류하려면 출력층의 뉴련을 10개로 지정


# << 손글씨 숫자 인식
# 여기서는 학습과정 생략, 추론과정만 구현
# 추론과정 = 신경망의 순전파forward propagation , 매개변수를 사용하여 입력데이터를 분류

# <MNIST dataset>
# 28*28크기의 이미지
import sys, os
os.getcwd()
sys.path.append(os.pardir)
from dataset.mnist import load_mnist  # script가 있는 폴더에 dataset 옮겨두기


# load_mnist 의 매개변수 =[flatten : 1차원 배열로 바꾸기, normalize = 0~1사이로 정규화, one_hot_label = false: 숫자형태로 레이블 저장]
(x_train, t_train),(x_test,t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)
print(x_train.shape, t_train.shape, x_test.shape, t_test.shape)

# 이미지 확인
import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))  # np로 저장된 이미지 데리터를 PIL용 데이터 객체로 변환
    pil_img.show()

(x_train, t_train),(x_test,t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)

img=x_train[0]
label=t_train[0]
print(label)  # 5

print(img.shape)  #flatten =True -> (784,)  28*28=784
img=img.reshape(28,28) # 원래 이미지 모양으로 변형
print(img.shape)  # reshape -> (28,28)

img_show(img)


#< 신경망의 추론처리 >
'''
입력층 784개, 출력층 10개
1번째 은닉층 50, 2번째는 100으로 임의로 정함
sample_weight.pkl에 저장된 학습된 가중치 매개변수 읽기 : w,b가 딕셔너리로 저장되어 있음 
'''
# 사전 준비
import pickle
# 소프트맥스 함수
def softmax(a):
    c=np.max(a)
    exp_a= np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    y=exp_a/sum_exp_a

    return y
# 시그모이드
import math
math.exp(1)
def sigmoid(x):
    return 1 / (1 +np.exp(-x))

# 신경망 구현

def get_data():
    (x_train, t_train), (x_test,t_test) = load_mnist(normalize=True, flatten=True,one_hot_label=False)
    return x_test, t_test
def init_network():
    with open('sample_weight.pkl','rb') as f:
        network=pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1=np.dot(x, W1) +b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2) + b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3) + b3
    y=softmax(a3)

    return y

x,t= get_data()
network=init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 가장 확률이 높은 원소의 인덱스!를 얻는다
    if p == t[i]:
        accuracy_cnt +=1

print('Accuracy:' + str(float(accuracy_cnt)/len(x)))  #>>>0.9352


# <배치처리
# 입력 데이터와 가중치 매개변수의 형상에 주의해서 살펴보기
x, _ = get_data()
network=init_network()

W1, W2, W3 = network['W1'], network['W2'], network['W3']
print(x.shape, x[0].shape, W1.shape, W2.shape, W3.shape)
''' 
>>> (10000, 784) ->  (784,) (784, 50) (50, 100) (100, 10)
0번째 이미지(원소 784개)가 10개원소의 1차원배열로 출력되는 흐름 
'''

# 배치 batch : 묶음!  입력 데이터를 하나로 묶음 / 큰배열을 한번에 처리하는게 더 효율적이고 빠름
# 100장씩 묶어서 배치처리
x, _ = get_data()
network=init_network()

batch_size = 100
accuracy_cnt=0

for i in range(0,len(x), batch_size):
    x_batch = x[i : i+batch_size]     #x[0:100],x[100:200]..으로 계속 100장씩 묶어서 꺼냄
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)    # 높은 확률을 가진(최대값) 원소의 인덱스 반환
    accuracy_cnt += np.sum(p==t[i : i+batch_size])   # p==t : bool배열로 만들어서True가 몇개인지 셈

print('Accuracy:' + str(float(accuracy_cnt)/len(x)))

# range(start, end , step) : start에서 end-1까지 step간격으로 증가하는 리스트반환
list(range(0,10))
list(range(0,10,3))

# np.argmax
x=np.array([[0.1,0.8,0.1],[0.3,0.1,0.6], [0.2,0.5,0.3],[0.8,0.1,0.1]])
y=np.argmax(x, axis=1)
print(y)
# >>> [1 2 1 0] : 0번 배열에서 1번째 원소인 0.8이 가장 크므로 1 반환

# p==t bool 배열
y=np.array([1,2,1,0])
t=np.array([1,2,0,0])
print(y==t)
np.sum(y==t)


# chapter 4) 신경망 학습
'''
학습 : 훈련 데이터로부터 가중치 매개변수의 최적값을 자동으로 획득하는 것
손실함수 loss function: 신경망이 학습할 수 있도록 해주는 지표 
-> 손실함수의 결과값을 가장 작게 만드는 가중치 매개변수를 찾는 것이 목표   -> 경사법 사용! 

* 퍼셉트론도 직선으로 분리가능하면 데이터로부터 자동으로 학습 가능 (퍼셉트론 수렴정리로 증명됨)
* 비선형 분리문제는 자동학습 불가 

* 알고리즘을 밑바닥부터 설계하는 것이 아니라, 데이터에서 특징 feature을 추출해서 학습
(보통 이미지를 벡터로 변환하고 변환된 벡터로 학습 진행) -> 적절한 특징은 사람이 잘 설계해야함 

* 신경망은 종단간 기계학습 end-to-end ML 처음부터 끝까지 데이터(입력)부터 목표한 결과(출력)까지 사람의 개입이 없다.

* 손실함수 : 현재의 상태를 하나의 지표로 표현, 지표를 가장 좋게 만들어주는 가중치 매개변수의 값을 탐색
- 평균제곱오차MeanSquaredError(MSE) /  교차 엔트로피 오차cross entropy error(SEE) 
'''

# < 평균제곱오차 MSE 수식>
'''
E= 1/2 시그마(아래k) (yk-tk)**2
yk: 신경망의 출력(예측값)
tk: 정답 레이블
k:데이터의 차원수 
'''
# 손글씨 숫자 인식의 예
y=[0.1,0.05,0.6,0,0.05,0.1,0,0.1,0,0]  #0~9까지 순서대로 일때의 해당하는 값(확률) / 여기선 0.6이 제일 크고, 숫자2에 해당함
t=[0,0,1,0,0,0,0,0,0,0] # 정답 레이블 원핫인코딩형태


# < 평균제곱오차 MSE func
# 오차가 적을수록 제대로 예측한 것
def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

mean_squared_error(np.array(y),np.array(t))
#>>>  0.0975 : MSE 값  -->  0.09의 오차

# 정답이 2인데, 7이 확률이 높다고 가정하면
y=[0.1,0.05,0.1,0,0.05,0.1,0,0.6,0,0]
t=[0,0,1,0,0,0,0,0,0,0]
mean_squared_error(np.array(y), np.array(t))
#>>> 0.5975 의 오차
# 즉 0.09의 오차가 더 작으므로 정답에 가까울 것이라고 생각할 수 있다.


# < 교차 엔트로피 오차 SSE 수식
'''
E = -시그마(아래k)tk *logyk
yk: 신경망의 출력(예측)
tk: 정답레이블(정답만 1, 나머지 0) --> 즉 실질적으로 정답일때 (tk=1일때 ) yk의 자연로그를 계산하는 식이 됨 

* 만약 정답레이블이 2일때 
yk=0.6 -> tk=-log0.61 = 0.51 : 오차가 0.51
yk=0.1 -> tk=-log0.1 = 2.3  : 오차가 2.3 
 --> 0.51의 오차가 더 작으므로 정답에 가깝다 라고 말할 수 있음 
'''
import math
math.log(1,10)   # log1 = 0 , log10=1, log100=2,    log0 = error

# 자연로그 그래프
plt.figure()
plt.grid()
x=np.arange(0,1,0.01)
y=np.log(x)
plt.plot(x,y)
plt.close()
# x=1 -> y=0 / x(정답에 해당하는 출력값 0.2-> 0.6)가 커질수록 y(오차 2.3->0.51)는 0에 다가면서 작아짐
# 정답인 x=1이면 오차인 y=0 : 매우 정확하게 예측했다!


# < 교차 엔트로피 func1 , 밑에 n차원용(완성)func 있음  >
# delta는 아주 작은 값 / log(0)은 -inf 라서 더이상 진행안됨
def cross_entropy_error1(y,t):
    delta=1e-7
    return -np.sum(t* np.log(y+delta))

t=[0,0,1,0,0,0,0,0,0,0]
y=[0.1,0.05,0.6,0,0.05,0.1,0,0.1,0,0]
cross_entropy_error1(np.array(y),np.array(t))  #>>> 0.5108

# 정답이 2인데, 7이 확률이 높다고 가정하면
y1=[0.1,0.05,0.1,0,0.05,0.1,0,0.6,0,0]
cross_entropy_error1(np.array(y1), np.array(t))  #>>> 2.302

# 즉 첫번째 추정이 정답일 가능성이 높다 ( 오차가 적기 때문)  MSE의 판단과 동일

# < 미니배치 학습
'''
훈련 데이터에 대한 손실함수의 값을 구하고 그 값을 최대한 줄여주는 매개변수 선택
훈련데이터가 100개면 손실함수의 100개 합을 지표로 삼아야 함 -> 훈련 데이터 모두에 대한 손실함수의 합을 구해야함

* 데이터가 N개 일때 교차엔트로피 에러함수
E = -1/N * 시그마(아래 N)* 시그마(아래 k) * t(밑지수nk)log(y(밑지수nk)) 
tnk = n번째 데이터의 k번째 값
마지막에 N으로 나누어 정규화 -> 평균 손실함수를 구함 

* 손글씨 데이터는 6만개. 일부만 골라 근사치로 이용하여 학습 : 미니배치mini-batch 학습
'''

import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape, t_train.shape)

# < 무작위로 10장만 뽑기 -> np.random.choice()
train_size = x_train.shape[0]
batch_size=10
batch_mask = np.random.choice(train_size, batch_size)
batch_mask  # 무작위로 뽑은 범위내 숫자
x_batch=x_train[batch_mask]  # 마스크를 인덱스로 뽑아내기
x_batch.shape
t_batch = t_train[batch_mask]

# < 배치용) 교차 엔트로피 오차 구현하기 cross_entropy_error func
# 데이터가 1개, 배치 둘다 가능 (데이터 1개일 경우, reshape로 형상 바꿔줌)
# 1e-7은 delta 매우 작은 값

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)

    batch_size = y.shape[0]
    return -np.sum(t* np.log(y+1e-7)) / batch_size   #정규화 및 1장당 평균 교차엔트로피 구해줌

# 정답 레이블이 원핫인코딩이 아니라 숫자로 표현된 레이블일 경우
# np.arange(batch_size)는 0부터 배치사이즈-1까지 배열 생성   5일경우 [0,1,2,3,4]
# t= 정답레이블은 [1,7,0,9,4..]로 되어있으니 y[np.arange(batch_size), t] 는  [ y[0,1], y[1,7]...] 인 넘파이 배열 생성

def CEE_not_one_hot(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size


# <손실함수를 설정하는 이유
'''
손실함수의 값을 가능한 작게 하는 매개변수의 값을 찾음
이때 매개변수의 미분(정확히는 기울기)을 계산하고 이를 단서로 매개변수의 값을 서서히 갱신하는 과정 반복

가중치 매개변수의 값을 아주 조금씩 변화시켰을때 손실함수가 어떻게 변하는지 보는 것이 '가중치매개변수의 손실함수의 미분'
기울기가 음수면 가중치를 양의 방향으로 변화시켜 손실함수의 값을 줄임(기울기 양수면 음의방향으로 변화)
기울기가 0이 되면 어느쪽으로 가도 변하지 않으므로 매개변수 갱신 중단

* 정확도는 대부분의 장소에서 기울기가 0, 매개변수를 갱신할 수 없기 때문 예) 계단 함수   
--> 그래서 시그모이드! (기울기가 항상0이 아님)
'''

# <<수치 미분>> numerical differentiation : 아주 작은 차분으로 미분하는 것
# 해석적미분 = 진정한 미분 = 수학시간에 배운 미분
# y=x**2 을 x=2일때 해석적 미분을 하면 2x = 4 가 되는 것.

# 수치 미분  lim h->0 (f(x+h)-f(x)) / h
# 잘못 구현한 미분
'''
h를 작은 값으로 쓰고 싶어서 매우 작은값 지정했지만 반올림오차(rounding error)문제일으킴 
-> 소수점 8자리 이하가 생략되어 계산결과가 다름

def numerical_diff_wrong(f,x):
    h= 10e-50
    return (f(x+h)-f(x))/ h

np.float32(1e-50)
# >>> 0.0 

* 진정한 미분은 x위치에서의 함수의 기울기(접선)
위의 방식은 (x+h)와 x 사이의 기울기 이기 때문에 오차가 발생함 : 수치미분 

* 그래서 중심/중앙 차분을 계산함 (x+h)와 (x-h)의 기울기를 구함 
(위의 방식은 전방차분이라고 함)
'''
# 개선된 수치미분(근사로 구한 접선)
def numerical_diff(f,x):
    h=1e-4
    return (f(x+h) - f(x-h)) /(2*h)

# <수치미분의 예

def func1(x):
    return 0.01*x**2 + 0.1*x

import numpy as np
import matplotlib.pyplot as plt
x=np.arange(0,20,0.1)
y=func1(x)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y)
plt.close()

numerical_diff(func1,5)   #>>> 0.1999999999990898     / 해석적 미분: 0.2
numerical_diff(func1,10)  #>>> 0.2999999999986347     / 해석적 미분 : 0.3

# < 편미분 >  δf/δx0, δf/δx1 처럼 변수가 2개, 해당하는 변수는 그대로 미분, 나머지는 상수 처리
# f(x0,x1)=x0**2 + x1**2 구현 func2
def func2(x):
    return x[0]**2 + x[1]**2  # or np.sum(x**2), x가 넘파이 배열일때
func2([1,2])

# 위의 함수를 x0에 대해 편미분하여라 (x0=3, x1=4일때)
def function_tmp1(x0):
    return x0*x0 +4.0**2.0
numerical_diff(function_tmp1, 3)  #>>> 6.00000000000378     / 해석적 편미분 = 6 (=2* x0)

# x1에 대해 편미분
def function_tmp2(x1):
    return 3**2 +x1*x1
numerical_diff(function_tmp2, 4)  #>>>  7.999999999999119   / 해석적 편미분 = 8 (=2 * x1 )


# < 기울기 Gradient >  : 모든 변수의 편미분을 벡터로 정리한 것 (δf/δx0, δf/δx1)
def numerical_gradient(f,x):    # x는 np배열 ,  각 원소에 대해 수치미분 구함
    h = 1e-4  # == 0.0001
    grad=np.zeros_like(x)   # x와 형상shape이 같은 배열 생성(원소는 모두 0)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx]=tmp_val + h   # f(x+h) 계산
        fxh1=f(x)

        x[idx]=tmp_val-h     # f(x-h)계산
        fxh2 = f(x)

        grad[idx] = (fxh1-fxh2) / (2*h)
        x[idx] = tmp_val   # 값 복원
    return grad

# (3,4), (0,2), (3,0)에서의 기울기 구해보기 (f(x0,x1) = x0^2 + x1^2 일 때 ==func2
numerical_gradient(func2, np.array([3.0,4.0]))  # 3.0이라고 써야함
numerical_gradient(func2, np.array([0.0,2.0]))
numerical_gradient(func2, np.array([3.0,0.0]))

# gradient_2d.py 실행하면 그래프 나옴
# 각 지점에서 낮아지는 방향을 가리킴(가운데 한점), 또 가운데에서 멀수록 화살표의 크기가 커짐 (기울기가 가장 낮은 장소라고 말할 순 없음 뒤에 나옴)
# 즉 기울기가 가리키는 쪽은 함수의 출력값을 가장 크게 줄이는 방향!!


# < 경사(하강)법 > Gradient Descent method
'''
기울기가 가리키는 방향이 실제로 최소값이 아니라, 극소값 or 안장점일수 있기 때문
복잡하고 찌그러진 데이터가 대부분 고원pletuar플레토 : 학습이 진행되지 않는 정체기 에 빠질수 있기 때문에 
n: eta 학습률(lr) 매개변수를 얼마나 갱신할지 정함
step_num: 얼마나 반복할지 
'''


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)

init_x=np.array([-3.0, 4.0])
gradient_descent(func2, init_x=init_x, lr=0.1, step_num=100)
#>>> 거의 0에 가까운 결과

# 학습률이 너무 크거나 작을 때 lr=10.0, 1e-10
init_x=np.array([-3.0, 4.0])
gradient_descent(func2, init_x=init_x, lr=10.0, step_num=100)   # 너무 큰값으로 발산

init_x=np.array([-3.0, 4.0])
gradient_descent(func2, init_x=init_x, lr=1e-10, step_num=100)  # 거의 갱신되지 못하고 끝남


# 신경망에서의 기울기 : 가중치 매개변수 W에 대한 손실함수의 기울기 ( 행렬로 나타나 있으므로 편미분 필요)
# 실제로 기울기 구하기
import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)  # 정규분포로 초기화
    def predict(self,x):
        return np.dot(x,self.W)
    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)

        return loss


net = simpleNet()
print(net.W)

x= np.array([0.6,0.9])
p= net.predict(x)
print(p)

np.argmax(p) # 최대값의 인덱스
t=np.array([0,0,1]) # 정답 레이블
net.loss(x,t)

f= lambda w:net.loss(x,t)
'''
lambda 아래처럼 쓸 수 있음 
def f(w):
    return net.loss(x,t)

'''
from common.gradient import numerical_gradient
dW = numerical_gradient(f, net.W)  # 위의 함수 불러와야 됨
print(dW)

# < 2층 신경망 클래스 구현하기>
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

net=TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape,net.params['b1'].shape,net.params['W2'].shape,net.params['b2'].shape)

x=np.random.rand(100,784) # 100장 분량 더미 데이터 입력
y=net.predict(x)

# 더미 데이터 입력/정답레이블 100장 분량
x=np.random.rand(100,784)
t=np.random.rand(100,10)

grads=net.numerical_gradient(x,t) # 기울기 계산
print(grads['W1'].shape, grads['b1'].shape, grads['W2'].shape, grads['b2'].shape)

# < 미니배치학습 구현하기>
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100  # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()




# chapter 5) 오차역전파법(역전파(법), backpropagation)


def gradient(self, x, t):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']
    grads = {}

    batch_num = x.shape[0]

    # forward
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2)

    # backward
    dy = (y - t) / batch_num
    grads['W2'] = np.dot(z1.T, dy)
    grads['b2'] = np.sum(dy, axis=0)

    da1 = np.dot(dy, W2.T)
    dz1 = sigmoid_grad(a1) * da1
    grads['W1'] = np.dot(x.T, dz1)
    grads['b1'] = np.sum(dz1, axis=0)

    return grads


# chapter 5) 오차역전파법 backpropagation  : 가중치 매개변수의 기울기를 효율적으로 계산하는 방법
'''
덧셈노드는 그대로
곱셈노드는 x,y를 바꿔서
exp도 그대로 -> 하나의 sigmoid로 변환가능 

* 오차역전파법으로 기울기를 계산하며, 4장의 수치함수는 기울기 검증으로 많이 사용
'''

# < 단순한 계층 구현하기 >
# 곱셈노드는 MulLayer / 덧셈노드는 AddLayer

# < 곱셈 계층
class MulLayer:
    def __init__(self):
        self.x=None
        self.y=None
    def forward(self,x,y):
        self.x= x
        self.y= y
        out =x*y
        return out
    def backward(self,dout):
        dx=dout *self.y
        dy=dout *self.x
        return dx,dy

# 사과 쇼핑의 변수
apple =100
apple_n = 2
tax=1.1

# 계층 선언
mul_apple_layer = MulLayer()
mul_tax_layer= MulLayer()

# 순전파
apple_price =mul_apple_layer.forward(apple, apple_n)
price = mul_tax_layer.forward(apple_price, tax)
print(price)

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_n = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_n, dtax)


# < 덧셈 계층
class AddLayer:
    def __init__(self):
        pass  # 초기화 필요없음
    def forward(self,x,y):
        out=x+y
        return out
    def backward(self,dout):
        dx=dout*1
        dy=dout*1
        return dx, dy

# < 사과2개 귤3개 계산그래프 구현 >
# 변수
apple =100
apple_n = 2
orange=150
orange_n = 3
tax=1.1

# 계층 생성
mul_apple_layer=MulLayer()
mul_orange_layer=MulLayer()
add_apple_orange_layer =AddLayer()
mul_tax_layer= MulLayer()

# 순전파
apple_price= mul_apple_layer.forward(apple, apple_n)
orange_price=mul_orange_layer.forward(orange, orange_n)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

# 역전파
dprice=1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dorange, dorange_n = mul_orange_layer.backward(dorange_price)
dapple, dapple_n = mul_apple_layer.backward(dapple_price)

print(price)
print(dapple_n, dapple, dorange, dorange_n,dtax)

# << 활성화 함수 계층 구현하기 - relu, sigmoid >
# < ReLU : x>0 일때 x출력 , 그외는 0출력>
# mask : T/F의 넘파이 배열을 출력 0이면 True, 1이면 False **
class Relu:
    def __init__(self):
        self.mask = None
    def forward(self,x):
        self.mask = (x <=0)
        out=x.copy()
        out[self.mask] = 0
        return out
    def backward(self,dout):
        dout[self.mask]=0
        dx=dout
        return dx


x= np.array([[1.0, -0.5],[-2.0,3.0]])
print(x)

mask=(x<=0)
print(mask)

# < sigmoid  y= 1/(1+exp(x^(-1))) >
'''
1) y=x^(-1) 을 미분하면 -1x^(-2) = -y^2
2) 덧셈 노드 그대로
3) exp 노드 : y=exp(x) 그대로 수행
4) 곱셈 노드 : x,y바꿔서 곱함

* 역전파에서 ) 순전파의 출력 y로 정리해서 -> 간소화 
dL/dy(상류값) * (y^2*exp(-x))  
= dL/dy(상류값) * y(1-y) 
'''

# 순전파 출력을 out에 보관했다가 역전파때 그 값을 사용
class Sigmoid:
    def __init__(self):
        self.out =None
    def forward(self,x):
        out=1 / (1+np.exp(-x))
        self.out= out
        return out
    def backward(self,dout):
        dx=dout * (1.0 -self.out) *self.out
        return dx

# << Affine/ Softmax 구현하기 >
# < Affine게층 - 1개의 X(입력 데이터) >
# affine transformation 어파인 변환 = 행렬의 곱  -> 그래서 어파인 계층이라고 사용
# 순전파에서 가중치 신호의 총합을 계산하기 때문에 행렬의 곱 np.dot() 사용
# 차원의 원소수를 일치시켜야함 ( 역전파시) 전치행렬 Transpose 사용 W,X에)

X=np.random.rand(2)
W=np.random.rand(2,3)
B=np.random.rand(3)

print(X.shape, W.shape, B.shape)
Y = np.dot(X,W) + B
print(Y)

# < 배치용 Affine 계층 - N개의 입력 데이터  + 배치(묶음)>

X_dot_W =np.array([[0.0,0.0,0.0],[10.0,10.0,10.0]])
B= np.array([1.0,2.0,3.0])

X_dot_W
X_dot_W+B

dY= np.array([[1,2,3],[4,5,6]])
dY

dB=np.sum(dY, axis=0)   # axis=0 열방향(아래로), axis=1 행방향(옆으로)
dB

dB1=np.sum(dY, axis=1)   # axis=0 열방향(아래로), axis=1 행방향(옆으로)
dB1


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx

# < softmax-with-loss 계층 :입력값을 정규화 하여 출력(확률로)
# affine 계층의 출력은 점수 score
# softmax -> 교차 엔트로피 오차 -> 출력 (간소화한 계산 그래프 p177)


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실함수
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태)

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

# << 신경망 학습의 전체 그림/ 순서 >>
'''
0) 전체: 신경망에는 적응가능한 가중치/편향이 있고, 이것을 훈련 데이터에 적응하도록 조정하는 과정을 학습

1) 미니배치 : 훈련 데이터중 일부를 무작위로 가져옴 (선별한 데이터를 미니배치라고 함)
 --> 미니배치의 손실함수 값을 줄이는 것이 목표
2) 기울기 산출 : 미니배치의 손실함수 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구하고, 
 --> 손실함수의 값을 가장 작게 하는 방법을 제시 
 --> 여기서 오차역전파법을 사용 ( 수치미분은 느리고, 기울기 검증에서 사용) 
3) 매개변수 갱신 : 가중치 매개변수를 기울기 방향으로 아주 조금 갱신 ( 경사 하강법 )
4) 반복 : 1~3단계를 반복 
'''

# << 오차역전파법으로 구한 기울기 검증 Gradient check >
# 오차역전파법과 수치미분으로 구한 기울기를 비교하는 작업 : 기울기 확인 Gradient check


import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from common.two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# 각 가중치의 절대 오차의 평균을 구한다.
for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
'''
W1:2.0255837762429904e-10
b1:1.136886147067506e-09
W2:6.923514885401327e-08
b2:1.3766053351266238e-07

대부분의 오차가 매우 작음 
즉 오차역전파법으로 제대로 기울기를 구했다 라고 말할 수 있음 

'''



# << 오차역전파법을 적용한 신경망 구현하기 >
import sys, os

sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from common.two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch)  # 오차역전파법 방식(훨씬 빠르다)

    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)


# chapter 6) 학습관련 기술들

# << 매개변수 갱신
# 매개변수의 최적값을 찾는 문제 : 최적화 optimizer
# < 확률적 경사하강법 SGD: 매개변수의 기울기를 구해 기울어진 방향으로 매개변수 갱신 반복

class SGD:
    def __init__(self, lr=0.01):
        self.lr=lr
    def update(self, params, grads):
        for key in params.keys():
            params[key]-= self.lr *grads[key]
# 문제에 따라서 비효율 적일 수 있음 한쪽의 변화폭이 클 경우 심한 지그재그 형태

# < 모멘텀 Momentum : 운동량, v=velocity속도, 물체가 아무런 힘을 받지 않을 때 서서히 하강시킴
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
# 지그재그 이지만 부드럽게 곡선으로 변화(SGD보다 x축 방향으로 빠르게 다가감)

# < AdaGrad: 매개변수를 갱신할때 학습률을 조정(많이 움직인 원소는 학습률이낮아지게) -> 원소마다 학습률적용이 다름
# 어느순간 갱신량이 0이 되면 전혀 변화가 없어짐-> 대신 RMSProp를 사용함(새로운 기울기 정보를 크게 반영 지수이동평균EMA)

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
# 최솟값을 향해 효율적으로 움직임 (y는 초반에 크게 움직이지만 큰폭으로 작아지도록 조정되어 거의 지그재그가 없음


# < RMSProp 참고 >

class RMSprop:

    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


# < Adam : 모멘텀과 에이다그래드를 융합 -> 편향 보정이 진행됨 부드러운 곡선 보양이지만 최소값에 가깝게
class Adam:
    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

# 보통 SGD, adam을 많이 사용함 , SGD가 가장 느림 하지만 매개변수들에 따라 결과가 다 달라짐


# << 가중치의 초기값 : 가중치 감소 weight decay기법 : 가중치 값을 작게 하여 오버피팅 일어나지 않게
# 초기값이 0이라면 갱신이 되어도 계속 0이므로 의미가 없다. -> 초깃값을 무작위로 설정해야함




# << 배치 정규화
# << 바른학습을 위해 - 오버피팅/가중치감소/드롭아웃
# << 적절한 파라미터값 찾기 - 검증데이터/하이퍼 파라미터 최적화


# chapter 7) 합성곱 신경망 CNN


# < 4차원 배열
# (N, C, H, W) ==(데이터 개수=배치크기, 채널개수, 높이, 너비)
x = np.random.rand(10,1,28,28)
x.shape

x[0].shape  # >>> (1,28,28) 첫번째 데이터(N=0)
x[1].shape   #>>> (1,28,28) 두번째 데이터 (N=1)
x[0,0]  # == x[0][0] >>> 첫번째 데이터의 첫 채널의 공간 데이터를 볼 수 있음

# < im2col (image to columns)  common/util.py 참조
'''
im2col(input_data, filter_h, filter_w, stride =1, pad=0)
input_data = (데이터, 채널수, 높이, 너비)의 4차원 배열로 이뤄진 입력데이터
filter_h = 필터의 높이
filter_w = 필터의 너비
stride 스트라이드(간격), pad 패딩(확대할 픽셀크기)

im2col을 사용하면 2차원 배열로 전개(나열)

'''
import sys, os
sys.path.append(os.pardir)
from common.util import im2col

x1= np.random.rand(1,3,7,7)  # 데이터 1개(배치크기=1), 채널3개, 높이7, 너비7
col1 = im2col(x1, 5,5, stride=1, pad=0)  # ( 입력 데이터(안에 데이터 개수, 채널수, 높이 너비 있음), 채널높이, 채널너비, s, p)
print(col1.shape)

x2 = np.random.rand(10,3,7,7) # 데이터 10개
col2 = im2col(x2, 5,5, stride=1, pad=0)
print(col2.shape)

# < 합성곱 계층 구현 forward
class Convolution :
    def __init__(self, W,b, stride =1, pad=0):
        self.W=W
        self.b= b
        self.stride = stride
        self.pad= pad

    def forward(self,x):
        FN, C, FH, FW = self.W.shape
        N,C,H,W = x.shape
        out_h = int(1+(H +2*self.pad - FH) / self.stride)
        out_w = int(1 + (W+ 2*self.pad - FW)/ self.stride)

        # 입력데이터를 전개하고 필터도 reshape.T를 활용해 2차원으로 전개 그리고 곱함
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T   # 필터 전개(나열)
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)
        '''
         * reshape 
        reshape에 -1을 지정하면 다차원 배열의 원소수가 변환후에도 똑같이 유지되도록 적절히 묶어줌 
        W = (10,3,5,5)면 원소의 수가 총 750개, 여기에 reshape(10,-1)로 지정하면 750개의 원소를
        10개의 묶음으로 즉 shape가 (10, 75)인 배열로 만들어줌 
        
        * transpose
        다차원 배열의 축 순서를 바꿔줌
        (N,H,W,C)의 인덱스는 0,1,2,3   --> (N,C,H,W) 로 바꿔야 하니 0,3,1,2! 
        '''

        return out

# 합성곱 계층의 역 전파는 common/utils의 col2im에 구현됨 (합성곱계층의 역전파는 layer.py에도 있음)
# 구현방식은 Affine의 역전파와 똑같음

# < 풀링 계층 구현하기 : im2col사용해서 입력데이터 전개  forward 만!

# 하지만 풀링은 채널쪽이 독립적이라는 점이 합성곱계층과 다름 (풀링 적용영역을 채널마다 독립적으로 전개)
# p.248 ) 풀링의 shape 모양대로 입력데이터를 전개해서 다 붙인다음에, 최댓값을 구하고, reshape 해서 풀링 모양대로 성형

class Pooling :
    def __int__(self, pool_h, pool_w, stride =1, pad=0):
        self.pool_h = pool_h
        self.pool_w= pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N,C,H,W = x.shape
        out_h = int(1+ (H-self.pool_h)/ self.stride)
        out_w = int(1+ (W-self.pool_h)/ self.stride)

        # 1) 우선 데이터 전개
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        # 2) 최댓값 구하기
        out = np.max(col, axis =1)  # 행방향 (가로)

        # 3) 성형(적절한 모양으로 변형)
        out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

        return out
# 풀링 계층의 역전파는 relu계층 구현할때 사용한 max역전파 참고
# 풀링 전체 구현은 common/layer.py에 있음

# < CNN 구현하기  - 손글씨
'''
SimpleConvNet 구현 
입력 -> conv/relu/pooling -> affine/relu -> affine/softmax -> 출력 

초기화 받는 인수
input_dim = 입력데이터(C, H,W)의 차원
conv_param - 합성곱 계층의 하이퍼파라미터(딕셔너리) , 아래는 딕셔너리의 키
    filter_num 필터수  
    filter_size 필터크기
    stride 스트라이드
    pad 패딩
    hidden_size 은닉층(완전연결)의 뉴런수
    output_size 출력층(완전연결)의 뉴런수
    weight_int_std 초기화때의 가중치 표준편차

'''

class SimpleConvNet:
    def __init__(self, input_dim = (1,28,28), conv_param = {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size = 100, output_size =10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad= conv_param['pad']
        filter_stride =conv_param['stride']
        input_size = input_dim[1]
        conv_output_size= (input_size-filter_size + 2*filter_pad) / filter_stride+1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # 매개변수 초기화 부분
        self.params ={}
        self.params['W1']=weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1']=np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size,hidden_size)
        self.params['b2']= np.zeros(hidden_size)

        # 계층 생성 - CNN 을 구성하는 계층들
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """기울기를 구한다（수치미분）.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

    # 오차역전파법 -기울기 구현
    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i + 1)]
            self.layers[key].b = self.params['b' + str(i + 1)]


# CNN 시각화
'''
1번째 층의 합성곱계층의 가중치(W) = (30,1,5,5) 필터 30개, 채널1개, 5*5크기 (1채널의 회색조 이미지로 시각화가능하다는 뜻)

visualize_filter.py 참조 (학습전 후 가중치 비교) 
'''


def filter_show(filters, nx=8, margin=3, scale=10):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


network = SimpleConvNet()
# 무작위(랜덤) 초기화 후의 가중치
filter_show(network.params['W1'])

# 학습된 가중치
network.load_params("params.pkl")
filter_show(network.params['W1'])


# chapter 8) 딥러닝
# ch8/deep_convnet.py 참조

