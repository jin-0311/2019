# 출처: 파이썬 자료구조와 알고리즘 : 기초 튼튼, 핵심 쏙쏙, 실력 쑥쑥
# http://www.hanbit.co.kr/media/books/book_view.html?p_code=B8465804191
# < Ch1 숫자

# 정수 integer : immutable불변형
# int(문자열, 밑) 밑은 2~36사이의 선택적 인수
(999).bit_length() # 10 : 바이트수 확인
s='11'
d=int(s)
d
int(s,2) # 2진법
int(s,10) # 10진법

# 부동 소수점 float
0.2 * 3 ==0.6
1.2-0.2==1.0
# false로 나올때 밑처럼 함수로 쓰면 True라고 나옴
def a(x,y, places=7):
    return round(abs(x-y), places) ==0
a(0.2*3, 0.6)

45/6
45//6
45%6
divmod(45,6) # 몫, 나머지

round(100.96, -2)  # 100.0 -> n만큼 반올림한값을 반환(음수일때)
round(100.96, 2) # 100.96 -> 소수점 n자리로 반올림

2.75.as_integer_ratio() # 소수점을 분수로 반환
0.25.as_integer_ratio()

# 복소수 complex number z=4+3j 한쌍을 갖는 불변형
z=4+3j
z.real # 실수부 4.0
z.imag # 허수부 3.0
z.conjugate() # 켤레 복소수 4-3j

# 복소수 사용시 cmath import
# cmath. phase()/polar()/rect()/pi/e 등 복소수 전용 함수 있음
import cmath
cmath.e

# 분수 사용시 fraction 모듈
from fractions import Fraction

def rounding_floats(number1, places):
    return round(number1, places)

def float_to_fraction(number):
    return Fraction(*number.as_integer_ratio())

def get_denominator(number1, number2):
    # 분모 반환
    a=Fraction(number1, number2)
    return a.denominator

def get_numerator(number1, number2):
    # 분자 반환
    a=Fraction(number1, number2)
    return a.numerator

def test_testing_floats():
    number1=1.25
    number2=1
    number3=-1
    number4=5/4
    number6=6
    assert (rounding_floats(number1, number2) ==1.2)
    assert (rounding_floats(number1*10, number3) ==10)
    assert (float_to_fraction(number1) == number4)
    assert (get_denominator(number2, number6) == number6)
    assert (get_numerator(number2, number6) == number2)
    print('통과!')

if __name__ == '__main__':
    test_testing_floats()

# decimal 정확한 10진법의 부동소수점 숫자가 필요할때! 정확한 exp를 쓸때 등
# 불변타입인 decimal.Decimal 사용 (정수, 문자열 가능)
from decimal import Decimal
Decimal.from_float(10.2)

sum(0.1 for i in range(10)) ==1.0  # F
sum(Decimal('0.1') for i in range(10)) ==Decimal('1.0')  #T

# 2,8,16진수
bin(999)
oct(999)
hex(999)

float(999)
int(999)

# 진법 변환 연습문제
# 다른 진법의 숫자를 base 진수로 변환(2 <= base <=10)
def convert_to_decimal(num, base):
    mul, result=1,0
    while num >0 :
        result += num %10 * mul   # 반복문을 돌면서 일의자리 숫자를 하나씩 가져와서 계산
        mul *= base
        num =num//10
    return result

convert_to_decimal(0b1111100111,10)

def test_convert_to_decimal():
    num, base =1001,2
    assert(convert_to_decimal(num,base) ==9)
    print('통과')

if __name__ =='__main__':
    test_convert_to_decimal()

# 10진수를 다른 base진수로 바꾸는 함수
def convert_from_decimal(num, base):
    mul, result=1,0
    while num > 0:
        result += num % base * mul
        mul *= 10
        num = num //base
    return result

def test_convert_from_decimal():
    num, base=9,2
    assert (convert_to_decimal(num, base) == 1001)
    print('통과')

if __name__=='__main__':
    test_convert_to_decimal()

# base가 10보다 큰경우 문자를 사용함 A=10, B=11, C=12 등
# 20이하 진법으로 변환
def convert_from_decimal_larger_bases(num, base):
    s='0123456789ABCDEFGHIJ'
    result=''
    while num > 0 :
        digit=num % base
        result=s[digit] + result
        num = num //base
    return result
def test_convert_from_decimal_larger_bases():
    num, base=31,16
    assert (convert_from_decimal_larger_bases(num, base) =='1F')
    print('pass!')
if __name__=='__main__':
    test_convert_from_decimal_larger_bases()

# reculsive function을 사용한 진법 변환
def convert_dec_to_any_base_rec(num, base):
    converS='012345679ABCDEF'
    if num <base:
        return converS[num]
    else:
        return convert_dec_to_any_base_rec(num//base,base) + converS[num % base]
def test_convert_dec_to_any_base_rec():
    num=9
    base=2
    assert (convert_dec_to_any_base_rec(num, base) =='1001')
    print('pass!')
if __name__=='__main__':
    test_convert_dec_to_any_base_rec()

# 최대공약수 Greater Common Divisor GCD
def finding_gcd(a,b):
    while (b!=0):
        result=b
        a,b =b, a%b
    return result
def test_finding_gcd():
    num1=21
    num2=12
    assert (finding_gcd(num1, num2)==3)
    print('pass!')

if __name__=='__main__':
    test_finding_gcd()  # 4넣으면 assertionErr


# random모듈
import random
def test_random():
    # random module test
    values=[1,2,3,4]
    print(random.choice(values))
    print(random.choice(values))
    print(random.choice(values))
    print(random.sample(values,2))  # 2개
    print(random.sample(values,3))  # 3개

    # shuffle the value list
    random.shuffle(values)
    print(values)

    # make the random integers (0~10)
    print(random.randint(0,10))
    print(random.randint(0,10))

if __name__ =='__main__':
    test_random()


# 피보나치 수열 fibonacci sequence 1st,2nd 항이 1, 나머지는 앞 두항의 합 : 1 1 2 3 4 8 13 21 ...
'''
rec : 재귀호출 O(2^n)
iter : 반복문 O(n)
form : 수식 O(1) 하지만 70번째 이상의 결과는 정확하지 않음 
generator : 제너레이터(파이썬 시퀀스를 생성하는 객체, 아주 큰 시퀀스도 순회가능, 마지막 호출을 기억) 

만약 5n^3 + 3n 의 식이 있다면 계수와 낮은차수의 항 지우고 O(n^3)이라고 할 수 있음 
'''

import math
def find_f_seq_iter(n) :
    if n < 2:
        return n
    a,b=0,1
    for i in range(n):
        a,b=b, a+b
    return a
find_f_seq_iter(70)

def find_f_seq_rec(n):
    if n < 2:
        return n
    return find_f_seq_rec(n-1) + find_f_seq_rec(n-2)
find_f_seq_rec(70)

def find_f_seq_form(n):
    sq5=math.sqrt(5)
    phi=(1+sq5) /2
    return int(math.floor(phi ** n / sq5))
find_f_seq_form(70)  # 오래걸림

def fib_generator():
    a,b=0,1
    while True:
        yield b
        a,b=b, a+b
fib_generator(10) #으로 하면 안돼

if __name__ =='__main__':
    fg=fib_generator()
    for _ in range(10):
        print(next(fg), end='/')


# 소수 Prime number 3가지 방법으로 판단
'''
1. 브루트 포스 brute force : 무차별 대입방법
2. 제곱근 이용 m*m=n (== root(n) = m) 
    n이 소수가 아니면 n = a* b -> m * m = a* b 이고, m은 실수, n,a,b는 자연수 
    a>m -> b<m 
    a=m -> b=m
    a<m -> b>m 
위 세가지경우 --> 모두 min(a,b) <=m  --> m까지의 수를 검색하면 적어도 하나의 n과 나누어 떨어지는 수 발견
--> n은 소수가 아니다 라고 판단 가능
3. 확률론적 테스트와 페르마의 소정리 사용(Fermat's little theorem)
어떤 수가 소수일 간단한 필요조건에 대한 정리 
(p: 소수, a가 p의 배수일때, a^(p-1)=1 (==mod p) --> a^(p-1) % p =1
'''

import math, random
def finding_prime1(number):
    num=abs(number)  # -일수도 있으니까
    if num < 4:
        return True
    for i in range(2, num): # 2부터 나눠서 나머지가 하나라도 0이 아니라면 소수가 아니지
        if num % i == 0:
            return False
    return True
finding_prime1(12)

def finding_prime2_sqrt(number):
    num=abs(number)
    if num < 4 :
        return True
    for i in range(2, int(math.sqrt(num))+1):  # num=12 -> 144+1 까지 하나씩 나눠서 나머지가 0 이 나오면 소수 아님
        if number % i == 0:
            return False
    return True
finding_prime2_sqrt(13)


def finding_prime3_fermat(number):
    if number <=102 :
        for a in range(2, number):
            if pow(a, number-1, number) != 1:
                return False
        return True
    else:
        for i in range(100):
            a = random.randint(2, number-1)
            if pow(a, number-1, number) !=1 :
                return False
        return True

# pow(a,b) --> a^b
# pow(a,b,c) --> a^b % c
pow(2,4,3) # 2^4=16  16%3 = 1  --> 결과는 1
# sqrt(x) x의 제곱근을 구해줌 sqrt==root

# random 모듈을 사용해 n비트 소수 생성
# 3 입력 --> 5(101(2)) 또는 7(111(2)) 의 결과가 나옴
import math, random, sys
def finding_prime_sqrt(number):
    num=abs(number)
    if num < 4 :
        return True
    for i in range(2, int(math.sqrt(num))+1):
        if number % i == 0:
            return False
    return True
def generate_prime(number=3):
    while 1:
        p=random.randint(pow(2, number-2), pow(2, number-1)-1)
        p=2*p+1
        if finding_prime_sqrt(p):
            return p
if __name__ =='__main__':
    if len(sys.argv) < 2:
        print('Usage: generate_prime.py number')
        sys.exit()
    else :
        number=int(sys.argv[1])
        print(generate_prime(number))

# numpy  : 임의의 차원(dimension)을 가짐  / 리스트보다 빠름
# array method를 사용하여 시퀀스의 시퀀스(리스트 또는 튜플)를 2차원 넘파이배열로 생성가능
import numpy as np
x=np.array( ((11,12,13), (21,22,23), (31,32,33)))
x.ndim # ndim attribute(속성)은 배열의 차원수를 알려줌

x=np.array([1,2,3,5,10])
y=np.array([3,4,5])
z=np.array([1,2,3])

x*2
x+10
np.sqrt(x)
np.cos(x)
x-y

np.min(x) # ==x.min()
np.max(x) # == x.max()

np.argmin(x) # index of min    ==x.argmin()
np.argmax(x) # index of max
np.where('조건')  # 조건에 맞는 인덱스 반환

np.where(x>=3)  # 조건에 맞는 인덱스
x[np.where(x>=3)]  # 조건에 맞는 값 반환

# for문이나 if 대신해서 사용하기!
np.where(x>=3, 3, x) # 3보다 크면 3으로 바꾸고, 조건에 안맞으면 그대로
np.where(x<2, x, 10) # 2보다 작으면 그대로, 크면 10으로

m= np.matrix([z,y,z]) # matrix대신에 array로 사용하기
m
m=np.array([z,y,z])
m
m.T

# shape따로 지정할때
grid1=np.zeros(shape=(10,10), dtype=float)
grid1

grid2= np.ones(shape=(10,10), dtype=float)
grid2

# 특정 array모양에 비슷하게 [1,1,1,1]로 직접 넣어도 가능
np.zeros_like(m)
np.ones_like(m)

grid1[1]+10
grid2[:, 2:4]*2
grid2[2:4, :]*2


# speed test of numpy
import numpy as np
import time

def trad_version():
    t1=time.time()
    X=range(10000000)
    Y=range(10000000)
    z=[]
    for i in range(len(X)):
        z.append(X[i]+Y[i])
    return time.time() -t1

trad_version() # 3.8159

def np_version():
    t1=time.time()
    x=np.arange(10000000)
    y=np.arange(10000000)
    z=x+y
    return time.time() - t1
np_version()  # 0.0809

# < ch2 내장 시퀀스 타입
'''
sequence type data가 가진 속성(attribute)
멤버쉽 membership 연산 : in 사용
크기 size 함수 : len(seq)
슬라이싱 slicing 속성 : seq[:-1]
반복성 iterability : 반복문에 있는 데이터를 순회가능 

내장 시퀀스 타입: 문자열/튜플/리스트/바이트배열/바이트 
네임드 튜플은 표준 라이브러리인 collections모듈에서 사용 
'''
l=[]
type(l)
s=''
type(s)
t=()
type(t)
ba=bytearray(b'')
type(ba)
b=bytes([])
type(b)

# 가변성 mutable  : 리스트 / 바이트
# 불변성 immutable : 튜플  / 문자열 / 바이트배열
# 파이썬의 모든 변수는 객체참조 reference이므로 가변 객체를 복사할때 매우 주의 ! 깊은 복사 deep copy개념 필수
a=b # a는 실제b가 가리키는(참조하는)곳을 가리킴

#deep copy of list
list=[1,2,3,4]
new1=list[:]
new1

# deep copy of set
people={'buffy','angel','amy'}
s=people.copy()
s
s.discard('amy')
s.remove('angel')
s
people

# deep copy of dict
mydict={'hello':'world'}
newdict=mydict.copy()

# 다른 객체도 복사할때 copy모듈 사용
import copy
obj='some of my object'
newobj=copy.copy(obj) # 얕은 복사 shallow copy
newobj2=copy.deepcopy(obj) # 깊은 복사 deep copy

# 슬라이싱 연산자 : 파이썬 시퀀스 타입의 슬라이싱
# seq[start]
# seq[start:end]
# seq[start:end:step]

word='hello world! nice to meet you'
word[-1]
word[-2]
word[-2:]
word[:-2]
word[-0]

# 문자열 = sequence of characters -> string 불변
'''
파이썬 모든 객체는 두가지의 출력 형식이 있음 
string(사람을 위해)  :str(), object.__str__(self) 등
representational(파이썬 인터프리터에서 사용, 디버깅할때 사용)  : repr(), object.__repr__(self) 
'''
# 유니코드 Unicode : 전세계언어의 문자를 정의하기 위한 국제표준 코드 (공백, 특문, 숫자 등 기호들도 있)
# 문자열 앞에 u를 붙이면 유니코드 문자열 만들수 있음
u'잘가\u0020세상 !'

# 문자열 메서드
# a.join(b) : list b에 있는 모든 문자열을 하나의 단일 문자열 a로 결합    / +도 가능하지만 비효율적
s=['angel','amy','sera']
' '.join(s)
'-<>-'.join(s)
''.join(s)      # Out[159]: 'angelamysera'
''.join(reversed(s))  # Out[160]: 'seraamyangel'

# a.ljust(width, fillchar), a.rjust(width, fillchar)  채우기
# ljust 문자열 a의 맨처음부터, (a포함한 길이width)만큼 뒤에 채움 -> 문자열 다음으로 채움  (문자열이 왼쪽)
# rjust -> 문자열 앞에 채움 (문자열이 오른쪽)

name='sera'
name.ljust(50,'-')
name.rjust(50,'*')

# a.format() : a에 변수를 추가하거나 형식화하는데 사용  # 띄어쓰기 주의
'{0} {1}'.format('hi','python')
'name : {who}, age : {age}'.format(who='amy',age='17')
'name : {who}, age : {0}'.format(12, who='amy')
# 인덱스 생략가능 자동으로!
'{} {} {}'.format('python','algo','!')

# +연산자를 사용하면 더 간결하게 결합가능
# 지정자 s는 str(문자열 형식) / r : 표현(repr 형식)  / a : 아스키코드 형식을 의미
import decimal
'{0} {0!s} {0!r} {0!a}'.format(decimal.Decimal('99.999'))

# 문자열 매핑 언패킹 mapping unpacking (언패킹: 컬렉션의 요소를 여러변수에 나누어 담는 것)  **
# 함수로 전달하기에 적합한 키-값 딕셔너리 생성됨

# locals() method 는 현재 스코프scope에 있는 지역변수 local variable를 딕셔너리로 반환
hero='ironman'
number=999
a='{number}:{hero}'.format(**locals())
a
type(a)

# a.splitlines() -> \n을 기준으로 분리한 결과를 문자열 리스트로 반환
s='sera\nphina'
s.splitlines()

# a.split(t,n) : 문자열 a에서 문자열 t를 기준으로 정수 n번만큼 분리한 문자열 리스트를 반환
# n지정 안하면 t로 최대한 분리  / t도 지정안하면 공백문자whitespace로 구분
s='buffy*chris-merry*199'
s0=s.split('*',1)
s0
s1=s.split('*')
s1
s2=s1[1].split('-')
s2

# split 사용, 문자열에서 스페이스 제거하는 함수
def erase_space_from_string(s):
    s1=s.split(' ')
    s2=''.join(s1)
    return s2
erase_space_from_string('hello world I am jiny')

start='안녕*세상*!'
start.split('*',1)  # ['안녕', '세상*!']
start.rsplit('*',1)  # ['안녕*세상', '!']

# a.strip(b)
