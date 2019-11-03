# 출처 : 점프 투 파이썬
# https://wikidocs.net/book/1

#%% 점프투파이썬
#%% -<Ch01 파이썬 둘러보기> 
#%% 기초 
# 실행 : 블록설정(shift + 방향키) 하고 ctrl + enter / F9 
# 간단한 계산 IPython console에서 가능(오른쪽 아래) 
# """ 긴글 """ 로 주석처럼 긴 줄 쓸수 있음 

1+1
a = 1
b = 33
c = a + b
c
a = "Python" 
a          #'Python' 으로 나옴
print(a)   # Python

a = 2 + 3j   # 복소수 == j
b = 3
a * b

a = 3
if a > 1:
    print("a is aa")

# 조건문 if
a = 3
if a > 1:
    print("a is greater than 1")

# 반복문 for
for a in [1, 2, 3]:
    print(a)

# 반복문 while 
i = 0
while i <3 :
    i = i + 1
    print(i)

# 함수
def add(a, b):
    return a+b
add(3,4)

#%%
#%% <Ch 02 파이썬 프로그래밍의 기초, 자료형>
#%% 02-1) 숫자형 Number
# 정수integer, 실수Floating-point(소수점), 8진수Octal, 16진수Hexadecimal
a = 4.24e10
b = 4.24e-10
c = 0o177
d = 0xABC 

# 사칙연산 
3 + 4
3 * 4
3 ** 4   # n 제곱
4 / 3    # 나누기
4 // 3   # 몫 반환
4 % 3    # 나머지 반환


# 간단한 계산기 
a = 100
b = 50
result = a + b
print(a, "+", b, "=", result)

# input 사용하는 계산기 
a = int(input("첫번째 숫자를 입력해주세요:"))  # input은 string으로 받음  -> int() 사용
b = int(input("두번째 숫자를 입력해주세요:"))
result = a + b
print(a, "+", b, "=", result)

# input문 만들기 
a= input("당신의 나이는 몇 살이세요?:")
print("제 나이는",a,"살 입니다.")


#%% 02-2) 문자열 자료형 String ->  " "
"Hello"
'python'
"""Life is too short, You need python"""
'''life'''

# 문자열에 작은 따옴표 포함시키기 
food = "Python's favorite food is perl"  # 작은 따옴표로 묶으면 오류 
food
# 문자열에 큰 따옴표 포함
say = '"Python is very easy." he says.'
say
# \ 사용해서 포함시키기 : 'or " 앞에 \붙이기! 
food1 = 'python\'s favorite food is perl.'
food1
say1 = "\"python is very easy.\" he says"
say1

# 여러줄 문자열 변수에 대입 \n : escape code or """ or '''사용 
# (console에는 \n으로 나오지만 variable explorer에는 잘 나옴)
ma = "life is too short \nyou need python" 
ma
mb = """life is too short
you need python
"""
mb
mc = '''
life is too short
you need python
'''
mc
print() # 하면 세개 동일하게 나옴!

# escape code : 미리 정의된 문자조합 
# \n 줄바꾸기,     \t 탭 간격 주기,      \\ 문자 \를 그대로 표현
#  \' 작은따옴표 그대로 표현,       \" 큰따옴표 그대로 표현  

# 문자열 연산하기
# 문자열 더해서 연결 concatenation
head = "Python"
tail = " is fun."
head + tail

# 문자열 곱하기(반복하기)
a = "python"
a*2             #>>>'pythonpython'

# 응용
print("=" * 50)
print("my program")
print("=" * 50)

# 문자열 길이 구하기
a = "Life is too short"
len(a)

# 무조건 0부터 시작!! 
# 문자열 인덱싱indexing : 가리킨다 / 공백도 한칸!  
a = "Life is too short, You need Python"
a[0]
a[3]
a[-1]   # 뒤에서 1번째 
a[-0]   # -0 == 0 
a[12]


# 문자열 슬라이싱 slicing:  a[시작번호 : 끝번호]    시작번호 <= a < 끝번호 
a = "Life is too short, You need Python"
a[0:4] #L부터 e까지 -> Life 
a[5:7]
a[12:17]
a[19: ]   # 19부터 끝까지
a[ :17]   # 16까지 나타냄 
a[:]      # 전체
a[19:-7]  # <- 끝번호가 -7 이므로 19부터 -8까지 나타냄 (You need)  

# 슬라이싱으로 문자열 나누기 

# 8을 기준으로 나눌 수 있어서 편함! 
# 0부터 시작하지만 끝번호를 포함하지 않으므로 편함!
a = "20010331Rainy"
data = a[:8]
weather = a[8:]

year = a[:4]
year
month = a[4:6]
month
day = a[6:8]
weather = a[8:]

# pithon을 python으로 바꾸기
a = "pithon"
a[:1]
a[2:]
a[:1] +"y" +a[2:]

# 문자열 포매팅 formatting - 문자열 안의 특정한 값을 바꿔야 할 때 

# 문자열 포맷 코드
# %s 문자열(string),   %c 문자1개(character),   %d 정수(integer),   %f 실수(floating-point) 
# %o 8진수,         %x 16진수,        %% '%'그자체 


print("100")  # str
print("%d" % 100) # 포맷을 int로 지정해주고 100 프린트하라함! 
print("100+100")

print("%d"%(100+100))
print("%d"%(100,200))  # error 숫자가 2개여서 (포맷은 1개)
print("%d %d"% (100))  # error 숫자가 1개여서 (포맷은 2개)
print("%d %d" %(100,200))
print("%d/%d=%d"%(100,200,0.5))  # 0나옴 (int는 소수점 ㄴ)
print('%f/%f=%f'%(100,200,0.7))   #>>>0.700000

print("%d" % 123)      #>>>123
print("%5d" % 123)     #>>>  123 (5자리 할당해서, 앞에 빈칸2칸 생김)
print("%05d" % 123)    #>>>00123 

print("%f" % 123.45)      #>>>123.450000 (f로 지정하면 소수점 6자리 까지)
print("%7.1f" % 123.45)   #>>>123.5
print("%7.3f" %123.45)    #>>>123.450

print("%s" % "python")  
print("%10s" % "python")   #앞에 4자리 빈칸할당하고 출력


age = int(input("당신의 나이는?:"))
print("제 나이는 %d 입니다." %age)

name = input("당신의 이름은?:")
print("제 이름은 %s 입니다." %name)

age = int(input("당신의 나이는?:"))
name = input("당신의 이름은?:")
print("제 나이는 %d이고, 이름은 %s입니다."%(age,name))

# 숫자 바로 대입 %d 
"I eat %d apples." %3
# 문자열 바로 대입 %s (뒤에 나오는 % "str" 큰따옴표 필요 )
"I eat %s apples." % "five"
# 숫자값을 나타내는 변수로 대입
number = 3
"I eat %d apples." %number
# 두개 이상의 값 대입
number = 10
day = "three" 
"I ate %d apples. so I was sick for %s days." %(number,day)

# %s (string) -> 어떤 값을 넣어도 다 나와!  대신 다 string으로 변환됨 
a = "I have %s apples." %3
b = "rate is %s" %3.234
print(a)
# %d와 %를 같이 쓸땐 %%
"Error is %d%." %98   # error
"Error is %d%%." %98  # >>> 'Error is 98%.'


# 포맷코드와 숫자 함께 사용하기

# 오른쪽 정렬
"%10s" % "hi"       #>>>'        hi'  
# 왼쪽 정렬
"%-10sJain" % "hi"  #>>>'hi        Jain'

# 소수점 표현하기( . 도 한칸임!)
"%0.4f" % 3.42134234
"%10.4f" % 3.42134234

# format함수를 사용한 포매팅 
# {}: 인덱스 0부터 시작 /  .format() 인덱스에 들어갈 값 넣기 
"I ate {0} apples.".format(3)
"I ate {0} apples.".format("five")

number=3
"I ate {0} apples.".format(number)

number=10
day="three"
"I ate {0} apples. so I was sick for {1} days.".format(number,day)        # 1) 변수를 미리 지정하거나,

"I ate {number} apples. so I was sick for {day} days.".format(number=10, day=3)    # 2) format(name=value) 넣기

"I ate {0} apples. so I was sick for {day} days.".format(10, day=3)       # 3) 인덱스{0} 과 name=value 혼용 가능 


# 왼쪽 정렬 {}안에 :<10 표현식 사용하면 문자열 왼쪽으로 정렬, 10자리수 
"{0:<10}".format("hi")  #>>>'hi        ' 

# 오른쪽 정렬 {0:>10} 표현식 사용
"{0:>10}".format("hi")  #>>>'        hi'

# 가운데 정렬 {0:^10}
"{0:^10}".format("hi")  #>>>'    hi    '

# 공백 채우기 {} 안에 넣을 표현식(>오른쪽정렬,<왼쪽정렬,^가운데정렬) 앞에 지정한 문자값(=,!..) 넣기 
"{0:=^10}".format("hi")
"{0:!<10}".format("hi")

# 소수점 표현하기
y = 3.141592
"{0:0.4f}".format(y)
"{0:10.4f}".format(y) #>>>'    3.1416'

# {}을 그대로 표현하기
"{{and}}".format() #>>>'{and}'

# f문자열 포매팅(파이썬3.6부터 사용가능)  f' 작은 따옴표로 쓰기! 
name = "홍길동"
age = 30
f'나의 이름은 {name}입니다. 나이는 {age}입니다. '
# 응용   #표현식에 사칙연산 가능
age = 30
f'나는 내년이면 {age+1}살이 된다.' 

# 딕셔너리 : key와 value를 한쌍으로 갖는 자료형  + f 포매팅 
d = {"name":"홍길동", "age":30}
f'나의 이름은 {d["name"]}입니다. 나이는 {d["age"]}입니다.'
# 정렬
f'{"hi":<10}'  # 왼쪽정렬
f'{"hi":>10}'  # 오른쪽 정렬
f'{"hi":^10}'  # 가운데 정렬 
# 공백채우기
f'{"hi":=^10}'   # ^뒤에 띄어쓰기 하면 안돼
f'{"hi":!<10}'
# 소수점
y=3.141592
f'{y:0.4f}'
f'{y:10.4f}'
# {and} 그대로 출력 
f'{{and}}'


# 문자열 관련 내장함수 -> 변수이름.함수()

# 문자열 개수 세기  .count()
a = "hobby"
a.count("b")

# 문자 위치 알려주기 1    .find(요소)
a = "Python is best choice"
a.find("b")   # 10번째 자리(0부터 세기)
a.find("k")   # 문자열이 존재하지 않으면 -1 반환 

# 문자 위치 알려주기 2    .index(요소)
a = "Life is too short"
a.index("t")
a.index("K") # value error


# 문자열 삽입(join)  - 리스트, 튜플에서도 사용가능 
a = ","
a.join("abcd")

# 소문자 -> 대문자 (upper) /  대문자 -> 소문자 (lower)
a = "hi"
a.upper()
a= "HI"
a.lower()

# 왼쪽 공백 지우기 .lstrip() 연속된 공백 삭제 
a = ' hi'
a.lstrip()

# 오른쪽 공백 지우기 .rstrip()
a = ' hi '
a.rstrip()

# 문자열 바꾸기 .replace(old, new) 
a = 'life is too short'
a.replace('life', 'your leg')

# 문자열 나누기 .split()  -> 공백을 기준으로 /  .split(':') ->  :기호를 기준으로 나눔  작은따옴표 꼭 해주기!  
a = 'life is too short'
a.split()
b = 'a:b:c:d'
b.split(':')


#%% 02-3) 리스트 자료형 list  -> 리스트명 = [요소1,요소2,요소3,...] 
# 리스트 만들기 
odd = [1,3,5,7,9]
a = list() # 비어있는 리스트 생성
a = []
b = [1,2,3]
c = ['life', 'is', 'too', 'short']
d = [1,2,'life', 'short']
e = [1,2, ['life','is']]   #리스트 안에 리스트 가능 

print(b)
print(d)
print(e)

# 리스트의 인덱싱
a= [1,2,3]
a

a[0]          # 첫번째 요소
a[0] + a[2]   # 첫번째 요소 + 세번째 요소
a[-1]         # 마지막 요소

a = [1,2,3,['a','b','c']]
a[0]
a[-1]
a[3]
a[-1][0] #마지막 요소(리스트형태)의 첫번째 요소 
a[3][1]
a[-1][2]

a = [1,2,['a','b',['life','is']]]
a
a[2][2][0]

colors = ['red','blue','green']
colors[0]
colors[2]


# 리스트의 슬라이싱 (문자열과 동일!)
a = [1,2,3,4,5]
a[0:2]
b = a[:2]
b
c = a[2:]
c

a = [1, 2, 3, ['a','b','c'], 4, 5]
a[2:5]
a[3][:2]

# 리스트 연산하기 
# 리스트 연결concatenation : + 
a = [1,2,3]
b = [4,5,6]
a + b     #>>> [1, 2, 3, 4, 5, 6]

# 리스트 반복 :  * 
a * 3     #>>> [1, 2, 3, 1, 2, 3, 1, 2, 3]

# 리스트 길이 구하기 (요소 개수를 구함 )
len(a)

color1 = ['red','blue','green']
color2 = ['orange','black','white']

color1 + color2
color1 *2

# 리스트 연산시 주의할점
a = [1,2,3]
a[2] + "hi"    # >>> int + str 불가!
str(a[2]) +"hi"

# 리스트 수정
a = [1,2,3]
a[2] = 4
a

# 리스트 삭제 del함수 사용 : del 객체 : del 변수[요소번호]     - 밑에 써있 remove, pop 함수로도 삭제 가능
a = [1,2,3]
del a[1]
a

a = [1,2,3,4,5]
del a[2:]
a

# 리스트 관련 함수 : 리스트 변수 뒤에 . 붙여서 함수사용
# 리스트에 요소 추가 .append(x) 맨 마지막에 x추가
a = [1,2,3]
a.append(4)
a

a.append([5,6])  # 모양 그대로 리스트로 들어감 
a

a.extend([5,6])  # extend는 요소로 추가, 리스트로 안들어감 
a

# 리스트 정렬 .sort()
a = [1,4,3,2]
a.sort()
a

a = ['a', 'b', 'c']
a.sort()    # reverse = False 가 디폴트 >>> a,b,c
a
a.sort(reverse = True )   # >>> c,b,a
a

# 리스트 뒤집기 .reverse()  현재의 리스트를 그대로 거꾸로 뒤집음(sort아니야!)
a = ['a', 'c', 'b']
a.reverse()
a

# 위치 반환(index) - 해당하는 요소가 어디 위치에 있는지 찾는 것! a.index(요소) 
a = [1,2,3]
a.index(3)
a.index(1)
a.index(0)  #>>> valueError

# 리스트에 요소 삽입 insert(a,b) a번째 위치에 b를 삽입  (기존 요소 삭제하는 건 아님)
a = [1,2,3]
a.insert(0,4)   #>>>[4, 1, 2, 3]
a
a.insert(3,5)   #>>>[4, 1, 2, 5, 3]
a

# 리스트 요소 제거 remove(x)   첫번째로 나오는 x 삭제
a = [1,2,3,1,2,3]
a.remove(3) # 앞의 3 제거
a
a.remove(3) # 뒤의 3도 제거
a

# 리스트의 맨 마지막 요소 끄집어내고 삭제하기 pop()
a = [1,2,3]
a.pop()
a

# pop(x) : x 번째 요소 돌려주고 삭제
a = [1,2,3]
a.pop(1)
a

# 리스트에 포함된 요소 x의 개수 세기: count(x)
a = [1,2,3,1]
a.count(1)

# 리스트 확장 : extend(x) : x 에는 리스트만 올수 있다. 기존 a 리스트에 x 연결
a = [1,2,3]
a.extend([4,5])
a
b = [6,7,8]
a.extend(b)
a

# 변수초기화
a,b,c,d, = 0,0,0,0

# 인풋 변수와 리스트 예제 
sum_abcd = 0
a = int(input("1번 숫자:"))
b = int(input("2번 숫자:"))
c = int(input("3번 숫자:"))
d = int(input("4번 숫자:"))
sum_abcd = a+ b+ c+ d
print("합계 ===> %d" % sum_abcd)
# 리스트 사용
aa = [0,0,0,0]
sum_aa = 0
aa[0] = int(input("1번 숫자:"))
aa[1] = int(input("2번 숫자:"))
aa[2] = int(input("3번 숫자:"))
aa[3] = int(input("4번 숫자:"))
sum_aa = aa[0] + aa[1] + aa[2] + aa[3]
print("합계===> %d" % sum_aa)



#%% 02-4) 튜플 자료형 tuple  -> 튜플명 = (요소1, 요소2,...)
# 튜플 안의 값을 생성,삭제, 수정 불가 (리스트는 됨)

t1 = ()
t1
t2 = (1,)  # 한개의 요소만 가질때는 , 꼭 찍어야함
t2
t2_1 = (1)    # tuple 아니고 int 
t2_1
t3 = (1,2,3)
t4 = 1,2,3 # 괄호 안써도됨 
t4
t5 = ('a','b',('ab','cd'))
t5

# 리스트와 다르게 삭제, 변경 불가
t1 = 1,2,'a','b'
t1
del t1[0]  #>>> type error
t1[0] = 'c'  #>>> type error

# 튜플 다루기
# 인덱싱
t1 = 1,2,'a','b'
t1[0]
t1[3]

# 슬라이싱
t1[1:]

# 튜플 더하기(연결)
t2 = 3,4
t1 +t2

# 튜플 곱하기(반복)
t2 * 3

# 튜플 길이 구하기(요소 개수)
len(t1)



#%% 02-5) 딕셔너리 자료형 Dictionary -> 딕셔너리명 = {key1:value1, key2:value2, ...} 
# key : value로 구성 되어있음
# 대응관계를 갖는 자료형 -> associative array == hash
# 리스트나 튜플처럼 순차적으로(sequential) 요솟값을 구하지 않고 key를 통해 value얻음 

dic = {'name' : 'pey', 'phone' : '01022224444', 'birth': '1118'}
a = {1:'hi'}
b = {'a': [1,2,3]}

# 딕셔너리 쌍 추가, 삭제
# 쌍 추가
a = {1 : 'a'} 
a[2] = 'b'
a
a['name'] = 'pey'
a
a[3] = [1,2,3]
a

# 쌍 삭제
del a[1]
a

# 딕셔너리를 사용하는 방법 
# 딕셔너리에서 key(앞)를 사용해 value얻기
grade = {'pey': 10, 'ju': 99}
grade['pey']
grade['ju']

a = {1:'a', 2:'b'}
a[1]
a[2]

a = {'a':1, 'b':2}
a['a']

dic = {'name' : 'pey', 'phone' : '01022224444', 'birth': '1118'}
dic['name']

# 딕셔너리 주의사항 -> key는 고유한 값 하나를 설정하면 마지막으로 지정한 설정만 남고 다 무시됨
# a = {1:'a', 1:'b', 1:'c'}
# a
# key에 리스트 사용불가
a = {[1,2]:'hi'} #>>>type error

# 딕셔너리 관련 함수
# key list 만들기 -> .keys()
# 반환하는 값을 리스트로 받으려면 list(a.keys()) 를 사용
# dict_keys, dict_values, dict_items등은 리스트로 변환하지 않아도 기본적인 반복(iterate) 구문(e.g.for문) 실행가능
# 하지만 리스트 고유의 append, insert, pop, remove, sort는 불가능
a = {'name' : 'pey', 'phone' : '01022224444', 'birth': '1118'}
a.keys()   #>>> dict_keys(['name', 'phone', 'birth'])

for k in a.keys() :  # k는 i처럼
    print(k)
# k는 i처럼 사용된 것 
    
list(a.keys())   # keys를 리스트로 변환
a.keys()         # key를 얻을 때(리스트로)
a.values()       # value를 얻을 때(리스트로)
a.items()        # key, value 쌍을 얻을때 튜플로 묶어서 

a.clear()       # key:value 쌍 모두 지우기
a  # 빈 딕셔너리도 {}로 표현

a = {'name':'pey', 'phone' : '01022224444', 'birth': '1118'}
a.get('name')    # == a['name']
a.get('nokey')   # 빈칸으로 나옴
print(a.get('nokey'))  #>>> none

# .get(x, '디폴트값')  딕셔너리 안에 key값이 없으면 미리 지정한 디폴트값을 대신 가져오게 할때 
a.get('foo','bar')  #>>>'bar' 

# 해당키가 딕셔너리 안에 있는지 조사 ->  'key' in 딕셔너리명 
'name' in a   #>>> true
'email' in a   #>>> false

#%% 02-6) 집합 자료형 set  -> set() 
# 중복 허용하지 않고 순서가 없다(unordered)  -> 중복제거 필터링으로 많이 사용    / variable explorer 에 안뜸 
# order자료형 : (인덱싱 가능) = 리스트[], 튜플(괄호생략가능)     / unordered자료형 : set(), dictionary{k:v} 

# set(리스트 or "문자열")
s= set()
s
s1 = set([1,2,3])
s1    #>>> {1, 2, 3}
s2 = set("hello")
s2    #>>> {'e', 'h', 'l', 'o'}

# 인덱싱으로 접근하려면 리스트나 튜플로 변환 필요
s1 = set([1,2,3])
l1 = list(s1)
l1[0]

t1= tuple(s1)
t1[0]


# 교집합, 합집합, 차집합 구하기
s1 = set([1,2,3,4,5,6])
s2 = set([4,5,6,7,8,9])

# 교집합  &
s1 & s2
s1.intersection(s2)

# 합집합   |   ->중복은 알아서 제거
s1|s2 
s1.union(s2)

# 차집합  -
s1-s2
s1.difference(s2)
s2-s1
s2.difference(s1)

# 집합 자료형 관련 함수
# 값1개 추가  .add(x)
s1 = set([1,2,3])
s1.add(4)
s1

# 값 여러개 추가 .update([x1,x2,..])
s1 = set([1,2,3])
s1.update([4,5,6,'seven'])
s1

# 특정값 제거 .remove(x)
s1.remove('seven')
s1


#%% 02-7) 불 자료형 bool ->  True(=0이 아닌숫자) / False(=0)
# True or False만 가질 수 있음 type(변수명) 으로 bool 확인가능
a = True
b = False

type(a)
type(b)

1 == 1  #>>> True
2 > 1   #>>> True
2 < 1   #>>> False

# 자료형의 참과 거짓
# 문자열   "python": T   / "" : F
# 리스트   [1,2,3] : T   / [] : F
# 튜플     () : F
# 딕셔너리 {} : F
# 숫자형   0이 아닌 숫자 : T  / 0 : F
# None : F

# 쓰임
a = [1,2,3,4]
while a:             # a가 참이면 while 계속 실행 
    print(a.pop())   # a의 마지막 요소를 하나씩 꺼냄  >>> 4 3 2 1   a가 빈리스트가 되면 거짓 -> while문 중지 
    
if [] :
    print("true")
else:
    print("false")
   
if [1,2,3] :
    print("true")
else: 
    print("false")
     
# 불연산
bool('python')   # T
bool('')   # F
bool([1,2,3])
bool([])
bool(0)
bool(3)

#%% 02-8) 자료형의 값을 저장하는 변수
# 변수 : 객체를 가리키는 것(객체 : 자료형)
# 변수 이름 = 변수에 저장할 값 

a = 1
b = 'python'

a = [1,2,3] 
# ->  [1,2,3]값을 가지는 리스트 자료형(객체)이 자동으로 메모리에 생성
# ->  변수 a는 자료가 저장된 메모리의 주소를 가리킴

# 메모리의 주소를 확인할때는 id() : 객체의 주소값을 돌려줌 
id(a)  #>>> 200036424

# 리스트 복사
a = [1,2,3]
b = a
id(a)
id(b)   # 둘다 주소값이 동일
a is b  # T

a[1] =4   # 두 번째 요소인 2를 4로 바꿈
a
b         # a,b 둘다 바뀜

#a와 b가 다른 주소를 가리키도록
# sol1) slicing [:]
a = [1,2,3]
b = a[:]
a[1] = 4
a
b
# sol2) use copy() <- module
# copy함수 뒤에서 배움
from copy import copy
a = [1,2,3]
b = copy(a)
id(a)
id(b)    # 다름 
b is a   # F 



# 변수를 만드는 여러가지 방법 - 튜플, 리스트 
a, b = ('python','life')
(a,b) = 'python','life'
[a,b] = ['python','life']
a=b='python'
id(a)
id(b)  # 같은 값 

# 두변수의 값을 바꾸는 간단한 방법 
a=3
b=5
a,b = b,a
a
b


#%%
#%% <Ch 03 프로그램 구조, 제어문 >
#%% 03-1) if문 : 조건에 맞는 상황 수행  
# 조건문을 테스트 참이면 if 블록 수행, 조건문이 거짓이면 elif블록 or else블록 수행 
money = True
if money :
    print("taxi")
else: 
    print("walk")

# 비교 연산자 x<y, x>y, x==y, x != y, x>=y, x<=y
x = 3
y = 2
x > y  #>>> true

money = 2000
if money > 3000:
    print('taxi')
else: 
    print('walk')

# and, or, not
# x and y : 둘 다 참 -> T
# x or y : 둘 중 하나만 참 -> T
# not x : x가 거짓이면 -> T 
    
m = 2000
card = True
if money >= 3000 or card:
    print("taxi")
else :
    print("walk")


# x in s,  x not in s
# x in list/tuple/string
# x not in list/tuple/string
    
1 in [1,2,3]  # >>> T
1 not in[1,2,3] # >>> F

'a' in ('a','b','c')  #>>> t
'j' not in 'python'  #>>> t

pocket = ['paper','cellphone','money']
if 'money' in  pocket :
    print("taxi")
else:
    print("walk")
    

# 조건문에서 아무 일도 하지 않게 설정
pocket = ['paper','cellphone','money']
if 'money' in  pocket :
    pass
else:
    print("card please")

    
# 다중조건 판단 elif
pocket = ['paper','cellphone','money','card']
if 'money' in  pocket :
    print('taxi')
elif 'card' in pocket :
    print('bus')    
else:
    print("walk")

    
pocket = ['paper','cellphone','money','card']
a = input("what's in your pocket?:")
if a in pocket:
    print('happy')
else :
    print('sad')
    

#조건부 표현식 conditional expression  문법) 조건문이 참인경우 if 조건문 else 조건문이 거짓인 경우
s = 50
if s >= 60 :
    message = 'success'
else:
    message = 'failure'    
print(message)    
#위를 조건부 표현식으로 작성하면  -> 가독성 
score = 61
message = 'success' if score >= 60 else 'failure'
print(message)

    
# if문 실습     
a = 200
if a < 100 :
    print("100보다 작아")
else : 
    print("100보다 커")
    
   
s = int(input("점수입력:"))
if s >= 60:
    print("합격")
else: print("불합격")    


t = int(input("온도:"))
if t > 30 :
    print("반바지")
else :print("긴바지")
print("이제 나가서 운동하세요!")
    
   
s = int(input("write your score:"))
if s >= 90 :
    print("a")
elif s >= 80 :
    print("b")
elif s >= 70 :
    print("c")
elif s >= 60 :
    print("c")
else: 
    print("f")
print("is your grade")



#%% 03-2 ) while문(반복문) : 반복해서 문장을 수행할 경우 
# 조건문이 참이면 while문 아래의 문장이 반복해서 수행됨 (조건문이 거짓이 되면 멈추거나 break 필요)

# while문의 기본구조
tree=0
while tree < 10 :
    tree = tree + 1      # tree += 1로 쓰기도 함 
    print("나무를 찍은 횟수:%d" % tree)
    if tree == 10:
        print("나무 아파요")

i=0
while i <= 10:
    print(i, end =" ")
    i = i+1

i = 0
while i <= 20 :
    print(i, end = ' ')    # 매개변수 end=' ' 있어서 한줄에 다나옴
    i += 1

# while 문 만들기
# 여러가지 선택지중 하나는 선택해서 입력받는 예제 
prompt = """
1.add
2.del
3.list
4.quit

enter number: """

n = 0   # 번호를 입력받을 변수 
while n != 4 :  # 입력받은 변수가 4가 아니면 계속 반복 
    print(prompt)
    n = int(input())
    
# while문 강제로 빠져나가기 break   -> while문 안에 if 조건문 그리고 break! 여기서는 else없어도 됨 
coffee = 10
money = 300
while money :
    print("돈을 받았으니 커피를 줍니다.")
    coffee -=1
    print("남은 커피의 양은 %d개 입니다." %coffee)
    if not coffee :
        print("커피가 다 떨어졌습니다. 판매를 중지합니다. ")
        break
        
# 정리된 위의 예제
coffee = 10
while True :
    money = int(input("돈을 넣어주세요 :"))
    if money == 300 :
        print("커피를 줍니다.")
        coffee -= 1
        print('남은 커피의 양: %d개' %coffee)
    elif money > 300 :
        print("거스름돈 %d원을 주고 커피를 줍니다" %(money - 300))
        coffee -= 1
        print('남은 커피의 양: %d개' %coffee)
    else: 
        print("돈을 다시 돌려주고 커피를 주지 않습니다")
        print("남은 커피의 양: %d개" %coffee)
    if coffee == 0:
        print("솔드아웃! 판매중지!" )
        break

# while문의 맨 처음으로 돌아가기 :while 블록 안의 if 조건문 : continue 
# while문을 빠져나가지 않고 맨처음 (조건문)으로 돌아가고 싶을때 (그럼 밑으로 안내려가고 위로 다시)

# 1~10중 홀수 추출
a = 0
while a < 10 :
    a += 1
    if a % 2 == 0: continue  # 나머지가 0이 아닌것만 빠져나옴! 
    print(a)
    
# 3의 배수 추출
b = 0
while b < 100 :
    b +=1
    if b % 3 != 0 : continue 
    print(b)


# 무한 루프 Loop 무한히 반복
# while True :
#   수행할 문장1
#   수행할 문장 2 ...    

while True:
    print("ctrl c로 무한루프 빠져나가기!")     # 콘솔창에서 해야해

    
    
#%% 03-3) for 문(반복문) : 유용, 문장구조 파악 간편
# for문 문법) 
# for 변수 in 리스트/튜플/문자열 :
#   수행할 문장 1
#   수행할 문장 2        
print('hi'*3)

for i in range(5) :
    print('hi')
    
for i in range(0,3,1):  #(초기값, 최종값+1, 증감치) 
    print("hi")
    
for i in range(1,5,2):
    print('apple')   # 1,2,3,4 까지 반복하고 2씩 증가시켜 출력 (default:1)
    
sum_ = 0
for i in range(501,1000,2):
    sum_= sum_ + i
    print('홀수의 합:%d'%sum_)
    
    
# 전형적인 for문 
test_list = ['one', 'two','three']
for i in test_list:    # <- i에 one, two, three를 순서대로 대입
    print(i)
    
# 다양한 for문의 사용 
a= [(1,2), (3,4), (5,6)]
for (first, last) in a:
    print(first + last)
#>>> 3 / 7 / 11 a리스트의 요솟값이 튜플! 자동으로 first, last에 대입됨
    

# for문의 응용
marks= [90,25,67,45,80]
number = 0
for mark in marks : 
    number = number + 1
    if mark >= 60 :
        print('%d번 학생은 합격입니다.'%number)
    else :
        print('%d번 학생은 불합격입니다.'%number)

# for문과 continue문  
#  -> for문 블록의 문장을 수행하는 도중 continue를 만나면 다시 for문의 처음으로 돌아감        
# for문 블록 안에 if 조건문 : continue 
marks= [90,25,67,45,80]
number = 0
for mark in marks:
     number = number + 1
     if mark < 60 : continue                  # 60점 미만은 밑으로 안내려가고 다시 위로 올라감 
     print('%d번 학생 축하! 합격!' %number)   # 60점이상은 continue 안하고 밑으로 내려가서 print 
     
# 위의 반대      
s = [90,25,67,45,80]
number = 0
for i in s :
    number = number + 1
    if i > 60 : continue
    print('%d번 학생 불합격!' %number)
    
 
# for문과 함께 자주 사용하는 range 함수
# range(a) : 0부터 a미만의 숫자 리스트를 자동으로 만들어줌 
# range(a,b,c)  : a시작숫자, b끝숫자(출력은 b미만까지), c증감치

a= range(10)   # 0~9
a= range(1,11) # 1~10

# range함수의 예시 : 1부터 10까지의 합 
add = 0
for i in range(1,11) :
    add = add + i
print(add)
    
# 위의 60점 이상 합격 range문
marks= [90,25,67,45,80]
for number in range(len(marks)) :      #range(len(marks)) == range(5)
    if marks[number] < 60: continue    #a[b] : a의 b번째 값 
    print('%d번 학생 합격!'%(number +1))

# for, range를 사용한 구구단
# 매개변수 end -> 해당 결과값을 출력할때 다음줄로 넘기지 않고 그 줄에 계속출력하기 위해 (2단이 옆으로 쭉)
# print(' ')  -> 2번 for문이 끝나면 결과값을 다음줄 부터 출력하기 위해 (라인바꾸기)
# end=' '와 print(' ') 안쓰면 한줄에 하나씩 나오고 print(' ')를 안쓰면 옆으로 쭉 나옴 
for i in range(2,10) :      # 1번 for문 (>>> 2~9까지의 숫자)   1단도 넣으려면 range(1,10)
    for j in range(1,10) :  # 2번 for문 (>>> 1~9까지의 숫자)
        print(i*j, end=' ') 
    print(' ') 
    
# input, range 사용 구구단     
(i, multi) =(0,0)
multi = int(input("단을 입력하세요:"))
for i in range(1,10,1):
    print('%d X %d = %d' % (multi, i, multi*i))

# 1부터 10까지 더하기
i = 0
j = 0
for i in range(1,11) :
    j = j+i    
    print(j)

# 리스트 내포 사용하기 : 리스트 안에 (for문을 포함하는 리스트 내포list conprehension)를 사용하면 편리
# 사용전 : a리스트의 각항목에 3곱해서 result 리스트에 담기 .append(x) : 마지막 요소로 집어넣기 
a = [1,2,3,4]
result = []
for num in a :
    result.append(num*3) # num == i
print(result)            # print를 탭하고 쓰면 for구문의 모든 결과가 다 나옴 
    
r = []
for i in a :
    r.append(i*3)
print(r)

# 리스트 내포 문법 ) a = [표현식 for 항목 in 반복가능객체 if 조건]   
# 예: a = [1,2,3]  apple = [i*5 for i in a if a<4]  print(apple)

# 사용 후
a = [1,2,3,4]
r = [i*3 for i in a]
print(r)
# a중 3배해서 짝수만 골라서 넣기
a = [1,2,3,4]
r1 = [i*3 for i in a if i%2==0]
print(r1)

# for문을 2개 이상 사용하기도 가능
#  [표현식 for 항목1 반복가능객체1 if 조건1
#          for 항목2 반복가능객체2 if 조건2 ... ]
# 구구단을 한번에 다 표시 
result = [x*y for x in range(2,10)
    for y in range(1,10)]
print(result)


#%%
#%% <Ch 04 프로그램의 입력과 출력 - 함수, 입출력, 파일읽고쓰기>
# 커피 자판기 코드 
coffee = 0
coffee = int(input("어떤 커피?1.보통 2.설탕 3 블랙:"))
print()       # 빈칸 출력 
print("#1.뜨거운 물을 준비한다")
print("#2.종이컵을 준비한다")
      
if coffee == 1:
    print("#3.보통")
elif coffee == 2 :
    print("#3.설탕커피")
elif coffee == 3 :
    print("#3.블랙커피")
else: 
    print("#3.아무거나\n")

print("#4.물을 붓는다")
print("#5.스푼으로 젓는다")
print()
print("손님 여기있습니다.")


# 커피 자판기 코드 + 함수

# 1) 전역변수 선언 
coffee = 0  
# 2) 함수 선언 
def coffee_machine(button):  #parameter: button 
    print()     
    print("#1.(자동)뜨거운 물을 준비한다")
    print("#2.(자동)종이컵을 준비한다")
          
    if button == 1:
        print("#3.(자동)보통")
    elif button == 2 :
        print("#3.(자동)설탕커피")
    elif button == 3 :
        print("#3.(자동)블랙커피")
    else: 
        print("#3.(자동)아무거나\n")

    print("#4.(자동)물을 붓는다")
    print("#5.(자동)스푼으로 젓는다")
    print()
# 3) 메인코드 작성 (여기에서 먼저 돌아감 )
coffee = int(input("어떤 커피?1.보통 2.설탕 3 블랙:"))
coffee_machine(coffee)
print("a손님 여기있습니다.")

coffee = int(input("어떤 커피?1.보통 2.설탕 3 블랙:"))
coffee_machine(coffee)
print("b손님 여기있습니다.")

coffee = int(input("어떤 커피?1.보통 2.설탕 3 블랙:"))
coffee_machine(coffee)
print("c손님 여기있습니다.")



#%% 04-1) 함수

# 파이썬 함수의 구조 
# def 함수이름(매개변수) : 
    #수행할문장1 
    #수행할문장2 
    # ...

# def : 함수를 만들 때 사용하는 예약어 

# 함수의 이름은 add, a,b값을 받으면 결과값은 a+b 
def add(a,b) :
    return a+b 
# 함수 사용
a = 3
b = 4
c = add(a,b)
print(c)

# 매개변수parameter: 함수에 입력으로 전달된 값을 받는 변수(a,b)
# 인수argument:함수를 호출할때 전달하는 입력값(3,4)
def add(a,b):
    return a+b
print(add(3,4))


# 여러가지 함수 : 입력값 -> 함수 -> 결괏값 : 입력/결과값의 존재유무에 따라 4가지 유형으로 나뉨

# 1-0)일반적인 함수(입력ㅇ, 결과ㅇ)    
# def 함수이름(매개변수): \n수행할문장.. return 결과값     
def add(a,b):
    result = a+b
    return result   # a+b의 결과값 반환
# 1-1)사용법 :   결과값 받을 변수 = 함수이름(입력인수1, 입력인수2 ...)
a = add(3,4)
print(a)

# 2-0)입력값이 없는 함수(입력x , 결과 o)
def say():
    return 'hi'
# 2-1)사용법 :   결과값 받을 변수 = 함수이름()
a= say()
print(a)

# 3-0)결과값이 없는 함수(입력o, 결과x)
def add (a,b):
    print('%d, %d의 합은 %d 입니다.' % (a,b,a+b))
# 3-1)사용법 : 함수이름(매개변수1,매개변수2)
add(3,4)     #>>> 3, 4의 합은 7 입니다. 
# 주의) 결과값은 오직 return 명령어로만 받을 수 있고, 위는 단지 print
a = add(5,6)
print(a)      #>>>None <- 거짓을 나타내는 자료형

# 4-0) 입력값과 결과값이 없는 함수(입력x, 결과x) : 입력인수를 받는 매개변수도 없고 리턴도 없음
def say():
    print('hi')
# 4-1) 사용법 : 함수이름()
say()


# 매개변수 지정하여 호출하기(함수를 호출할때 매개변수 지정) 순서에 상관없이 호출가능
def add(a,b):
    return a+b
result = add(a=3, b=7)
print(result)
 
result = add(b=5, a=3)
print(result)


# 입력값이 몇개가 될지 모를 때 : def 함수이름( * 매개변수) : 수행할 문장 ...
# *매개변수 : 입력값을 전부 모아서 튜플로 만들어줌   / 보통 *args 로 많이 사용 
def add_many(*args) :
    result = 0
    for i in args:
        result += i
    return result 

result = add_many(1,2,3,4,5,6,7,8,9,10)
print(result)

# 여러개의 입력을 처리할 때
def add_mul(choice, *args):
    if choice =='add':   # 매개변수 choice에 add입력받을때
        result = 0
        for i in args:
            result += i  # *args에 입력받은 모든 값을 더함
    elif choice == 'mul':
        result = 1
        for i in args:
            result *= i  # *args에 입력받은 모든 값을 곱한다  result = result * i 
    return result

result = add_mul('add', 1,2,3,4,5)
print(result)    #>>> 15
result = add_mul('mul', 1,2,3,4,5)
print(result)    #>>> 120

result = add_mul('mul', 2,4,6,8,10)
print(result)


# 키워드 파라미터 : **매개변수  **kwargs 로 자주쓰임 
# 밑의 kwargs는 딕셔너리가 되고 key=value형태로 결과값이 그 딕셔너리(kwargs)에 저장됨
def print_kwargs(**kwargs) :
    print(kwargs)
    
print_kwargs(a=1)   #>>> {'a': 1}      
print_kwargs(name= 'foo', age=3)  #>>> {'name': 'foo', 'age': 3}

# 함수의 결과는 언제나 하나
def add_and_mul(a,b):
    return a+b, a*b
result = add_and_mul(3,4)   # >>>(7,12) 튜플 형태로 결과반환! 

result1, result2 = add_and_mul(3,4)  # 1개의 튜플 값을 2개의 결과값으로 반환할때  
print(result1, result2)    #>>> 7  12

# return이 2개일때 맨 위의 리턴만 실행!  / 리턴문을 만나는 순간 결과값 돌려주고 함수 빠져나감 
def add_and_mul1(a,b):
    return a+b
    return a*b
result = add_and_mul1(2,3)   
print(result)    #>>> 5  


# return의 또다른 쓰임 -> 단독사용시 함수 빠져나갈 수 있음
def say_nick(nick):
    if nick =='바보' :
        return
    print("나의 별명은 %s입니다." %nick)
    
say_nick('nickname')  #>>> 나의 별명은 nickname입니다.
say_nick('바보')      #>>> 결과값 안나옴 


# 매개변수에 초기값 미리 설정하기 : default는 항상 맨 마지막에 넣기!
def say_myself(name, old, man=True):  # man=T이며 남자입니다. F면 여자입니다.
    print('내이름은 %s' %name)
    print('나이는 %d살'%old)
    if man: 
        print('남자입니다')
    else :
        print('여자입니다')
# 함수 사용
say_myself('박응용',27)
say_myself('박응용',27, True)
say_myself('박응선',27, False)

# man=True 가 중간에 들어가면 (name, man=True, old) 로 함수를 만들면 오류!(27을 man,old중 어디에 넣을지 모르기때문)

# 함수 안에서 선언한 변수의 효력범위 : 함수안에서만 사용됨! 
a = 1 #함수 밖 변수a
def vartest(a):
    a +=1
vartest(a)
print(a)

vartest(3)
print(a)    #>>> NameError: name 'a' is not defined

# 함수 안에서 함수 밖의 변수를 변경하는 법 2가지
# 1)
a = 1
def vartest(a):
    a +=1
    return a
a = vartest(a)    # a(함수밖 변수)에 vartest를 대입
print(a)
# 2) global 명령어  : 함수밖 변수를 직접 사용하겠다는 뜻 즉 이방법은 비추! 
a = 1 
def vartest():
    global a
    a +=1
vartest()
print(a)


# lambda : def와 동일한 역할 (예약어중 하나) , 함수를 한줄로 간결하게 만들때 사용
# lambda 매개변수1, 매개변수2, ...: 매개변수를 사용한 표현식
add = lambda a,b : a+b
result = add(3,4)
print(result)


# 실습 

def minus (a,b):
    return a-b
result = minus(4,10)
print(result)

def maxvar(a,b):
    if a > b :
        return a
    else :
        return b
result= maxvar(3,7)
print(result)



#%% 04-2) 사용자 입력과 출력 
# input 사용자 입력 : 사용자가 입력한 값을 변수에 대입하고 싶을 때

a= input()  # 콘솔or 아나콘다 프롬프트 에서 해야 나옴 
a   

# 프롬프르 값을 띄워서 입력받기
input('질문내용')
number = input('숫자를 입력하세요:')  # 입력한 숫자를 number에 할당(str로 저장됨)
print(number)

# print 자세히 알기 - 기존) 자료형 출력
a= 123
print(a)
a = 'python'
print(a)
a = [1,2,3]
print(a)

# print - 큰따옴표로 둘러싸인 문자열은 +연산(연결하기)과 동일하다
print("life" "is" "too short")
print("Life"+"is"+"too short")

# print - 문자열 띄어쓰기는 콤마로 한다.
print("Life","is","too short")

# print - 한줄에 결과값 출력하기  print(i, end = ' ')
for i in range(10):
    print(i, end = ' ')


#%% 04-3) 파일 읽고 쓰기

# 파일 생성하기 : 파일 객체 = open("파일위치/파일이름.확장자", '파일열기모드')
# 파일열기 모드 : r(읽기모드), w(쓰기모드, 이미존재할 경우 다 지워지고 새로), a(추가모드, 파일의 마지막에 새로운 내용을 추가할때 )    
f = open("C:/새파일.txt", 'w')
f.close()   # 열려있는 파일 객체 닫아주는 역할 

# 파일을 쓰기 모드로 열어 출력값 적기
f = open("C:/new1.txt",'w')
for i in range(1,11) :
    data = "%d번째 줄입니다.\n"%i
    f.write(data)
f.close()


# 이건 모니터에 출력하는 것, 위는 파일에 저장! 
for i in range(1,11) :
    data =  "%d번째 줄입니다.\n"%i
    print(data)


# 외부에 저장된 파일 읽는 여러가지 방법
    
# 1)readline 함수 사용 -- 가장 첫번째줄만 출력    type(line)= str
f = open("C:/new1.txt",'r')  # read mode 
line = f.readline()
print(line)
f.close()
# 모든 줄을 읽어서 출력하려면 while loop사용
f = open("C:/new1.txt",'r')
while True :
    line = f.readline()
    if not line : break     #>>> 모든 줄 출력하고 <function TextIOWrapper.close>
    print(line)
f.close

# 2) realines 함수 이용 - 모든줄을 읽어서 각각의줄이 요소가 되어 ->  리스트로 돌려줌  type(lines) = list
f = open("C:/new1.txt",'r')
lines = f.readlines()
for line in lines :
    print(line)
f.close()

# 3) read 함수 이용 - 파일의 전체 내용을 문자열로 돌려줌    type(data) = str 
f = open("C:/new1.txt",'r')
data= f.read()
print(data)
f.close()


# 파일에 새로운 내용 추가하기 
# 원래값 유지 + 새로운 값 추가 -> 추가모드 'a'로 열면 됨 
f = open("C:/new1.txt",'a')
for i in range(11,20):
    data= "%d번째 줄입니다.\n"%i
    f.write(data)
f.close()

# with문과 함께 사용하기 : 파일을 열고 닫는 것을 자동으로 처리
with open ("C:/foo.txt","w") as f:
    f.write("Life is too short, you need python")

# sys모듈로 매개변수 주기  확인필요() p.177
# 명령 프롬프트(DOS) 에서 C:\type a.txt <-type명령어는 바로 뒤에 적힌 파일 이름을 인수로 받아 그 내용을 출력해줌 
# 명령 프롬프트 명령어 [인수1 인수2 ...] / 파이썬에서도 sys모듈을 사용해 매개변수를 직접 줄 수있음

# sys1.py
import sys
args = sys.argv[1:]    
for i in args:
    print(i)
# 입력받은 인수를 for문을 사용해 차례대로 하나씩 출력하는 예 
# sys모듈의 argv는 명령창에서 입력한 인수를 의미, argv[0]은 파일이름 sys1.py가 되고 argv[1]부터는 뒤에 따라오는 인수가 차례대로 argv의 요소가 된다.
# sys1.py aaa bbb ccc <- 순서대로 argv[0] argv[1] argv[2] argv[3] 

# sys2.py    소문자를 대문자로 바꿈(upper)
import sys
args = sys.argv[1:]    
for i in args:
    print(i.upper(), end = ' ')
    
    
# 파일 불러오기 예제 
with open ("C:/dream1.txt", "r") as a:
    data= a.read()
    print(data)


# 텍스트의 통계정보를 읽어와야 할때 split(), len() 함수 사용
with open("C:/dream1.txt",'r') as my_file:
    contents = my_file.read()
    word_list = contents.split(" ")    # 빈칸 기준으로 단어를 분리하여 word_list에 할당
    line_list = contents.split("\n")   # 한 줄씩 분리하여 line_list에 할당 
    
print("총 글자수 :", len(contents))
print("총 단어수 :", len(word_list))
print("총 줄수 :", len(line_list))

#%% 
#%% < CH 05 파이썬 날개 달기 - 클래스, 모듈, 패키지, 예외처리, 내장/외장함수>
#%% 05-1) 클래스 class 

# 계산기는 이전에 계산한 결과값을 항상 메모리 어딘가에 저장해야 한다.
# 기존값 저장후 더하기! 
result = 0
def add(num) :
    global result
    result += num
    return result

print(add(3))
print(add(4))

# 계산기가 두개 필요할 때 
result1 = 0
result2 = 0

def add1(num):  # 계산기 1
    global result1
    result1 +=num
    return result1

def add2(num):  # 계산기 2
    global result2
    result2 +=num
    return result2

print(add1(3))
print(add1(4))
print(add2(3))
print(add2(7))

# 더 복잡한 계산기를 만들기 위해 클래스 사용 (결과는 동일)   #self 는 instance
class cal:
    def __init__(self):        #__init__ : initialize초기화 
        self.result = 0        # 초기화할 코드 
    def add(self, num) :
        self.result += num
        return self.result
cal1= cal()
cal2= cal()

print(cal1.add(3))
print(cal1.add(4))
print(cal2.add(3))
print(cal2.add(7))

# 생성자의 개념 : 인스턴스를 생성하면서 필드값을 초기화시키는 함수
# 생성자의 기본형태 : __init__() 언더바는 파이썬에서 예약해놓은 것 (이걸로 변수명 지정 하지말것) 
# 생성자 문법) class 클래스명 : \n def __init__(self) : \n초기화할 코드 

# 생성자(__init__)는 인스턴스를 생성하면 무조건 호출되는 method이고,  self외에 매개변수를 사용하지 않음
# method의 첫번째 매개변수에 self를 사용하는 이유는 매서드 안에 필드(result)에 접근하기 위해서
# 즉 self를 매개변수로 보기보다는 매서드 안에 무조건 사용해야하는 것으로 이해하는 것이 좋고. 매서드 안에서 필드에 접근할 일이 없다면 self는 생략가능

# class:객체를 만들기 위한 사용자정의 자료형(쿠키틀)
# method: 객체의 기능을 반영하여 클래스 내부에 선언된 함수
# instance :클래스를 기반으로 만들어진 구체적인 객체  
# object: 클래스에 의해 정의된 데이터 구조의 인스턴스 


# cal 클래스로 만든 별개의 계산기인 cal1,cal2(<- 파이썬에선 객체라고 부름)가 각각 역할수행
# 결과값은 각각 독립적으로 유지, 계산기를 늘리려면 객체만 추가하면 됨 

# 빼기 기능을 더하려면 밑의 블록을 추가하면 됨 
def sub(self, num) :
    self.result -= num
    return self.result 

    
# 더하기 빼기 같이 있는계산기 확인()
class cal:
    def __init__(self):       
        self.result = 0       
    def add(self, num) :
        self.result += num
        return self.result
    def sub(self, num) :
        self.result -= num
        return self.result 
cal1= cal()
cal2= cal()
cal3= cal()

print(cal1.add(3))
print(cal1.add(33))
print(cal2.add(3))
print(cal2.add(333))
print(cal3.sub(3333))
print(cal3.sub(333))

# 클래스(class)== 쿠키틀  
# 객체(object)==쿠키  : 클래스로 만든 객체를 인스턴스 라고도 함  ->  a 객체는 cookie의 인스턴스
# (a는 인스턴스 보다는)  a는 객체  /  (a는 cookie의 객체 보다는)  a는 cookie의 인스턴스  라는 표현이 더 잘 어울림 
class cookie:
    pass
a = cookie()  #cookie()의 결과값을 돌려받은 a가 객체 
b = cookie()


# class 클래스 이름[(상속 클래스명)] :
#     클래스 변수1
#     클래스 변수2
#     ...
#     def 클래스함수1(self[,인수1,인수2,...]) :
#         수행할 문장1
#         수행할 문장2
#         ..
#     def 클래스함수2(self[,인수1,인수2,...]):
#         수행할 문장1
#         수행할 문장2 


# 사칙연산 클래스만들기

# 0) 클래스 fourcal 만들기 
class fourcal:
    pass
# 1) 객체 a 생성
a= fourcal()
print(type(a))   #>>> <class '__main__.fourcal'>
# 2) 객체에 숫자 지정할 수 있게 만들기 ->  a.setdata(4,2) 가 가능하게끔  setdata함수를 만들었음 (클래스 내부의 함수: method!)
class fourcal :
    def setdata(self, first, second) :   # def method_name(매서드의 매개변수)
        self.first = first               # 매서드의 수행문
        self.second = second
        
# 2-1) setdata method의 매개변수 : self, first, second 3개의 입력값을 받음
# self(첫 번째 매개변수)에는 setadata method를 호출한 객체a나 b가 자동으로 전달되기 때문에 2개의 값(first, second)만 입력
# 파이썬 매서드에서 첫번째 매개변수 이름은 관례적으로 self사용, 객체를 호출할때 호출한 자기자신이 전달되기 때문
a= fourcal()
a.setdata(4,2)  
print(a.first)
b= fourcal()
b.setdata(3,7)
print(b.second)

# 참고) 또다른 매서드 호출방법 .setdata 앞에 뭘 써주느냐에 따라 괄호안에 a를 불러주거나 안부르거나 차이 있음 
a= fourcal()
fourcal.setdata(a,4,2)

a=fourcal()
a.setdata(4,2)

# 2-2) setdata method의 수행문
# self.first = first \n self.second= second  와 a.setdata(4,2)를 호출하면 self.first=4 즉 a.first =4로 해석가능 
# a.first =4문장이 수행되면 a객체에 객체변수first가 생성되고 값 4가 저장됨 (객체변수: 객체에 생성되는 객체만의 변수 )

a=fourcal()
a.setdata(4,2)
print(a.first)  # a객체에 객체변수 first 생성됨
print(a.second) # b객체에 객체변수 second 생성됨 

a = fourcal()
b = fourcal()

a.setdata(4,2)
print(a.first)

b.setdata(3,7)
print(b.first)
# 같은 first지만 클래스로 만든 객체의 객체변수(first)는 독립적인 값을 유지한다.(3과 4) 
# id함수를 이용해 명확하게 표현가능(주소값이 다름! )
id(a.first)  #>>> 1948174160
id(b.first)  #>>> 1948174128

# 3) 더하기 기능 만들기
class fourcal :
    def setdata (self, first, second) :
        self.first = first
        self.second = second 
    def add(self) :
        result = self.first +self.second
        return result
    
a = fourcal()
a.setdata(4,2)
print(a.add())
# self.first -> a.first -> 4 -> print(a.add())를 통해 4+2=6 돌려줌 

# 4) 곱하기 빼기 나누기 기능 만들기   self 지정했으면, 함수의 괄호에도 self로 
class fourcal :
    def setdata(self, first, second) :
        self.first = first
        self.second = second
    def add(self) :
        result = self.first +self.second
        return result
    def mul(self) :
        result = self.first * self.second
        return result
    def sub(self) :
        result = self.first - self.second
        return result
    def div(self) :
        result = self.first / self.second
        return result
    
a= fourcal()
b= fourcal()
a.setdata(10,20)
b.setdata(30,40)

a.add()
a.mul()
a.sub()
a.div()

b.add()
b.mul()
b.sub()
b.div()

# 5)생성자constructor : 객체가 생성될 때 자동으로 호출되는 method 
a.fourcal()
a.add()
# 둘만 수행하면 setdata method 수행(= a의 객체변수 first, second생성) 하지 않아 오류뜸!

# 객체에 초기값을 설정해야할 필요가 있을때 setdata 같은 method를 호출하려 초기값을 설정하기 보다는 생성자를 구현하는 것이 안전
# 맨 윗줄에 def __init__() 추가 
class fourcal():
    def __init__(self,first, second):
        self.first = first
        self.second = second
    def setdata(self, first, second) :
        self.first = first 
        self.second = second
    def add(self) :
        result = self.first +self.second
        return result
    def mul(self) :
        result = self.first * self.second
        return result
    def sub(self) :
        result = self.first - self.second
        return result
    def div(self) :
        result = self.first / self.second
        return result
    
a = fourcal() #>>> TypeError: __init__() missing 2 required positional arguments: 'first' and 'second'
# => 1st,2nd에 해당하는 값을 전달하여 객체 생성해야함
a = fourcal(4,2)  # -> 매개변수:값 일때, self:생성되는객체(a), first:4, second:2 로 대입됨
print(a.first)  #>>> 4

# 계산기 실제 사용
a= fourcal(5,10)
a.add()
a.div()

# 5) 클래스의 상속 inheritance 문법=>  class 클래스명(상속할 클래스 이름)  => class newclass_name(oldclass_name)
# 기존 클래스가 라이브러리 형태거나 수정불가일때 상속으로 사용! / 클래스의 기능을 확장시킬 때 
# 상속개념을 이용하여 fourcal 클래스에 a**b기능 추가한 morefourcal 만들기
class morefourcal(fourcal):
    pass

# forcal을 상속받은 morefourcal에서 사칙연산 가능
a = morefourcal(10,5)
a.add()
a.sub()
a.mul()
a.div()

class morefourcal(fourcal) :
    def pow(self):      # 제곱을 하기위한 pow method 추가
        result = self.first ** self.second
        return result
    
a = morefourcal(2,4)
a.pow()   #>>> 2**4 ==16


# 6) 매서드 오버라이딩 overriding 덮어쓰기! 
# 부모 클래스(old)에 있는 기존 매서드를 동일한 이름으로 다시 만드는것! (== 상속받는 클래스에서 함수를 재구현하는 것 )

a= fourcal(4,0)
a.div()  #>>> ZeroDivisionError: division by zero 

# 4/0 = 0으로 결과가 나오게 만들기!
class safecal(fourcal) :
    def div(self) : 
        if self.second == 0:
            return 0
        else :
            return self.first /self.second
a= safecal(4,0)
a.div()  #>>> 0

# 204쪽 연습문제 
class multical(fourcal) :
    def mul(self) :
        if self.second == 0 :
            return "Fail"
        else :
            return self.first * self.second
       

a = multical(4,0)
a.mul()



# 클래스 변수 
# 객체 변수 : 다른 객체들에 영향을 받지 않고 독립적으로 그 값을 유지함
class fam :
    lastname = '김'  # fam class에서 선언한 lastname이 클래스변수 
    
print(fam.lastname)

a= fam()
b=fam()
print(a.lastname)
print(b.lastname)

fam.lastname = "박"   # 여기서 수정하면 다 바뀜
print(a.lastname)
print(b.lastname)

id(fam.lastname)   # 클래스 변수는 id 즉 주소값도 동일함 모두 같은 메모리! 
id(a.lastname)
id(b.lastname)



# %% 05-2) 모듈 
# 함수/변수/클래스를 모아놓은 파일

#모듈 만들기
# mod1.py
def add(a,b):
    return a+b
def sub(a,b) 
    return a-b

# 모듈 불러오기  -> 오른쪽 위에 디렉토리에 저장되어있어야함
# 디렉토리 옮길때는 cmd or prompt에서 cd C:/  <- 디렉토리 지정
    
   
# cmd 에서 디렉토리로 이동(C:/module) python 실행
# 1) import 모듈이름(확장자 제외)
import mod1
print(mod1.add(3,4))
print(mod1.sub(3,4))
# 2) from 모듈이름 import 모듈함수
from mod1 import add
add(3,4)

# 3) 여러개 불러오기 
# from 모듈이름 import 모듈함수1, 모듈함수2,... 
# from 모듈이름 import * : 모듈의 모든 함수 가지고옴 
from mod1 import add, sub
add(3,4)
sub(3,4)
from mod1 import*


# mod1_1  = mod1 + print문
def add(a,b):
    return a+b
def sub(a,b):
    return a-b
print(add(1,4))
print(sub(4,2))

# cmd 에서 import mod1_1 하면 프린트문까지 같이 자동으로 실행됨
# 프린트문 위에 아래의 코드 넣어줘야함 
# 직접 이 파일을 실행하면 if문이 참 -> if 블록 실행
# 인터프리터(cmd)나 다른 파일에서 이모듈을 불러서 사용하면 거짓이 되어 블록 수행 안함 
if __name__ =="__main__":
    print(add(1,4))
    print(sub(4,2))

# __name__ 변수 : 내부적으로 사용하는 특별한 변수이름 직접 mo1_1을 실행하면 __name__변수에 __main__값이 저장 
# 파이썬 쉘이나 다른 모듈에서 import할경우에는 __name__변수에 mod1_1.py의 모듈 이름값인 mod1_1이 저장되어 if문이 거짓  


# 클래스나 변수 등을 포함한 모듈 
   
# mod2.py   -> C:/module에 저장 
PI = 3.141592
class math :
    def solve (self, r) :
        return PI * (r**2)

def add(a,b):
    return a+b    

#cmd (C:/module), python 
import mod2
print(mod2.PI) #>>>3.141592

a= mod2.math()
print(a.solve(2))  #>>> 12.566368
print(mod2.add(mod2.PI, 4.4))  #>>> 7.541592


# 다른 파일에서 모듈 불러오기  
# modtest.py 만들기  주의) mod2, modtest가 동일 디렉토리에 있어야함 
import mod2
result = mod2.add(3,4)
print(result)

# 모듈을 불러오는 다른 방법
# 0) cmd에서 새로운 디렉토리 만들고 mod2옮기기 (파일(module)안에 mod2.py가 있고, mymod새파일 하나 더생성해서 옮기)
C:/module>mkdir mymod   
C:/move mod2.py mymod
# 1)sys.path.append(모듈을 저장한 디렉토리) 사용하기
python> import sys
sys.path         # <- 모든 파이썬 라이브러리가 설치되어있는 디렉토리 확인가능 
# 파이썬 모듈을을 sys.path에 추가하면 아무곳에서나 불러올 수 있음
sys.path.append("C:/module/mymod")  # sys.path하면 맨뒤에 추가되어있음 
# 되는지 확인
import mod2
print(mod2.add(3,5))


# 2) pythonpath 환경변수 사용해서 모듈 불러오기
# set명령어를 이용해 모듈이 있는 디렉토리를 환경변수에 설정, 추가 작업 없이 모듈 사용가능 
C:/module> set PYTHONPATH=C:/module/mymod
python> import mod2 \n print(mod2.add(3,5))


#%% 05-3) 패키지 - .(점)을 사용하여 파이썬 묘듈을 계층적(==디렉토리 구조)으로 관리가능
# 모듈이름이 a.b -> a는 패키지, b는 모듈 
# p.216~p.221 참고


#%% 05-4) 예외처리  - try, except 활용해서 오류 무시/처리

# 오류 예
f = open("없는파일",'r') #>>>  No such file or directory: '없는파일'
4/0      # >>> ZeroDivisionError: division by zero
a = [1,2,3]
a[4]  #>>> IndexError: list index out of range

# 오류 예외처리 기법 
# try...except

# try : 
#    ...
# except [발생오류[as 오류 메세지 변수]] : \n ...

# 1) try, except만 쓰는 방법 -> 오류 종류에 상관없이 오류가 발생하면 except문 수행
try:
    4/0
except : 
    print(0)  #>>> 0
    
# 2) 발생오류만 포함한 except문 -> 미리정한 오류 이름과 일치할 때만 except문 수행
try :
    4/0
except  IndexError :
    print(0)   # >>> 발생오류가 일치하지 않아 error라고 뜸 

# 3) 발생오류, 오류 메세지변수까지 포함한 except문 -> 내용까지 알고 싶을 때 사용
try :
    4/0
except ZeroDivisionError as e:
    print(e)     #>>> division by zero
    

# try...finally : try문 수행도중 예외 발생여부와 상관없이 항상 finally절 수행됨 (리소스를 close할때 많이 사용)
f= open('C:/foo.txt','w')
try :
    print(f)  # 무엇인가 수행 
finally :
    f.close()    #>>> 예외 발생여부와 상관없이 f.close()로 열린파일 닫을 수 있음
    
# 여러개의 오류 처리하기 
try :
    a = [1,2]
    print(a[3])
    4/0
except ZeroDivisionError :
    print("0으로 못나눠")
except IndexError :
    print("인덱싱 안돼")
#>>> print(a[3]) 이 먼저라 인덱싱 안돼 만 나옴
    
try :
    a = [1,2]
    print(a[3])
    4/0
except(ZeroDivisionError, IndexError) as e:
    print(e)
    


# 오류 회피하기 - 특정오류 통과
try :
    f= open("없는 파일",'r')
except FileNotFoundError:
    pass   


# 오류 일부러 발생시키기 - raise명령어로 오류 강제발생
#  NotImplementedError : ㅍ파이썬 내장오류, 꼭 작성해야하는 부분이 구현되지 않았을 때 일부러 오류를 발생시키기 위해 
    
# 예: bird클래스를 상속받는 자식 클래스는 반드시 fly를 구현하도록 만들려면
class bird :
    def fly(self):
        raise NotImplementedError
    
class eagle(bird):   #부모:bird ->(상속) -> 자식eagle
    pass
e = eagle()
e.fly()    #>>> NotImplementedError 

#오류 발생되지 않게 하려면 상속받는 클래스에 함수 구현해야함 
class eagle1(bird):
    def fly(self):
        print('very fast')
e1=eagle1()
e1.fly()


# 예외만들기 - 특수한 경우에만 예외처리를 하기위해 종종 예외를 만들어서 사용 
# 파이썬 내장 클래스인 Exception클래스를 상속하여 만들 수 있음
class myerror(Exception) :
    pass

def say_nick1(nick) :
    if nick == '바보':
        raise myerror()
    print(nick)

say_nick1('천사') #>>> 천사
say_nick1('바보')  #>>>     raise myerror()  \n\n myerror

# 위의 예외 <-  처리기법을 사용하기
try:
    say_nick1("천사")
    say_nick1("바보")
except myerror :
    print("허용되지 않는 별명")

say_nick1("바보")


# 아래처럼 오류메세지를 사용하려면 코드 작성하고
try :
    say_nick1("천사")
    say_nick1("바보")
except myerror as e :
    print(e)
# myerror class에 추가적으로 __str__메서드 구현 필요
class myerror(Exception):
    def __str__(self):
        return "허용되지 않는 별명입니다"

say_nick1("바보")





#%% 05-5) 내장함수 
# print, del, type 등은 파이썬 내장함수

# 파이썬 내장함수

# abs(x) : 절대값
abs(-3)

# all(x) : 반복가능한(iterable) 자료형x를 입력인수로 받고, x가 모두 참이면 T, 거짓이 한개라도 있으면 F
all([1,2,3])  #T
all([1,2,3,0]) #F

# any(x) : 하나라도 참이 있으면 T, 모두 거짓이면 F
any([1,2,3,0])
any([0,""])    

# chr(i) : 아스키(ASCII) 코드값을 입력받아 해당하는 문자 출력
# 아스키 코드: 0~127 사이의 숫자를 각각 하나의 문자나 기호에 대응시켜놓은것
chr(97)    # 97은 a
chr(48)    # 48은 숫자 0

# ord(c) : 문자 -> 아스키코드로 변환
ord('a')
ord('b')    
ord('1')

# dir() : 객체가 자체적으로 가지고 있는 변수나 함수를 보여줌 
# 다음 예는 숫자, 튜플,리스트, 딕셔너리 객체 관련 함수들(=메서드)를 보여줌
dir(1)
dir((1,2))
dir([1,2,3])
dir({'1':'a'})    
    
# divmod(a,b) : 2개의 숫자를 입력받고, a를 b로 나눈 몫과 나머지를 튜플 형태로 반환
divmod(10,3)
10//3
10%3    

# enumerate(리스트,튜플, 문자열) : 열거하다. 순서가 있는 자료형을 입력받아 
# 인덱스값을 포함하는 enumerate객체로 돌려줌     
# for문과 함께 사용하여, 반복되는 구간에서 객체가 현재 어느위치를 알려주는지 인덱스 값이 필요할 때 사용  
for i, name in enumerate(['body','foo','bar']) :
    print(i,name)
    
# eval('expression') : 실행가능한 문자열을 입력받아 문자열을 실행한 결과값으로 반환 / 파이썬 함수나 클래스를 동적으로 실행하고 싶을 때
# 문자열 인경우 ("'str1' + 'str2'") 로 써야함
type(eval('1+2'))   # -> int 
eval("'hi'+'jin'")
eval('divmod(4,3)')

# filter : 첫번째 인수로 함수이름, 두번째인수로 그 함수에 차례로 들어갈 반복가능한 자료형
# 그리고 자료형 요소가 함수에 입력되었을대 반환값이 참인 것만 걸러내서 반환
# positive.py
def pos(l):
    result =[]  # 반환 값이 참인 것만 받을 변수
    for i in l :
        if i > 0:
            result.append(i)  # i가 0보다 크면 result에 추가 
    return result
print(pos([1,-3,2,0,-5,6]))  #>>> [1,2,6]
# filter를 사용하면
def pos1(x):
    return x> 0
print(list(filter(pos1,[1,-3,2,0,-5,6])))
# lambda를 사용하면 더욱 간편
list(filter(lambda x: x>0, [1,-3,2,0,-5,6]))

# hex(x) : 정수값을 입력받아 16진수(hexadecimal)로 변환
hex(234)
hex(0)    

# id(object) : 객체를 입력받아 객체 고유주소값(레퍼런스) 을 반환
a = 3    
id(a)    

# input([prompt]) : 사용자 입력을 받는 함수     # [] 기호는 생략가능 하다는 뜻
# 매개변수로 문자열을 주면 그 문자열은 프롬프트가 됨
a = input()  # >>> prompt에 입력하면 그 입력값을 a에 할당 
a
b = input("enter:")    
b    

# int(x) : 문자열형태의 숫자, 소수점이 있는 숫자를 정수 형태로 돌려줌 
int('3')
int(3.4)   #>>> 3    
float(3)   #>>> 3.0

# int(x, radix) : radix진수로 표현된 문자열x를 10진수로 반환
int('11',2)    #>>> 2진수로 표현된 11의 10진수 값 -> 3
int('1A',16)   #>>> 16진수로 표현된 1A의 10진수 값 -> 26

# isinstance(object, class)  : 첫번째 인수로 인스턴스, 두번째 인수로 클래스를 받는다.
# 입력으로 받은 인스턴스가 그 클래스의 인스턴스 인지를 판단하여 참이면 T, 거짓이면 F반환
class person :
    pass
a=person()
isinstance(a,person)   #>>> T
b=3
isinstance(b,person)

# len(s) :입력값 s의 길이(요소의 전체개수)를 돌려줌
len('python')
len([1,2,3,4,5,6])
len((1,'a'))

# list(s) : 반복가능한 자료형 s를 입력받아 리스트로 만들어줌 
list('python')
list((1,2,3,4,5))

a= [1,2,3]
type(a)
b=list(a)
b

# map(f, iterable) : 함수f와 반복가능한iterable 자료형을 입력받음 
# map은 입력받은 자료형의 각 요소를 함수f가 수행한 결과를 묶어서 돌려줌
def two_times(numberlist):
    result = []
    for number in numberlist :
        result.append(number*2)
    return result
result = two_times([1,2,3,4])
print(result)
# map을 사용하면
def two_times1(x) :
    return x*2
list(map(two_times1, [1,2,3,4]))
#lambda 사용하면
list(map(lambda a : a*2, [1,2,3,4]))

# max(iterable) 
max([1,2,3,4,100])
max('python') # a가 작은값 z가 큰값
min('pythona')

# min(iterable)

# oct(x) : 정수 형태의 숫자를 8진수 문자열로 반환
oct(34)
oct(12345)

# open(filename,[mode]) : 파일이름, 읽기 방법 입력받아 파일 객체를 돌려주는 함수 
# mode 생략시 default는 r (읽기전용모드)
# [mode] w:쓰기모드   r:읽기모드    a:추가모드    b:바이너리모드,   rb:바이너리 읽기모드  

# pow(x,y) : x**y
pow(2,4)


# range([start],stop [,step]) : for문과 함께 자주 사용, 입력받은 숫자에 해당하는 범위값을 반복가능한 객체로 만들어 돌려줌
# 인수가 1개면 0부터 시작
list(range(5))
# 인수개 2개면 시작부터 끝전까지 
list(range(1,7))  #>>> 1~6
# 인수가 3개면 시작,끝전,숫자사이거리(간격)
list(range(1,11,2))
list(range(10,-10,-1))  #-9까지 

# round(number [,ndigits]) : 숫자입력받고 반올림   [,ndigits]는 소수점 자리수
round(4.6234, 3) 
round(4.5)   #0.5초과 올림
round(4.51)

# sorted(iterable) : 입력값을 정렬하고 그 결과를 리스트로 돌려줌
sorted([3,5,1,3,2])
sorted(['apple','cina','banana','grape'])
sorted('zero')
sorted((1,4,2))
# sorted(1,3) 이렇게는 안됨 적어도 튜플!
# 리스트 자료형의 sort 함수도 있지만 리스트 객체 그자체를 정렬만 할뿐 반환하지 않음 

#str(object) : 문자열로 객체 반환
str(3)
str('hi')
str('hi'.upper())

# sum(iterable) : 입력받은 리스트나 튜플의 모든 요소의 합을 반환
sum((1,2))
sum([1,2,3,4,5])

# tuple(iterable) : 반복가능한 자료형을 입력받아 튜플로 반환   -> int는 iterable아님 
tuple(123)  #>>> TypeError: 'int' object is not iterable
tuple((1,2,3))
tuple([1,2,3])
tuple('abc')

# type(object) : 입력값의 자료형이 무엇인지 알려줌
type(open('test','w'))  #>>> _io.TextIOWrapper : 파일자료형

# zip(*iterable) : 동일한 개수로 이루어진 자료형을 묶어줌    # *iterable : 반복가능한 자료형 여러개를 입력가능
list(zip([1,2,3],[1,5,6]))
list(zip("asd","qwe"))


#%%
#%%% 05-6) 외장함수 == 파이썬 라이브러리

# sys 모듈 : 파이썬 인터프리터가 제공하는 변수와 함수를 직접 제어할 수 있게 해주는 모듈  p.247,248참고

# pickle : 객체의 형태를 유지하면서 파일에 저장하고 불러올 수 있게 하는 모듈
# pickle의 dump함수를 사용하여 딕셔너리 객체인 data를 그대로 파일에 저장 
# 예:) 딕셔너리 저장 
import pickle
f = open('test.txt','wb')
data = {1:'python', 2:'you need'}
pickle.dump(data,f)
f.close()

import pickle
f= open('test.txt','rb')
data1= pickle.load(f)
print(data1)

# os모듈: 환경변수나 디렉털, 파일등의 os자원을 제어할 수 있게 해주는 모듈
# 내 시스템의 환경변수값을 알고싶을때 
import os
os.environ
# 딕셔너리 형태로 반환하기 때문에 특정값 호출가능
os.environ['PATH'] 

# 디렉토리 위치 변경하기
os.chdir("절대경로")

# 디렉토리 위치 확인
os.getcwd()

# 시스템 명령어 호출하기 - os.system("명령어")
os.system("dir")
# 실행한 시스템 명령어의 결과값 반환 - os.popen("명령어")
f= os.popen("dir")
# 읽어들인 파일 객체의 내용을 보기위해 
print(f.read())
# 기타 os관련 함수
os.mkdir("")       # 디렉터리 생성
os.rmdir("")       # 디렉토리 삭제 (빈 파일이어야 함 )
os.unlink("")      # 파일삭제
os.rename(abc,defg)# abc라는 파일이름을 defg로 변경 

# shutil : 파일복사해주는 파이썬 모듈
import shutil 
shutil.copy("a.txt","b.txt")  #b가 디렉토리면 a이름으로 b디렉토리에 복사, 동일한경우 덮어씀 

# glob(pathname) : 디렉토리 안의 파일들을 읽어서 돌려줌 *나 ?등 메타 문자를 써서 원하는 파일만 읽을수 있음
import glob
glob.glob("C:/파일명*")

# tempfile.mktemp() : 파일을 임시로 만들어서 사용할때 - 중복되지 않는 임시파일의 이름을 무작위로 만들어서 반환
import tempfile
filename = tempfile.mktemp()
filename  #>>> 'C:\\Users\\a\\AppData\\Local\\Temp\\tmp1dlw0w5f' 

# tempfile.TemporaryFile() : 임시저장공간으로 사용할 파일 객체를 돌려줌 -> 기본적으로 바이너리 쓰기모드(wb) 
# f.close() 호출되면 자동으로 사라짐
import tempfile
f = tempfile.TemporaryFile() 
f.close()

# 시간관련 time모듈

import time
time.time() # 1970/1/1/00:00:00를 기준으로 지난시간을 초단위로돌려줌
time.localtime(time.time()) 
    #>>> time.struct_time(tm_year=2019, tm_mon=7, tm_mday=26, tm_hour=16, tm_min=29, tm_sec=46, tm_wday=4, tm_yday=207, tm_isdst=0)
time.asctime(time.localtime(time.time()) )  #>>> 'Fri Jul 26 16:30:24 2019'
time.ctime() #>>> 'Fri Jul 26 16:30:46 2019'   현재시간 반환
time.strftime('출력할 형식포맷코드',time.localtime(time.time()))
# 여러가지 포맷코드 p.254참조 
# %Y:연도4자리    %y:연도2자리      %b:달 줄임말     %B:달     %m:달[01,12]   
# %a:요일 줄임말  %A:요일(full)     %d:날(day)     %c:날짜와 시간     %Z:시간대 출력 
# %x: 현재설정된 지역에 기반한 날짜출력, %X: 현재 설정지역 시간 
# %H:24시간  %I:12시간   %M:분[01,59]  %p:am/pm  %S:초   
# %w:숫자로된요일(0:일요일)   %W:1년중 누적주(월요일을 시작)      %%:  문자% 
time.strftime('%x', time.localtime(time.time()))
time.strftime('%Z', time.localtime(time.time()))

# time.sleep 루프안에서 자주사용/ 일정한 시간간격을 두고 루프 실행가능
import time
for i in range(10):
    print(i)
    time.sleep(1)

# calendar
import calendar
print(calendar.calendar(2019))
calendar.prcal(2019)   # 위와 동일
calendar.prmonth(2019,7)

calendar.weekday(2019,7,26)  #>>>4 : 금요일 ! (0 이 월요일)
calendar.monthrange(2019,7)  # 월에 며칠까지 있는지 튜플형태로 돌려줌


# random 난수(규칙이 없는 임의의 수) 발생
# 0.0~1.0 사이의 실수 중에서 난수 돌려줌 
import random
random.random()

random.randint(1,10)  #1<=x<10 정수중 하나 돌려줌

#예제 함수1 - 리스트의 요소중에서 무작위로 하나 선택하여 반환 꺼내고 나서 pop매서드에 의해 사라짐
import random
def ranpop(data) :
    number = random.randint(0, len(data)-1)
    return data.pop(number)
if__name__ == "__main__" :
    data=[1,2,3,4,5]
    while data :
        print(ranpop(data))
# 직관적으로바꾼 함수
def ranpop1(data):
    number=random.choice(data)
    data.remove(number)
    return number
ranpop1([1,2,3,4,5])

# 리스트의 항목을 무작위로 섞고 싶을때
import random
data= [1,2,3,4,5]
random.shuffle(data)
data


# webbrowser - 자동실행! 
import webbrowser
webbrowser.open("http://google.com")
webbrowser.open_new("http://google.com")


# ㅆ스레드를 다루는 threading 모듈
# 컴퓨터에서 동작하고 있는 프로그램을 프로세스라고 한다. 1개의 프로세스는 1개의 일을 하지만, 스레드 사용시 2가지 이상 수행가능
# thread_test.py
import time
def long_task() :    # 한번에 오초가 걸리는 함수
    for i in range(5):
        time.sleep(1)
        print("working:%s\n" %i)
print("Start")
for i in range(5):  # 함수를 5번 반복 
    long_task()
print('end')

# 총 25초의 시간이 걸리지만 스레드를 사용하면 동시에 실행가능 (아래)
# thread_test1.py
import time
import threading

def long_task() :   
    for i in range(5):
        time.sleep(1)
        print("working:%s\n" %i)
print("Start")
threads =[]
for i in range(5): 
    t = threading.Thread(target = long_task)
    threads.append(t)
    
for t in threads:
    t.start()
    
print('end')

#start end 실행되고 00000 11111...이렇게 되는걸 수정하기 위해 -> 아래
# thread_test2.py
import time
import threading

def long_task() :   
    for i in range(5):
        time.sleep(1)
        print("working:%s\n" %i)
print("Start")
threads =[]
for i in range(5): 
    t = threading.Thread(target = long_task)
    threads.append(t)
    
for t in threads:
    t.start()
for t in threads:
    t.join()   # 스레드가 종료될 때까지 기다림 
    
print('end')



#%%
#%% <Ch 06 파이썬 프로그래밍>
# 입력과 출력을 생각하기! 그리고 plan 

# 06-1) 구구단함수
"""
0. 2단 구구단 만들준비
1. result=gugu(2) - 2를 입력값으로 주면 result라는 변수에 결과값 넣기
2. 2,4,...18까지 리스트 자료형 선택 
3. 필요한 함수작성

def gugu(n) :
    print(n)
gugu(2) #>>>2


4. 결과값을 담을 리스트 하나 생성 result =[]
print(n)은 삭제! 입력이 잘되는지 확인한것
5. 리스트에 요소를 추가하는 append 내장함수 사용시 
"""
def gugu(n) :
    result =[]
    result.append(n*1)
    result.append(n*3)
    result.append(n*4)
    result.append(n*4)
    result.append(n*5)
    result.append(n*6)
    result.append(n*7)
    result.append(n*8)
    result.append(n*9)
    return result
print(gugu(2))

"""
6. n* 다음에 오는 숫자만 다르게 들어가 있으니 반복문 적용하고
7. 완성하기 

"""
i = 1
while i <10:   # i가 10보다 작은 동안
    print(i)   # i 출력
    i += 1     # i값 1 증가 
    
# 구구단 함수
def gugu_final(n) :
    result =[]
    i = 1
    while  i < 10:
        result.append(n*i)
        i += 1
    return result
print(gugu_final(9))


# 06-2) 3과 5의 배수 합하기
# 1000미만의 자연수에서 3의 배수와 5의 배수를 합하시오
"""
입력값 : 1~999
출력값 : 3n, 5n
주의할 사항: 어떻게 찾고, 겹치면 어떡하지?
"""
#1) 1000미만의 자연수는
n =1 
while n <1000:
    print(n)
    n +=1
# for문을 사용하면
for n in range(1,1000) :
    print(n)
# 3의 배수는 -> 3으로 나눴을때 나머지가 0
for n in range(1,100) :
    if n % 3 == 0:
        print(n)
# 최종 풀이는
result= 0
for n in range(1,1000) :
    if n % 3==0 or n%5 ==0 :
        result += n
print(result)
# 만약 if문을 3,5 각각 쓰면 15배수는 2번 더해져서 안돼! 
   

# 06-3) 게시판 페이징하기 (게시판의 페이지수를 보여주는것)
"""
입력값 : 게시물 총 건수 m , 한 페이지 내의 게시물수 n
출력값 : 총 페이지 수 

총 페이지 수 = (총 건수/한페이지당 보여줄 건수 ) + 1 
"""
def total(m,n) :
    return m // n+1   # 소수점 버리려고 // 
total(5,10)    # 총 건수 5개, 1페이지에 10개 가능 -> 1
total(25,10)   # 3
total(30,10)   # 3이 아니라 4가 나옴 -> 나머지가 0일때 처리안함 함수 수정

def total(m,n) :
    if m%n == 0:
        return m//n
    else :
        return m//n+1
total(30,10)

# 06-4) 간단한 메모장만들기 p.278~ 


#%% <Ch 07 정규 표현식>
# regular expression : 정규표현식, 정규식 -> 복잡한 문자열을 처리할 때 사용하는 기법! 
# 예제(주민등록번호 뒷자리를 *으로 표시) 공백을 문자마다 나누고, 다시 조립하고 복잡하지만 정규식을 쓰면
import re
data = '''
park 800905-1122333
kim 700905-2211333
'''
# print(data)
pat = re.compile("(\d{6})[-]\d{7}")
print(pat.sub("\g<1>-*******",data))


# 07-2) 정규 표현식 시작
# 정규식의 기초 == 메타문자 meta characters 원래 문자의 뜻이 아니라 다른 특별한 용도로 사용하는 것
# . ^ $ + ? {} [] \ | ()

# 문자 클래스 []
""" 
 문자 클래스로 만들어진 정규식의 뜻 :  [] 사이의 문자들과 매치 
 정규표현식 [abc] 뜻 : a,b,c 중 한개의 문자와 매치
 문자열 'a','before','dude'중 'dude'는 [abc] 와 매치되지 않는다. (a or b or c 아무것도 없음)
 [from - to] 하이픈으로 범위나타냄 [a-c] == [abc], [0-5] == [012345]
 [a-zA-Z] : 알파벳모두     [0-9]:숫자    
 [^0-9] : 숫자제외 문자모두! 
 
 자주사용하는 문자클래스
 \d  숫자  ==[0-9]
 \D  숫자가 아닌것 == [^0-9]
 \s  whitespace문자(space or tab처럼 공백을 표현하는 문자)와 매치 ==[\t\n\r\f\v]
 \S  위의 반대 == [^\t\n\r\f\v]
 \w  문자+숫자(alphanumeric)와 매치 == [a-zA-Z0-9_]
 \W  위의 반대 == [^a-zA-Z0-9_]
"""

# Dot(.) 
# 줄바꿈 문자(\n)을 제외하고 모든 문자와 매치됨 
# re.DOTALL 옵션을 주면 \n문자와도 매치됨 
"""
a.b 의미 =  "a+모든문자+b" 
정규식 a.b 는 aab,a0b와는 매치되지만 abc 와는 매치 안됨 
정규식 a[.]b는 a.b와 매치, a0b와는 매치안됨
"""

# 반복(*)
"""
*바로 앞의 문자가 0번이상 반복하면 매치 (2억개까지)
ca*t 는 ct,cat,caaaat와 매치    
"""

# 반복(+)
'''
+ 바로 앞의 문자가 1번이상 반복하면 매치
ca+t 는 cat,caaat와 매치, ct는 매치안됨
'''

# 반복({m,n},?)  
'''
{m,n}을 사용하여 반복횟수 고정
{1,} == +
{0,} == *

ca{2}t 는 caat와 매치 
ca{2,5}t는 ca(2~5번 반복)t와 매치 

ab?c == ab{0,1} : b가 없어도 되고 1번 있어도됨 -> abc , ac와 매치
'''

# 파이썬에서 정규표현식을 지원하는 re모듈
# re:regular expression
import re
p = re.compile('ab*')   #-> re.compile(==정규표현식을 컴파일함)로 정규식(ab*)를 p로 돌려줌 

# 정규식을 사용한 문자열 검색
# match()  문자열 처음부터 정규식과 매치되는지 조사
# search()  문자열 전체를 검색하여 매치하는지 조사
# findall() 정규식과 매치되는 모든 문자열(substring)을 리스트로 반환
# finditer() 정규식과 매치되는 모든 문자열을 반복가능한 객체로 반환

# match/search는 매치된 객체를 돌려주거나 none반환

# match() - 맨 처음 나오는 3이 없으니까 none
import re
p1 = re.compile('[a-z]+')

m = p1.match('python')
print(m)  #>>>  match='python'>

m = p.match('3 python')
print(m)  #>>> None

# 정규식 프로그램 예제
p = re.compile(정규표현식)
m = p.match('조사할 문자열')
if m :
    print('Match found : ', m.group())
else :
    print('No match')

# search- 전체 중 찾는거니까 반환 
m = p1.search('python')
print(m)

m= p1.search('3 python')
print(m)

#findall
result = p1.findall("life is too short")
print(result) #>>> ['life', 'is', 'too', 'short']

# finditer
result = p1.finditer("life is too short")
print(result)
for r in result :
    print(r)


# match와 search메서드를 수행한 결과로 반환된 match객체에 대해 알아보기!
# group() : 매치된 문자열 반환
# start() : 매치된 문자열의 시작위치 반환
# end() : 끝 위치 반환
# span() : 매치된 문자열의 (시작, 끝)에 해당하는 튜플 반환 
    
# 예제
import re
p = re.compile('[a-z]+')
m = p.match('python')
m.group()
m.start()
m.end()
m.span()

import re
p = re.compile('[a-z]+')
m = p.search('3 python')
m.group()
m.start()
m.end()
m.span()

# 모듈 단위로 수행하려면 (2줄 -> 1줄)
m = re.match('[a-z]+', 'python')
print(m)

# 정규식 컴파일 옵션 : 옵션이름 / 약어 : 설명 
# DOTALL  / S  : dot문자(.)가 줄바꿈 문자를 포함하여 모든 문자와 매치 
# IGNORECASE  / I  : 대소문자에 관계 없이 매치 
# MULTILINE  / M   : 여러 줄과 매치 (^,$ 메타문자와 관계가 있는 옵션)
    #^a: 문자열의 처음은 a로 시작해야하고, $b:문자열의 마지막은 b로 끝나야 한다.  
# VERBOSE   / X  : verbose 모드를 사용한다(정규식을 보기 편하게 만들수 있고, 주석등을 사용가능)

# 옵션 사용시 re.S 처럼 사용

import re
p = re.compile('a.b', re.S)
m = p.match('a\nb')
print(m) 

import re
p = re.compile('[a-z]+', re.I)
m = p.search('3 pythON')
print(m)

import re
p = re.compile("^python\s\w+")  # 정규식 설명 : python으로 시작,그뒤에 whitespace, 그뒤에 단어
data = '''python one
life is too short
python two
you need python
python three
'''
print(p.findall(data))  #>>> 'python one'

import re
p = re.compile("^python\s\w+", re.M)   #>>> ^메타문자를 전체의 처음이 아니라 각 라인별 처음을 구하고자 하면 re.M
data = '''python one
life is too short
python two
you need python
python three
'''
print(p.findall(data)) #>>> ['python one','python two',' python three']

# re.X는 복잡한 정규식을 한줄한줄 나눠서 주석까지 달 수 있음 p.306

# 백슬래시\가 포함된 문자열을 찾기위해선 \\ 두번쓰기! 아니면 \s로 해석됨(==whitespace)
p = re.compile('\\section') 
# 하지만 2개의 \\를 전달하려면 4개 사용해야함
# raw string 규칙 -> 원래대로 써도 동일
p = re.compile(r'\\section')
data= 'hello this is your \\section'
print(p.search(data))

#%% 07-3) 강력한 정규표현식 

# 메타문자 

# | == or
p =re.compile('crow|servo')
m = p.match('crowhello')
print(m)

# ^ 문자열의 맨 처음 
print(re.search('^life','my life'))  #>>> none 

# $ 문자열의 맨 마지막
print(re.search('short$', 'life is short, you need'))  #>>> none

# \A : ^와 동일하지만    re.M 사용시 -> 줄에 상관없이 전체 문자열의 처음만 매치
# \Z : $와 동일하지만,   re.M 사용시 -> 전체 문자열의 끝만 매치 

#\b : 단어 구분자(word boundary) 보통 단어는 whitespace에 의해 구분됨
# (r'\b문자열\b) 로 사용해야함 / 단독 \b는 백스페이스 
print(re.search(r'\bclass\b', 'no class at all'))

# \B: 위와 반대 (문자 앞뒤가 빈칸이 아니여야함 )
print(re.search(r'\Bclass\B', 'no subclass at all'))
print(re.search(r'\Bclass\B', 'the declassified'))



# 그루핑 gruoping  - ABC문자열이 계속해서 반복되는지 조사하는 정규식 작성할때!
(ABC+)

p = re.compile('(ABC)+')
m = p.search('ABCABCABC OK?')
print(m)   #>>> , match='ABCABCABC'>
print(m.group(0))


p = re.compile(r"\w+\s+\d+[-]\d+[-]\d+")
m = p.search('park 010-1234-1234')

# \w+\s+\d+[-]\d+[-]\d+  -> 이름 +"" +전화번호 형태의 문자열을 찾는 정규식 
# 위에서 이름만 뽑아내고 싶다면 match객체의 group(인덱스숫자)메서드를 사용하여 뽑아낼수 있음 
# group(0) :매치된 전체 문자열    group(1) : 첫번째 그룹에 해당하는 문자열      group(n) :n번째 그룹에 해당하는 문자열  
# 그룹이 중첩되어있다면 안쪽으로 들어갈수록 인덱스 값이 커짐 
p = re.compile(r"(\w+)\s+\d+[-]\d+[-]\d+")
m = p.search('park 010-1234-1234')
print(m.group(1))


p = re.compile(r"(\w+)\s+(\d+[-]\d+[-]\d+)")
m = p.search('park 010-1234-1234')
print(m.group(2))


# 010만 뽑고 싶으면
p = re.compile(r"(\w+)\s+(\d+)[-]\d+[-]\d+")
m = p.search('park 010-1234-1234')
print(m.group(2))


# 그루핑된 문자열 재참조하기(backreference)
p = re.compile(r'(\b\w+)\s\1')  #(그룹)+""+그룹과 동일한 단어   와 매치됨을 의미,  \2를 쓰면 두번째 그룹 참조
p.search('Paris in the the spring').group()


# 그루핑된 문자열에 이름 붙이기 - 정규식 안에 그룹이 많으면 복잡해짐
# 그룹을 인덱스가 아닌 이름(named group)으로 참조한다면?   (?P<그룹이름>...) 으로 정규식을 만들면 됨 
(?P<name>\w+)\s+((\d+)[-]\d+[-]\d+)   # (\w+) 가 (?P<name>\w+)으로 달라짐 

p = re.compile(r'(?P<name>\w+)\s+((\d+)[-]\d+[-]\d+)')
m = p.search('park 010-1234-1234')
print(m.group('name'))

p = re.compile(r'(?P<word>\b\w+)\s+(?P=word)')
m = p.search('Paris in the the spring').group()
print(m)




# 전방탐색 lookahead assertions : 확장구문

p = re.compile(".+:")
m = p.search("http://google.com")
print(m.group())  #>>> http:
# .+: 과 일치하는 http: 반환

# 긍정형 전방탐색 (?=...)  : ... 에 해당되는 정규식과 매치필요,  조건이 통과되어도 문자열이 소비되지 않는다
# 부정형 전방탐색 (?!...)  : ...에 해당되는 정규식과 매치되지 않아야 하며, 조건이 통과되어도 문자열이 소비되지 않는다.


# 긍정형 전방탐색 (http: 에서 : 떼고 싶을때 )
# : 에 해당하는 문자열이 정규식 엔진에 의해 소비되지 않아(검색에는 포함되지만 검색 결과에는 제외됨) 
# 검색 결과에서는 :이 제거된 후 돌려주는 효과가 있다.
p = re.compile(".+(?=:)")
m = p.search("http://google.com")
print(m.group())   #>>> http

# .*[.].*$   : 파일이름 + . + 확장자를 나타내는 정규식 
# -> 여기에 .bat파일은 제외해야한다는 조건이 생기면 .*[.][^b].*$ 를 해야하지만 .bar파일도 제거됨 

# -> .*[.]([^b]..|.[^a].|..[^t])$  : | 메타문자를 사용하여, 확장자의 첫번째 문자가 b가 아니거나 두번째가 a가 아니거나 세번째가 t가 아닌 경우
# foo.bar는 살아남고, auto.bat은 제외됨 (확장자가 2글자면 포함못함 )

# -> .*[.]([^b].?.?|.[^a]?.?|..?[^t]?)$  : 확장자가 2글자여도 통과됨.......머야 이게 



# 부정형 전방탐색 : bat과 exe를 제외하라고 하면 더 복잡한 정규식이 생김 하지만 부정형을 사용하면 간단해짐
# .*[.](?!bat$).*$   :확장자가 .bat이 아니면 통과
# .*[.](?!bat$|exe$).*$   : .bat, .exe가 아니면 통과 



# 문자열 바꾸기 - sub 메서드를 사용하면 정규식과 매치되는 부분을 다른 문자로 바꿀 수 있음 
# .sub('바꿀 문자열','대상문자열')
p = re.compile('(blue|white|red)')
p.sub('colour', 'blue socks and red shoes')
# >>> 'colour socks and colour shoes'

# 한번만 바꾸고 싶다면 count =1 옵션주면 됨
p.sub('colour', 'blue socks and red shoes', count=1)
# >>> 'colour socks and red shoes'


# subn 메서드는 반환결과를 튜플로 돌려줌 
p = re.compile('(blue|white|red)')
p.subn( 'colour', 'blue socks and red shoes')
# >>> ('colour socks and colour shoes', 2)   2는 바꾸기 발생횟수 

# sub 메서드 사용시 참조 구문 사용하기
# sub의 바꿀 문자열 부분에 \g<그룹이름>을 사용하면 이름+전화번호  ->  전화번호+이름 으로 출력가능 
p = re.compile(r"(?P<name>\w+)\s+(?P<phone>(\d+)[-]\d+[-]\d+)")
print(p.sub("\g<phone> \g<name>", "park 010-1234-1234"))
# >>> 010-1234-1234 park

# 그룹이름대신 참조번호 사용해도 됨 
p = re.compile(r"(?P<name>\w+)\s+(?P<phone>(\d+)[-]\d+[-]\d+)")
print(p.sub("\g<2> \g<1>", "park 010-1234-1234"))

# sub 메서드의 매개변수로 함수 넣기
# match객체를 입력받아 16진수로 변환하여 돌려주는 함수
# p.sub(function, 'match객체')
def hexrepl(match) :
    value = int(match.group())
    return hex(value)

p = re.compile(r'\d+')
p.sub(hexrepl, 'Call 12345 for printing, 67890 for user code.')
#>>> 'Call 0x3039 for printing, 0x10932 for user code.'


# greedy vs non-greedy
s = '<html><head><title>Title</title>'
len(s)  #>>>32
print(re.match('<.*>', s).span())  #>>>(0,32)
print(re.match('<.*>', s).group())  #>>> <html><head><title>Title</title>

# 정규식 <.*>의 매치 결과로 <html> 문자열을 돌려주지 않고, 
# *메타문자는 탐욕스러워서, 매치할수 있는 최대한의 문자열인 <html><head><title>Title</title> 문자열을 모두 소비함
# <html>만 소비하도록 막으려면 non-greedy문자인 ?을 사용하기
print(re.match('<.*?>',s).group()) #>>> <html>

# non-greedy문자인 ?는 *?  +?   ??  {m,n}? 처럼 사용할 수 있음 -> 가능한 가장 최소한의 반복을 수행하도록 도와주는 역할 

import pandas as pd
df=pd.read_csv()





