# a,b 입력받고(1,2) ->더하기
a,b=map(int,input().split())

a,b,c=map(int,input().split())
a=int(input())
b=int(input())
print(a*((b%100)%10))
print(a*(b-((b%100)%10) -(b//100)*100)//10)
print(a*(b//100))
print(a*b)


a,b=map(int,input().split())
if a > b :
    print('>')
elif a < b:
    print('<')
else :
    print('==')

test=int(input())
if test >= 90 and test <= 100:
    print('A')
elif test >= 80 and test <90:
    print('B')
elif test >= 70 and test <80:
    print('C')
elif test >= 60 and test<70 :
    print('D')
else:
    print('F')

# 윤년 계산
i = int(input())
if i%4==0 and i%100 != 0 or i%400==0:
    print(1)
else:
    print(0)

# 알람시계
h,m=map(int, input().split())
if m < 45 and m >= 0 and h <=23 and h>0 :
    print(h-1, m-45+60)
elif m==45:
    print(h,0)
elif h==0 and m> 45:
    print(h,m-45)
elif h==0 and m< 45:
    print(23, m-45+60)
else:
    print(h,m-45)


a,b,c=map(int, input().split())

if a<=b and b<=c:
    print(b)
elif a<=c and c<=b:
    print(c)
elif b<=a and a<=c:
    print(a)
elif b<=c and c<=a:
    print(c)
elif c<=a and a<=b:
    print(a)
else:
    print(b)

a,b,c=map(int, input().split())
x=sorted([a,b,c])
print(x[1])

# 구구단
n=int(input())
r=[]
for i in range(0,9):
    i+=1
    r=n*i
    print(n,'*',i, '=',r)


t = int(input())
for i in range(0,t):
    t +=1
    ta,tb=map(int, input().split())
    print(ta+tb)

n = int(input())
j=0
for i in range(0,n):
    i +=1
    j +=i
print(j)

# input 대신에 sys.stdin.readline 사용하기 (문자열을 저장할땐 .rstrip()추가하기)
import sys

for i in sys.stdin.readline():
    print(i)

# 입력값 리스트로 저장
a = list(map(int, sys.stdin.readline().split()))
a

map(int, input().split())
map(int, sys.stdin.readline().split())

'''
여러줄 입력받을 때 라는데 안돼 이상함 
for i in sts.stdin:
    print(i)
for line in sys.stdin:
    li.append(tuple(map(int, line.strip().split())))
'''

# t값도 readline으로 처리!
import sys
for i in range(int(sys.stdin.readline())):
    a,b=map(int, sys.stdin.readline().split())
    print(a+b)


import sys
for i in range(1,int(sys.stdin.readline())+1):
    print(i)


for i in range(int(sys.stdin.readline()),0,-1):
    print(i)

# 이렇게변수로 지정해주고 사용하기
import sys
read=sys.stdin.readline
t=int(read())

for i in range(t):
    ab=[int(j) for j in read().split()]
    print('Case #{}: {} + {} = {}'.format(i+1, ab[0],ab[1],ab[0]+ab[1]))


for i in range(0,6):
    print('*'*i)


import sys
read=sys.stdin.readline
i = int(read())
for j in range(1,i+1):
    print('*'*j)


import sys
read=sys.stdin.readline
j = int(read())
for i in range(1,j+1):
    print(' '*(j-i)+'*'*i)


import sys
read=sys.stdin.readline
n,x = map(int, read().split())
a=list(map(int, read().split()))
s= map(str, list(filter(lambda i :i < x, a)))
print(' '.join(s))
'''
for i in range(n):  # 이상하게 나옴 ㅠ for대신 lambda 습관들이기 
    if a[i] < x:
        print(a[i])
'''
# < map, lambda, filter
# map : map(f, iterable) : 함수f와 반복가능한 자료형 입력받음
# lambda 매개변수 : 식
a=lambda x : x+10

# 람다 표현식에서 변수 호출하기 (lambda 매개변수들 : 식)(인수)
(lambda x:x+10)(1)
a(1)

# filter
li=[-3,-2,1,2,0,6,8]
list(filter(lambda x :x>0, li))
list(filter(lambda x: x%3==0, li))

# reduce : 요소가 한개가 남을때까지 function실행
from functools import reduce
reduce(lambda x,y: x+y,[1,2,3,4,5])

# 리스트 내포 문법 ) a = [표현식 for 항목 in 반복가능객체 if 조건]
a = [1,2,3,4]
r1 = [i*3 for i in a if i%2==0]
print(r1)

result = [x*y for x in range(2,10)
    for y in range(1,10)]
print(result)



# 실습
li=[1,2,3]
r=list(map(lambda i : i**2, li))
r

li=[-3,-2,1,2,0,6,8]
r=list(map(lambda i:'pos' if i>0 else('neg' if i<0 else 0), li))
r

import sys
read=sys.stdin.readline
a,b=map(int, read().split())  # 얘를 위에서 선언하면 계속 실행함   ctrl + C로 중지
while True:
    if a==0 and b==0 :
        break
    print(a+b)


while True:
    try:
        a, b= map(int, read().split())
        print(a+b)
    except:
        break

x=1
a=x//10
a
b=x%10
b

'''
26 : 2+6=8   a+b=c
6+8=14      b+c//10 = d
8+4=12      c//10 +d//10 = e
4+2=6       d//10 + e//10 = f

if e//10=a, f//10=b break 
'''
# 함수 만들어서 풀려면 기존 숫자에 대해 1자리 인지 2자리인지 부터 먼저 정정하고 새로운 숫자만든거 검사
# 함수로 만든 변수를 다시 변수로 넣어도 됨

number=int(sys.stdin.readline())
cycle=1
def some_num(number):
    num=int(number)
    if num <10:
        return(num*10)+num
    else:
        a=(num//10)+(num%10)
        b=((num%10)*10)+(a%10)
        return b

def check(b):
    num=int(number)
    global cycle
    if num!= b:
        cycle +=1
        final_num=some_num(b)
        check(final_num)

check(some_num(number))
print(cycle)



# 이렇게 해도 되지만 위처럼 함수 만들어도됨
import sys
read=sys.stdin.readline
n = int(read())
check=n
new_n = 0
count = 0

while True:
    tmp=n//10 + n%10
    new_n=(n%10)*10 + tmp%10
    count +=1
    if new_n==check:
        break
print(count)


from sys import stdin
read=stdin.readline
x=int(read())
y= list(map(int, read().split()))
z=sorted(y)
print(min(z),max(z))

from sys import stdin
read=stdin.readline
x=list(map(int,read().split()))
print(max(x))
print(x.index(max(x)))


# 1줄에 1개씩 입력받아야 할때
max_n=0
max_index=0
for i in range(1,10):  # == range(9)
    x=int(read())
    if (x > max_n):
        max_n = x
        max_index=i+1

print(max_n, max_index)


import sys
read=sys.stdin.readline
a=list(map(int, read().split()))


a,b=list(map(int, input().split()))

a= input()
b= list(map(int, input().split()))
print(b)

for i in range(0, len(b)-1):
    for j in range(i+1, len(b)):
        if b[i] > b[j]:
            tmp=b[i]
            b[i]=b[j]
            b[j]=tmp
print(b)


# greedy 11399
import sys
def atm(line):
    total=0
    wait=0
    line.sort()      # 이 있어야 최소시간을 구할수 있음 순서는 자유롭게 바꿀수 있으니까
    for time in line:
        wait += time
        total += wait
        print(line)
    return total
n=int(sys.stdin.readline())
line=list(map(int, sys.stdin.readline().strip().split()))
print(atm(line))


# 11047 동전
import sys
def a(bank, k):
    count=0
    bank=list(reversed(bank))  # 큰단위 동전을 먼저 많이 사용
    total= k

    for coin in bank :
        if total % coin >= 0 and total >= coin :
            cnt= total//coin
            count += cnt  # 어짜피 개수의 최소값을 구하는 것
            total -= cnt * coin
        if total == 0:
            break
    return count

n,k=list(map(int, sys.stdin.readline().strip().split()))
bank=[]
for i in range(n):  # 위에서 n을 먼저 지정해주고 그 값만큼 coin입력받기
    coin=int(sys.stdin.readline().strip())
    if coin > k :
        break
    bank.append(coin)
a(bank,k)


def solution(v):
    answer=[]


    return answer


a, b = map(int, input().strip().split(' '))
print(a + b)

# 데모 테스트 1
def S(a,b):
    a,b=map(int,input().strip().split(' '))
    for i in range(0,b):
        ans= print('*'*a)
    return ans

S(a,b)


# xor 게이트 문제 !
def sol(v):
    v=list(map(int, input().split()))
    ans[0]=v[0][0]^v[1][0]^v[2][0]
    ans[1]=v[0][1]^v[1][1]^v[2][1]
    return ans

sol(v)




e_list=[]
def sol(emails):
    e=list(map(str, input().split(',')))


for i in range(len(e)-1):
    if e[i][-4:] in ['com','net','org']:
        print(e)


num, start, pages = list(map(int, input().split()))
printlist=[]
time=[]
wait=[]

for i in range(num) :
    printlist.append(int(input()))


for i in range(1, time )



e_list = []


def sol(emails):
    e = list(map(str, input().split(' ')))

abc.def@x.com abc abc@defx abc@defx.xyz
d@co@m.com a@abc.com b@def.com c@ghi.net

 for i in range(len(e) - 1):
    if e[i][-3:] in ['com', 'net', 'org']:
           print(e)

e[0][-3:]
e[1].count('@')
count=0

sol(e)


for i in range(len(e)-1):
    if len(e[i]) < 7:
        e.pop(i)
        print(e)

def sol(e):
    answer=[]
    alpht = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(len(e)-1):
        if len(e[i]) < 7 :
            e=e.pop(i)
        else :
            for i in range(len(e)-1):
                top=e[i][-4:]
                name = e[i][:-4].islower()
                dot=e[i].count('@')
                ch1=e[i][0]  # 영어소문자로 시작 and 1글자 충족
                ch2=e[i][-5] # 도메인 영어소문자로 시작, 1글자 충족

                if top in ['.com','.net', '.org'] and  dot == 1 and name == 1 and ch1 in alpht and ch2 in alpht:
                    answer.append(e[i])
                else:
                    pass
    return len(answer)

sol(e)

e=['a@b.com','avd']
len(e)
sol(e)

 a='@'


d@co@m.com @abc.com b@def.com c@Ghi.net








a= e[3][:-3].islower()
a == True

for i in range(3):
    name=e[i][:-3].islower()
    print(name)

e[0][:-3].islower()




e[0].replace('""',' ')
e.replace('""','')

e[0][-4:]
e[1][-4:]

a='ADS'
