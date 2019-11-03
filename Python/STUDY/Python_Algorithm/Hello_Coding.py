# 출처: Hello Coding 그림으로 개념을 이해하는 알고리즘
# http://www.hanbit.co.kr/media/books/book_view.html?p_code=B5896248244

# < Ch1 알고리즘 소개 >
# 이진탐색 (단순 탐색은 100개면 다 확인)
# log (100,10) = 10을 2번 곱해야 100이 됨 로그 <-> 거듭제곱
# 리스트에 1024개의 숫자가 있다면 10번만 확인하면 나옴(반씩) 1024, 512, 256,128,64,32,16,8,4,2,1

def binary_search(list, item):
    low=0
    high=len(list)-1

    while low <= high :
        mid=(low+high)//2
        guess=list[mid]
        if guess == item:
            return mid
        if guess > item:
            high=mid -1
        else:
            low=mid+1
    return None
my_list=[1,3,5,7,9]
print(binary_search(my_list,3))
print(binary_search(my_list, 100)) #> None

# 빅오 표기법 Big O notation
'''
- 알고리즘이 얼마나 빠른지 표시 : 속도가 아니라 연산횟수가 어떻게 증가하는지로 측정 
- 크기가 n인 리스트 확인하기 위해 log n 번의 연산 필요 
    -> O(log n) 으로 표기 
- 이진탐색은 입력데이터가 기하급수적으로 늘어도 그렇게 시간이 오래 걸리지 않음 
O(log n) 로그시간
O(n) 선형시간 linear time
O(n*log n) 예: 퀵정렬, 빠름
O(n**2) 예: 선택 정렬, 느림
O(n!)  예: 외판원 문제, 제일 느림 
'''
# 외판원 문제 traveling salesperson problem 짧은 거리로 5개의도시 방문
# n개의 도시라면 n! 번의 연산 필요 O(n!) 시간

# < CH2 선택정렬 Selection Sort >
'''
- 연결 리스트 linked list : 컴퓨터의 메모리에 저장을 할때 아무곳에나 원소를 넣고 주소를 함께 저장해서 연결
- 배열은 한번에 쭉 있는거고, 연결 리스트는 이곳저곳이지만 연결이 되어있는 것(특정원소를 찾으려면 최악)
- 원소의 위치 = index 예: 20은 위치 1에 있다 -> 20은 인덱스 1에 있다. 
      배열 리스트 
읽기  O(1) O(n)
삽입  O(n) O(1)
삭제  O(n) O(1)

- 배열은 선형시간, 리스트는 고정시간  
- 리스트의 가운데에 삽입하는건 배열에 넣는 것보다 쉬움, 빠름 (배열은 하나씩 뒤로 다 미뤄야함)   / 삭제도 동일 
- 자료 접근법 : 임의접근random access(배열 읽기) , 순차 접근 Sequential access(연결리스트)

# 선택정렬 - 깔끔 하지만 빠르지 않음 
가장 많이 들은 노래 순대로 sort할때 O(n^2)의 시간이 걸림
(n개의 항목 점검 * n-1 * n-2 *..2*1 : 평균적으로 1/2*n개의 항목 점검 실행시간은 O(n*1/2*n) 이지만 상수항은 빼고 n^2 
전화번호부의 이름순 정렬, 여행날짜순 정렬, 이메일(새것-> 오래된 순) 정렬 등 
'''
# 선택정렬 작은수 -> 큰 수로 정렬(작은 수를 먼저 찾고 sort)
def findSmallest(arr):
    smallest=arr[0]
    smallest_index=0
    for i in range(1,len(arr)):
        if arr[i] < smallest:
            smallest=arr[i]
            smallest_index=i
    return smallest_index

def selectionSort(arr):
    newArr=[]
    for i in range(len(arr)):
        smallest = findSmallest(arr)
        newArr.append(arr.pop(smallest))
    return newArr

a=[40,24,2,9,22,52,61,7,34,90]
a
b=selectionSort(a)
a
b # pop으로 작은 것부터 꺼냈으니 a엔 없음


# < Ch3 재귀 >
# 1) while만 쓸 경우 - 의사 코드
def look_for_key(main_box):
    pile = main_box.make_a_pile_to_look_through()
    while pile is not empty:
        box=pile.grab_a_box()
        for item in box:
            if item.is_a_box():
                pile.append(item)
            elif item.is_a_key():
                print('열쇠를 찾았어용!')

# 2) 재귀 recursion : 함수가 자기 자신을 호출하는 것 - 의사코드
def look_for_key1(box):
    for item in box:
        if item.is_a_box():
            look_for_key1(item) #  본인 자기자신을 호출해서 반복
        elif item.is_a_key():
            print("I fonud a key!")


def count(i) :
    print(i)
    count(i-1)
count(5) # 끝업이 진행 여기선 오류!

def countdown(i):
    print(i)
    if i<=1 :  # base case 기본단계
        return
    else :      # recursive case 재귀 단계
        countdown(i-1)
countdown(5)

# 호출 스택 stack 후입선출   푸시, 팝 (푸시:삽입, 팝 : 꺼내기(기존에서 삭제))
# 여러개의 함수를 호출하면서 함수에 사용되는 변수를저정하는 스택 : 호출 스택 call stack
def greet(name):
    print('hello, '+ name + '!')
    greet2(name)
    print('getting ready to say goodbye!')
    bye()

def greet2(name):
    print('how are you, '+ name + '?')
def bye():
    print('ok bye!')

greet('sera')

# 재귀함수에서 호출스택 사용 팩토리얼 함수 factorial function
def fact(x):
    if x==1:
        return 1
    else :
        return x*fact(x-1)

fact(10)
# 스택은 메모리 사용이 높지만 할일이 줄음!

# < CH 4 퀵 정렬 quick Sort

# 유명한 재귀적 기술인 분할정복 전략 divide-and-conquer
# 기본문제를 해결(가능한 간단), 문제가 기본단계가 될때까지 나누거나 작게 만듦 (밭 나누기)
def sum(arr):
    total = 0
    for x in arr:
        total +=x
    return total

sum([1,3,2,4,56])
sum([])

# 퀵 정렬 Quick Sort  - 기준원소=pivot - 기준원소보다 작/큰 배열로 분할 partitioning - 분할한 하위배열을 재귀적으로 퀵정렬 호출
# 원소가 1개(첫 return 까지가 1개, 원소가 2개이상이면 None으로 뜸) , 원소가 여러개
def quicksort(array):
    if len(array) <2 :
        return array
    else :
        pivot=array[0]  # 첫번째 원소를 피봇으로
        less =[i for i in array[1:] if i <=pivot]
        greater = [i for i in array[1:] if i > pivot]
        return quicksort(less)+[pivot]+quicksort(greater)

quicksort([34,25,122,23,53,41,2])

# 병합정렬, 퀵 정렬 비교  : 상수때문에 생기는 차이 O(n log n)으로 동일하지만 퀵정렬이 더 작은 상수 - > 더 빠름
def print_item(list):
    for item in list:
        print(item)

print_item([1,4,25])

from time import sleep
def print_item2(list):
    for item in list:
        sleep(1)   # c: constant 상수 알고리즘이 소비하는 어떤 특정한 시간
        print(item)
print_item2([1,3,2,4,5])

# 최악의 경우: 피봇원소가 맨 처음일때 O(n^2)의 시간
# 최선의 경우 : 피봇원소가 가운데 O(n log n)의 시간 : 보통 퀵정렬이 무작위로 피봇선택해도 평균 이정도 속도!

# < Ch6 hash table : 모든 원소를 알고 있기 때문에 O(1)의 시간으로 읽을 수 있음   딕셔너리
# hash function : 문자열을 받아서 숫자 반환  (키:문자열 - 인덱스 저장 -> 밸류:숫자)

book=dict()
book['apple']=0.6
book['milk']=1.5
book['avocado']=1.6
print(book)
print(book['avocado'])

phonebook=dict() # == {}
phone={}

phone['jenny']= 543259
phone['emergency']=119
phone['police']=112
phone

# 중복 방지 - 누가 투표했는지 확인할때
voted={}
print(voted.get('tom')) #> None

# 전체 코드
voted={'sera':True, 'jiny'}
def check_voter(name):
    if voted.get(name):
        print('잘가! 이미했음!')
    else:
        voted[name]=True
        print('투표해!')
check_voter('sera')
check_voter('jiny')

# 해시 테이블로 캐싱 : 홈페이지에 접속에서 url 호출-> 해시테이블에 있으면 그 내용 전송/ 없으면 서버가 무엇인가 작업
# 작업 시간이 단축, 정보를 다시 계산하지 않고 저장했다가 알려주는 것이 캐싱
# 의사코드
cache={}
def get_page(url):
    if cache.get(url):
        return cache[url]

    else:
        data=get_data_from_server(url)
        cache[url]=data
        return data

# 충돌 apple, banana 입력되어있는데 avocado를 넣으려면 충돌!-> 같은 공간에 여러개의 키를 연결리스트로 만들기
# 기존에 해시테이블을 고르게 할당하는게 좋음 (배열에 값을 고루 분포)

# 성능
#O(1) : 상수시간 constant time - 해시테이블 크기에 상관없이 항상 똑같은 시간이 걸림
# 해시 테이블의 탐색, 삽입, 삭제 모두 O(1)의 시간이 걸림 (최악의 경우에는 모두 O(n)시간)
# 사용률 load factor : 해시 테이블에 있는 항목의 수/ 해시테이블에 있는 공간의 수
# -> 사용률이 커지면 테이블에 공간을 추가 : resizing (2배의 크기로 보통 리사이징) 보통 사용률이 0.7정도되면


# <Ch 6 너비우선탐색 BFS : breadth-first search
# 정점 A에서 정점 B로 거리 구해줌 , 최단거리도 구해줌
# 큐queue(대기열) :   삽입enqueue, 제거 dequeue : 큐는 선입선출 (스택은 후입선출) 123에 4 들어오면 1이 먼저 나감

# 그래프 구현
graph={}
graph['you']=['alice','bob','claire'] # you와 연결된 3명
graph['bob']=['an','peggy']
graph['alice']=['peggy']
graph['claire']=['tom','jonny']
graph['an']=[]
graph['peggy']=[]
graph['tom']=[]
graph['jonny']=[]

# 방향 그래프 directed graph: 화살표 있음(일방적인 관계) / 무방향 그래프 undirected graph : 선만 있지만 상호관계(로스-레이첼) 사실 싸이클

# 알고리즘: 확인할사람의 명단을 넣을 큐 준비(전체) -> 큐에서 한사람 꺼내고 -> 망고파는지 확인-> 예면 끝 / 아니면 그 사람의 이웃을 모두 큐에 추가 -> 반복 / 큐가 비어있다면 망고 판매상이 없
# 큐생성 - 양방향 큐 함수 : deque
from collections  import deque
# search_queue=deque()
# search_queue +=graph['you'] # 모든 이웃을 탐색큐에 추가

# 전체 코드
def person_is_seller(name):
    return name[-1] == 'm'   # 이름이 m으로 끝나면 망고판매상이라고 가정
def search(name):
    search_queue= deque()
    search_queue += graph[name]
    searched=[]
    while search_queue:   # 큐가 안비어 있으면 계속 실행
        person=search_queue.popleft()   # 큐의 첫번째 사람 꺼냄, 추가된 순서대로 사람을 확인해야함 즉 탐색목록이 큐가 되어야
        if not person in searched:
            if person_is_seller(person):
                print(person +'\t is a mango seller!')
                return True
            else:
                search_queue += graph[person]
                searched.append(person)   # 확인하고 나서 따로 안빼두면 무한반복
    return False

# 실행
search('you')
# 너비우선탐색: O(노드의 수+ 에지의 수) = O(V+E)

# < CH7 다익스트라 알고리즘 Dijkstra's algorithm 
'''
최단거리 - 너비우선탐색
최소시간 - 다익스트라 알고리즘

1) 가격이 가장싼 노드 찾기(걸리는 시간이 제일 적은)
2) 1노드의 이웃노드들 가격 조사 
3) 모든 노드에 대해 반복
4) 최종경로 계산 

-에지에 써있는 숫자 : 가중치 weight , 가중그래프! (없으면 균일그래프) 

* cycle: 어떤 노드에서 출발해서 한바퀴를 돌아 어떤노드로 돌아와서 끝남 
    -> 무방향 그래프는 두 정점이 서로를 향하고 있는 것! 사이클(로스-레이첼)
'''
# 다익스트라 알고리즘은 방향성 비순환 그래프(Directed Acyclic Graph)나 가중치가 양수인 사이클 그래프만 가능 

