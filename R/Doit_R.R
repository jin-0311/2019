##### 기초 ###### 
#실행 : ctrl +enter
#여러줄 선택 및 실행 : shift + 키보드 방향키,  ctrl +enter
#tools - global options - soft wrap : 자동 줄바꿈 
#tools - global options -code - saving 에서 text encoding 을 UTF-8로 해야 한글 안깨짐 
#새로운 스크립트 생성 : ctrl+shift+n 
#다섯개 제목 #다섯개 -> 블록형성 


##### ch 1 / ch 2 분석환경 설정 ###### 
1+2
2+4
print("Hello World!")

var1 <- seq(1,80, by=2)  #1부터 80까지 2씩 증가 
var1   #console 창에 [28] : 인덱스 


##### ch 3 데이터 분석을 위한 변수/함수/패키지 ##### 
#변수 만들기 
#문자로 시작필수, 소문자 추천, -,_가능 

a <- 1   #a에 1 할당   / 띄어쓰기 권장  /  variable <- value 형태 / <-는 할당 연산자
a        #a 출력 

b <- 2 
b

c <- 3
c

d <- 3.5
d

a+b

a+b+c

4/b

5*b

var1 <- c(1, 2, 5, 7, 8)    # c(a,b,c ...) combine 변수에 여러 개의 값을 넣음  
var1

var2 <- c(1:5)              # 1~5까지 연속 값으로 var2생성   
var2

var3 <- seq(1, 5)           # 1~5까지 연속 값으로 var3생성 
var3

var4 <- seq(1, 10, by=2)    # 1~10까지 2 간격 연속 값으로 var4 생성 
var4

var5 <- seq(1, 10, by=3)
var5

var6 <- seq(1, 10, 4)       # by= 대신 , 가능! seq(1,10, by=4) = seq(1,10,4)
var6

var1 + 2 

var1 + var2

x <- c(4, 5, 6, 7)          # c(1,2,3,4) = c(1:4) = seq(1,4) = seq(1:4)
y <- seq(1, 4)     
x
y

xy <- x+y
xy


#문자로 된 변수 만들기  - "" 안에 넣기, 연산 불가 /  문자열(string) 처리 함수 따로 존재   

str1 <- "a"
str1

str2 <- "text"
str2

str3 <- "Hello World!"
str3

str4 <- c("a", "b", "c")
str4

str5 <- c("Hello", "world", "is", "good!")
str5


# 숫자를 다루는 함수 

x <- c(1, 2, 3)
x

mean(x)   # 평균 
max(x)  # 최대값
min(x)  # 최소값 

#문자를 다루는 함수
#paste(variable, collapse= " " )  collapse=" " -> 파라미터, 매개변수(함수의 옵션 설정) 

str5
paste(str5, collapse =",")     # 단어를 쉼표로 구분되도록 
paste(str5, collapse = " ")    # 단어를 빈칸으로 구분되도록 

x_mean <- mean(x)
x_mean 

str5_paste <- paste(str5, collapse=" ")
str5_paste


#패키지(함수 꾸러미) 설치하기 

install.packages("ggplot2")   # 패키지 설치 (1번만 설치)  " "필수! 
library(ggplot2)              # 패키지 불러오기 (R열때마다 로드)

x <- c("a", "a", "b", "c")    # 여러 문자로 구성된 변수 생성 
x
qplot(x)   # 빈도 막대 그래프 출력 


# ggplot2로 mpg데이터 그래프 그리기 
# mpg(mile per gallon)
# 변수 설명 hwy(고속도로 연비), cty(도시 연비), drv(구동 방식 3가지)


View(mpg)   # 데이터 확인할 때 사용 / v는 대문자! 

qplot(data = mpg, x = hwy)     #x축:hwy, y축:null
qplot(data = mpg, x = cty)     
qplot(data = mpg, x = drv, y = hwy)     #x축:drv, y축:hwy  / 점 그래프
qplot(data = mpg, x = drv, y = hwy, geom = "line")     # 선 그래프
qplot(data = mpg, x = drv, y = hwy, geom = "boxplot")  # 상자그림 
qplot(data = mpg, x = drv, y = hwy, geom = "boxplot", colour = drv) 
      #상자그림, drv별 색표현      

?qplot  #  = help(qplot) 함수 매뉴얼 출력 


#ch3 연습문제(p.77)
x <- c(80, 60, 70, 50, 90)
x
mean(x)  #평균 70점
mean_x <- mean(x)
mean_x








##### ch 4 데이터 프레임 (data frame) #####

# 열: column(컬럼) = variable(변수) 세로
# 행: row = case 가로 

# 데이터 프레임 만들기

# 1)변수 만들기
eng <- c(90, 80, 60, 70)
eng
math <- c(50, 60, 100, 20)
math

# 2)데이터 프레임만들기 data.frame()  데이터프레임 생성할때 df_라고 붙이면 구별 쉬움
# data.frame(var1, var2, ...) -> var1,2 미리 생성해야함 
df_midterm <- data.frame(eng, math)
df_midterm 

# 3) 컬럼 추가 
class <- c(1,1,2,2)
df_midterm <- data.frame(eng, math, class)
df_midterm

# 4)분석하기 - 데이터 프레임 안의 변수를 지정할때 $ 사용!
mean(df_midterm$eng)
mean(df_midterm$math)


# 1)~3) 한번에 하기 
# data.frame(var1=c(a,b,c...), var2=c(a1,b1,c1,...)) 
df_midterm <- data.frame(eng = c(90,80,60,70)
                         ,math = c(50,60,100,20)
                         ,class = c(1,1,2,2)) 
df_midterm

# 두개 없으면 1에 맞춰서, (1,2)였으면 1212
# 데이터가 하나 없으면 그 열 다 안나옴 or 12반복 
# 관측갯수(values)가 같아야 함 

sort(df_midterm$eng, decreasing = T)  # decreasing = T :최대값부터 점점 작아지게 


# ch4 연습문제

df_prod <- data.frame(prod = c("사과", "딸기", "수박")
                      ,price = c(1800,1500,3000)
                      ,sales = c(24,38,13))
df_prod

df_prod <- data.frame(제품 = c("사과", "딸기", "수박")   # 변수명 한글 가능하지만 비추
                      ,가격 = c(1800,1500,3000)
                      ,판매량 = c(24,38,13))
df_prod

df_prod <- data.frame("제품" = c("사과", "딸기", "수박")   # 변수명 한글 "/'로 묶기 
                       ,'가격' = c(1800,1500,3000)
                       ,'판매량' = c(24,38,13))
df_prod

mean(df_prod$'가격')
mean(df_prod$'판매량')

#변수명 rename 할 때 
#install.packages("MASS")
#library(MASS)
#names(df_prod)
#df_prod_subset <- df_prod[,c(1:3)]
#names(df_prod_subset)
#names(df_prod_subset) <- c("제품", "가격", "판매량")
#df_prod <-df_prod_subset
#df_prod


# 외부 데이터 이용하기 (엑셀파일 불러오기)

install.packages("readxl")  #엑셀파일 불러오는 패키지 사용할때 : read_excel()
library(readxl)

getwd()               # getworkingdirectory : 현재 작업폴더 확인 
setwd("c:/Work_r")    # setwd() 작업폴더 변경
  
df_exam <- read_excel("data/excel_exam.xlsx")   # 프로젝트 폴더에 있으면 바로 불러옴 
                      # df_exam <- read_excel("c:/ .xlsx")  다른 폴더에 있을 때 
df_exam

mean(df_exam$english)
mean(df_exam$science)


# 엑셀파일 1행에 변수명이 아닐 때 : read_excel("파일명", col_names = T / F) 대문자 

# col_names() : 열(column) 이름을 가져올 것인가? - T:있을 때 가져옴 / F:없을 때
# 1행에 변수명 없을 때 'X__숫자'로 자동 지정해줌 
# file-import data에서 불러올 수도 있음 

df_exam_novar <- read_excel("data/excel_exam_novar.xlsx", col_names=F)   
df_exam_novar


# 엑셀파일의 시트가 여러개일때
df_exam_sheet <- read_excel("data/excel_exam_sheet.xlsx", sheet=3 )
df_exam_sheet


# csv파일 불러오기 csv(comma-separated values), 숫자로만 구성됨, 용량 작음 
# 1행에 변수명 없으면 header=F
df_csv_exam <- read.csv("data/csv_exam.csv", header = T) 
df_csv_exam

# 문자가 들어 있는 csv 불러올 때 
# stringsAsFactors = F : 문자를 factor 타입으로 변환하지 않도록 설정하는 파라미터
                       # 그래야 문자(chr)로 데이터에 넣을 수 있음 
df_csv_exam <- read.csv("data/csv_exam.csv", stringsAsFactors = F)
df_csv_exam

# 데이터 프레임을 csv파일로 저장하기
df_midterm <- data.frame(eng = c(90,80,60,70)
                         ,math = c(50,60,100,20)
                         ,class = c(1,1,2,2)) 
df_midterm

write.csv(df_midterm, file = "df_midterm.csv") # 프로젝트 파일(c:/Work_r)에 저장 


# Rdata파일 활용하기
# Rdata(.rda or .rdata) :r전용 데이터 파일, r에서 작업하면 이거 이용/ 안쓰면 csv로! 
save(df_midterm, file = "df_midterm.rda") 

# Rdata 파일 불러오기
rm(df_midterm)          # 데이터 삭제 

load("df_midterm.rda")
df_midterm


df_exam <- read_excel("data/excel_exam.xlsx") # 엑셀파일을 df_exam에 할당하기
df_csv_exam <- read.csv("data/csv_exam.csv")  # csv파일을 df_csv_exam에 할당하기
load("df_midterm.rda")      # rda 불러오기 (새 변수에 할당하는 것 아님)




##### ch 5 데이터 분석 기초 #####



# 데이터 파악하기
head(df명, 숫자) # 데이터 앞부분 출력, 숫자행까지 출력 (default:6)
tail(df명, 숫자) # 데이터 뒷부분 출력, 숫자행까지 출력 (default:6)
View() # 뷰어창에서 데이터 확인 V대문자 
dim(df명)  # 데이터 차원 출력  >>> 행, 열 출력 
str(df명)  # 데이터 속성 출력  >>> obs/var갯수, 변수의 이름/속성
summary(df명) # 요약 통계량 출력 : min,1st Qu, median, mean, 3rd Qu, max 


exam <- read.csv("data/csv_exam.csv")
head(exam, 10)
tail(exam, 10)
View(exam) 
dim(exam)  # 20 obseravaion 행 / 5 variables 열 
str(exam)  # 1행(속성: obs, variables), 2행~ (변수 이름, 속성)  / int(integer 정수)
summary(exam)  # 변수별 요약 통계량 


#mgp 데이터 파악하기
# ::더블콜른 사용시 특정 패키지에 들어있는 함수/데이터 지정가능! 

library(ggplot2)
mpg <- as.data.frame(ggplot2::mpg)  # :: == ggplot2에 들어있는 mpg를 불러오겠다 
mpg
View(mpg)

head(mpg, 12)
tail(mpg, 12)
dim(mpg)  # >>> 234행, 11열(columns)
str(mpg)  
# int=integer 정수 (소수점 없음)
# num=numeric 실수 (소수점 있음)
# chr=character 문자 

?mpg  #패키지에 들어있는 데이터는 help함수를 통해 설명 볼 수 있음 
summary(mpg)   # 요약통계량_ 문자형 : length(값의 개수), class / mode (변수의 속성)  
?summary
 

hw_ct <- data.frame(mpg$manufacturer, mpg$hwy, mpg$cty)  #변수 3개만 골라오기! 
hw_ct 
 

# 변수명 바꾸기 

install.packages("dplyr")
library(dplyr)

df_raw <- data.frame(var1 = c(1,2,1)
                     , var2 =c(2,3,2))
df_raw

df_new <- df_raw
df_new

df_new <- rename(df_new, v2 = var2)  # df <- rename(df, 새로운 변수명 = 기존 변수명)
df_new

# 연습문제 

mpg <- as.data.frame(ggplot2::mpg)
mpg

new_mpg <- mpg

new_mpg <- rename(new_mpg, city = cty, highway = hwy)
new_mpg
head(new_mpg)
tail(new_mpg, 10)

# 파생변수 만들기 : Derived Variables, 기존의 변수를 변형해 만든 변수 

# 변수를 조합해 파생변수 만들기 
df <- data.frame(var1 = c(4,3,8)
                 , var2 =c(2,6,1))
df$var_sum <- df$var1 + df$var2   # df에 var_sum 라는 새 변수 만들기(var_sum = var1+var2)
df

df$var_mean <- (df$var1+df$var2)/2  # df에 var_mean 새 변수 만들기 
df

# mpg 통합 연비 변수 만들기

head(mpg)
mpg$total <- (mpg$cty + mpg$hwy)/2  # mpg안에 total이라는 새로운 변수 만들기 
head(mpg)

mean(mpg$total)  # 통합연비의 평균값 

# 조건문을 활용한 파생변수 만들기 
# 조건문 함수(conditional function) :조건에 따라 서로 다른 값을 반환하는 함수  

# mpg의 연비기준을 충족해 고연비 합격 판정을 받은 자동차 찾기 

# 1) 기준값 정하기 : total 20이상-합격
summary(mpg$total)   #>>> median:20, mean:20.15
hist(mpg$total)      #히스토그램 
  # 20~25에 해당하는 모델이 가장 많고, 대부분 25이하, 25넘기는건 적다. 
  # total 변수가 20을 넘으면 합격   

# 2) 합격 판정 변수 만들기 
variable <- ifelse(조건, "조건이 맞을때 부여", "조건에 맞지 않을때 부여") 
mpg$test <- ifelse(mpg$total >=20, "Pass", "False")
head(mpg, 20)

# 3) 빈도표로 합격 판정 자동차 수 살펴보기 
# 빈도표: 변수의 각 값들이 몇 개씩 존재하는지 데이터의 개수를 나타냄 table()
table(mpg$test)

# 4) 막대 그래프로 빈도 표현하기
library(ggplot2)
qplot(mpg$test)    # 빈도표를 막대 그래프로!  


# 중첩 조건문 활용하기 : 3가지 이상의 범주로 값을 부여할 때! + 파생변수 
# 중첩 조건문- ifelse()안에 ifelse() 다시 넣기 
# total값이 30이상:A, 20~29:B,  20미만:C
mpg$grade <- ifelse(mpg$total >=30, "A"
                    , ifelse(mpg$total >=20, "B", "c"))    #c는 조건에 맞지 않을 때 부여 
head(mpg,20)

table(mpg$grade)
qplot(mpg$grade)


#원하는 만큼 범주 만들기
mpg$grade2 <- ifelse(mpg$total >= 30, "A"
                     , ifelse(mpg$total >= 25, "B"
                              , ifelse(mpg$total >= 20, "C", "D")))
head(mpg, 20)
table(mpg$grade2)
qplot(mpg$grade2)     #histogram은 숫자형만 가능 

# 연습문제

midwest <- as.data.frame(ggplot2::midwest)
View(midwest)

midwest1 <- midwest
summary(midwest1)
str(midwest1)
View(midwest1)
midwest1 <- rename(midwest1, total = poptotal, asian = popasian)  
# df <- rename(df, new_var_name =old_var_name )
midwest1$asian_percent <- midwest1$asian/midwest1$total 
midwest1$asian_percent1 <- (midwest1$asian/midwest1$total) *100 

head(midwest1,10)

mean(midwest1$asian_percent1)
hist(midwest1$asian_percent1)

summary(midwest1$asian_percent1)

midwest1$test <- ifelse(midwest1$asian_percent1 >= 0.4872, "large", "small" )

head(midwest1)

table(midwest1$test)
qplot(midwest1$test)




# 2019-07-04 

# 0. 준비  
# install.packages(readxl)
library(readxl)
exam <- read_excel("data/excel_exam.xlsx")
View(exam)
ex1 <- exam

# 1. 3과목의 합계
ex1$s_total <- ex1$math + ex1$english + ex1$science

# 2. 3과목 평균
ex1$s_mean <- (ex1$math + ex1$english + ex1$science)/3

head(ex1)

# 3. 평균 60점 이상 pass인 파생변수 생성
ex1$test1 <- ifelse(ex1$s_mean >= 60, "PASS", "FAIL")
ex1$test1

# 4. 3의 결과- 빈도수와 그래프  
table(ex1$test1)
qplot(ex1$test1)

# 5. 평균 80점 이상 a, 70점 이상 b, 그외 c인 파생변수 생성 
ex1$test2 <- ifelse(ex1$s_mean >= 80, "A"
                    ,ifelse(ex1$s_mean >= 70, "B", "C"))
ex1$test2

# 6. 5의 결과 상위 10개 
head(ex1, 10)
head(ex1$test2, 10)

# 7. 5의 결과 하위 7개 
tail(ex1,7)
tail(ex1$test2,7)

# 8. 5의 결과 - 빈도수와 그래프 
table(ex1$test2)
qplot(ex1$test2)



##### ch 6 자유자재로 데이터 가공하기 ######

# 데이터 전처리(data preprocessing) =가공, 핸들링, 랭글링, 먼징

#library(dplyr)
#filter() 행추출
#select() 열(변수) 추출
#arrange() 정렬
#matate() 변수 추가
#summarise() 통계치 산출
#group_by() 집단별로 나누기
#left_join() 데이터 합치기(열)
#bind_rows() 데이터 합치기(행)
#rename() 변수명 변경

# R에서 사용하는 기호들
# 논리 연산자 :  <,  >,  <=,  >=,  ==,  !=,  |(=또는),  &,   %in%(=매칭 확인)
# 산술 연산자 :  +,  =,  * ,  / ,  **(=^),    %/%(=나눗셈의 몫),    %%(=모듈러, 나눗셈의 나머지)


# 행 추출 : 조건에 맞는 데이터만 추출하기  : df %>% filter(var1 논리연산자 숫자 ) 

library(dplyr)
exam <- read.csv("data/csv_exam.csv")
exam


# 1반, 2반 추출
exam %>% filter(class == 1) 
exam %>% filter(class == 2)
exam %>% filter(class == 3)

# 1반/3반이 아닌 경우
exam %>%  filter(class != 1)
exam %>%  filter(class != 3)

# 초과, 미만, 이상, 이하 조건 걸기

exam %>%  filter(math > 50)
exam %>%  filter(math < 50)
exam %>%  filter(english >= 80)
exam %>%  filter(english <= 80)

# 여러 조건을 충족하는 행 추출 : 그리고and를 의미하는 & 
exam %>%  filter(class == 1 & math >= 50)       # 1반 & 수학 50점 이상
exam %>%  filter(class == 2 & english >= 80)    # 2반 & 영어 80점 이상
exam %>%  filter(class == 2 & science >= 80)


# 여러 조건중 하나 이상 충족하는 행 추출 : 또는or 을 의미하는 | (=vertical bar)
exam %>%  filter(math >= 90 | english >= 90)     # 수학 90이상 or 영어 90이상
exam %>%  filter(english < 90 | science < 50 )   # 영어 90미만 or 과학 50미만 


# 목록에 해당하는 여러개의 행 추출 df %>% filter(var %in% c(a,b,c..))
# %in% == matching operator 매치 연산 (매칭확인)

exam %>%  filter(class == 1 | class == 2 | class == 3)  #1반 or 2반 or 3반 
exam %>%  filter(class %in% c(1,3,5) )   #1,3,5반에 해당하면 추출


# 추출한 행으로 새로운 데이터 만들기 
class1 <- exam %>%  filter(class == 1)
class1
mean(class1$math)

class2 <- exam %>%  filter (class == 2)
class2
mean(class2$math)

totalmean <- sum(class1$math + class2$math) / (nrow(class1)+nrow(class2))
totalmean

nrow(class1)
nrow(class2)  

# 연습문제
mpg <- as.data.frame(ggplot2::mpg)  # 패키지에 있는 데이터 불러오기 

displ4 <- mpg %>%  filter (displ <= 4)  # displ1 == 배기량(displ)이 4이하인 차량
displ4
displ5 <- mpg %>%  filter (displ >= 5)  # displ2 == 배기량이 5이상인 차량
displ5

mean(displ4$hwy)    # >>> 25.963
mean(displ5$hwy)    # >>> 18.078

audi <- mpg %>%  filter(manufacturer == "audi") 
audi
toyota <- mpg %>%  filter(manufacturer == "toyota")
toyota

mean(audi$cty)   # >>> 17.611 
mean(toyota$cty) # >>> 18.529

# 이렇게 하면 각각 구해짐 
chevrolet <- mpg %>%  filter(manufacturer == "chevrolet")
chevrolet
ford <- mpg %>%  filter (manufacturer == "ford")
ford
honda <- mpg %>%  filter (manufacturer =="honda")
honda

# chevrolet or ford or honda 한번에 추출

car3 <- mpg %>% filter(manufacturer %in% c("chevrolet", "ford", "honda"))  #in 대문자로 쓰면 x
car3
mean(car3$hwy)  # >>> 22.509





#복습 

getwd()  #
library(readxl)
library(dplyr)   # filter쓰기 위해 필요 
data1 <- read.csv("data/2013년_프로야구선수_성적.csv")
data1
View(data1)


data_a <- data1 %>% filter(경기 >= 120)
data_a

data_b <- data1 %>% filter(경기 >= 120 & 득점 >= 80)
data_b

data_c <- data1 %>% filter(포지션 %in% c("1루수","3루수"))   #둘다 뽑으려면 ==대신에 %in% 
data_c

# 필요한 변수만 추출하기 select (열=변수 추출) 

# df <- select (var1, var2,...)
# df <- select (-var1, -var2...)   var1, var2... 빼고 추출 

library(readxl)
library(dplyr)

exam <- read.csv("data/csv_exam.csv")
exam

exam %>% select(math)
exam %>% select(english)

exam %>% select(class, math, english)
exam %>% select(-math)   # 수학 빼고 추출 
exam %>% select(-math, -english)  # 수학, 영어 빼고 추출 


# dplyr 함수 조합하기 
# dplyr 패키지의 함수들은 %>% 를 이용해 조합가능 

# 1) filter()와 select()조합 (필요한 행과 열 추출)
exam %>% filter(class == 1) %>% select(english)    # <- class == 1 먼저 선택하고 english 추출!  행->열 순서로 선택하기 
exam %>% filter(class == 1) %>% select(class, english)  # <- class도 나타내려고 할 때 뒤에 한번더 써줘야함 
exam %>% select(english) %>% filter(class == 1)    # >>> Error: Result must have length 20, not 4 

# 2) 가독성 있게 줄 바꾸기 대신 전체 한번에 실행! %>% + enter
exam %>% 
  filter(class == 1) %>%
  select(english) 

# 3) 일부만 출력하기  head() 사용 head >>> default:6
head(exam)
exam %>% 
  select(id, math) %>% 
  head(5)

# 연습문제
mpg <- as.data.frame(ggplot2::mpg)
mpg
mpg_1 <- mpg %>%
  select(class, cty) %>%
  head(10)
mpg_1

mpg_2 <- mpg %>% 
  select(class,cty)
mpg_2

mpg_2_s <- mpg_2 %>% filter(class=="suv")
mpg_2_s
mpg_2_c <- mpg_2 %>% filter(class=="compact")
mpg_2_c

test <- data.frame (mean(mpg_2_s$cty), mean(mpg_2_c$cty))
test


# 순서대로 정리하기 arrange()
# df %>% arrange(var) 오름차순(낮->높) 
# df %>% arrange(desc(var)) 내림차순 (높->낮)
# df %>% arrange(var1, var2) var1, var2 순서로 오름차순(낮->높)

exam %>%  arrange(math)           # math 오름차순 (낮->높)
exam %>%  arrange(desc(math))     # math 내림차순 (높->낮)
exam %>%  arrange(id, math)       # id 오름차순-> math 오름차순
exam %>%  arrange(desc(math),id)  # 먼저 쓴게 기준 
exam %>%  arrange(desc(math,id))
exam %>%  arrange(desc(id,math))

exam %>% 
  filter(class==3) %>% 
  select(-class) %>% 
  arrange(math) %>% 
  head

exam %>% 
  select(id, class, math) %>% 
  arrange(class,desc(math))   #반 별 정렬, 수학 점수 높은것부터 내림차순


# 연습문제

mpg <- as.data.frame(ggplot2::mpg)
mpg

audi <- mpg %>% filter(manufacturer=="audi") %>% 
  select(manufacturer,model,cty,hwy) %>% 
  arrange(desc(hwy)) %>% 
  head(5)
audi


# 파생변수 추가하기-기존 데이터에 파생변수 추가! : mutate(new_var = var1+var2..) 
# 데이터 프레임에서 배운건 변수명을 계속 써줘야함! 여기서 하는게 더 간편 
exam %>% 
  mutate(total = math + english + science) %>% 
  head

exam %>%                                 
  mutate(total = exam$math + exam$english + exam$science) %>%    # 이렇게 해도 되긴 됨! 
  head

# 여러 개의 파생변수 한번에 추가하기 mutate(new_var1 = 식, (enter) new_var2= 식)
exam %>% 
  mutate(total = math + english + science,
         mean = (math+english+science)/3 ) %>% 
  head

# mutate() 에 ifelse() 적용하기
exam %>% 
  mutate(test = ifelse(science >= 60, "PASS", "FAIL")) %>% 
  head

# 추가한 변수를 dplyr코드에 바로 활용하기     
exam %>% 
  mutate(total = math + english + science,
         mean = (math + english + science) / 3 ,
         test = ifelse(mean >= 70, "PASS", "FAIL")) %>% 
  select(id,class,total,mean,test) %>% 
  arrange(desc(mean),id) %>% 
  head(20)



# 복습 

getwd()  
library(readxl)
library(dplyr)   
data1 <- read.csv("data/2013년_프로야구선수_성적.csv")
data1

# 5.
data1 %>% 
  select("선수명", "포지션", "팀")

# 6.
data1 %>% 
  select ("순위", "선수명", "포지션", "팀", "경기", "타수")

# 7. 
data1 %>% 
  select (-"홈런", -"타점", -"도루")

# 8.
data1 %>% 
  filter(타수 > 400) %>%     # variable은 ""빼고 작성 
  select("선수명", "팀", "경기", "타수")  

# 9.
data1 %>% 
  filter(타수 > 400) %>%   
  select("선수명", "팀", "경기", "타수")  %>% 
  arrange(desc(타수))

# 10.
data1 %>% 
  filter(타수 > 400) %>%   
  select("선수명", "팀", "경기", "타수")  %>%   # " " 안써도 됨 
  mutate("경기*타수" = 경기*타수)     

# 10.1
data1 %>% 
  filter(타수 > 400) %>%   
  select("선수명", "팀", "경기", "타수")  %>%   # " " 안써도 됨 
  mutate(경기x타수 = 경기*타수)                 # 이렇게 해도 됨 


# 연습문제 

mpg <- as.data.frame(ggplot2::mpg)
mpg1<-mpg

mpg1 %>% 
  mutate(sum=hwy+cty) %>% 
  head

mpg1 %>% 
  mutate(sum = hwy + cty
         ,mean_sum=(hwy+cty)/2) %>% 
  arrange(desc(mean_sum)) %>% 
  head(20)

summary(mpg)
head(mpg,10)

mpg %>% 
  mutate (sum = hwy + cty,
          mean_sum = (hwy + cty)/2) %>% 
  arrange(desc(mean_sum)) %>% 
  select(-displ, -year, -cyl, -trans, -fl) %>%
  head(20)


mpg %>% 
  mutate (sum = hwy + cty,
          mean_sum = (hwy + cty)/2,
          test = ifelse(mean_sum >= 30, "a", ifelse(mean_sum >= 25, "b", "c") )) %>% 
  arrange(desc(mean_sum), manufacturer) %>% 
  select(-displ, -year, -cyl, -trans, -fl) %>%
  head(30)



# 집단 별로 요약하기 : group_by(),  summarise()

exam %>%  
  summarise(mean_math = mean(math))  #summarise(new_var = 함수 )

exam %>% 
  group_by(class) %>%                # 변수지정, 변수별로 데이터 분리
  summarise(mean_math = mean(math))  # 집단 별 요약 통계량 산출 

# 출력결과에 a tibble : 5 x 2
# tibble = df의 업그레이드 버전, 몇가지 기능 추가.
# int = integer정수, dbl=double 소수점이 있는 숫자(double:부동소수점)   


# summarise()에 자주 사용하는 요약 통계량 함수 - mean(), sd():표준편차, sum(), median(), min(), max(), n(): 빈도 

exam %>% 
  group_by(class) %>% 
  summarise(mean_math = mean(math),
            sum_math = sum(math),
            median_math = median(math), 
            n = n())                      # n=n()에 변수명 입력 안함 - 행의 갯수를 세는 것


# 각 집단별로 다시 집단 나누기 - 집단안에 하위 집단 만들기  
# 1) group_by(var1, var2) var1집단, var2집단으로 (보통 var1만 작성함 summarise에서 작성한것도 결과 나옴 )
# 2) summarise에서 만든 새 변수와 위에서 지정한 집단만 나옴 

mpg %>% 
  group_by (manufacturer, drv) %>%          # manufacturer로 나누고, drv로 한번더 나누고
  summarise(mean_sty = mean(cty)) %>% 
  head(10)


# 직접 해보기 : 회사별 :suv 자동차의 도시 및 고속도로 통합 연비 평균을 구해 내림차순, 1-5위까지 구하기

mpg %>% 
  group_by(manufacturer, class) %>%     # 우선 제조사 ->  class로 집단 나누고   # 여기에 class안쓰면 밑에서 안나옴 
  filter(class == "suv") %>%              # class가 suv인것만 
  mutate(total = (cty + hwy)/2 ) %>%    # 여기가 통합 연비 새변수 생성 
  summarise(mean_total = mean(total),   # 그룹(제조사)별로 통합연비의 평균을 구함
            n = n()) %>%   
  arrange(desc(mean_total)) %>% 
  head(5)


# 연습문제 

mpg %>% 
  group_by(class) %>% 
  summarise(cty_mean = mean(cty)) %>% 
  arrange(class) %>% 
  # summarise(hwy_mean = mean(hwy)) %>%  summarise 두개 같이 안됨 
  head(5)

mpg %>% 
  group_by(manufacturer) %>% 
  filter(class== "compact") %>% 
  summarise(n=n()) %>% 
  arrange(desc(n))


# 데이터 합치기
# 가로(변수): left_join(df1, df2, by = "연결할 기준 variable")  주의) "var"
# 세로(행) : bind_rows(df1, df2)  
# 주의)df1과 df2의 변수명 같아야됨 다르다면 rename()   # df <- rename(df, 새로운 변수명 = 기존 변수명) 

# 가로로 합치기 = column 추가! : left_join(df1, df2, by = "연결할 기준 variable")  
test1 <- data.frame (id = c(1, 2, 3, 4, 5),
                     mid = c(60, 80, 70, 90, 85))
test1

test2 <- data.frame (id = c(1, 2, 3, 4, 5),
                     final = c(70, 83, 65, 95, 80))
test2

total <- left_join(test1, test2, by = "id")   # id를 기준으로 total에 할당 
total 

# 다른 데이터를 활용해 변수 추가하기
name <- data.frame (class = c(1, 2, 3, 4, 5),
                    teacher = c("Kim", "Lee", "Park", "Choi", "Jung"))
name

exam_new <- left_join(exam, name, by = "class")   
exam_new


# 세로로 합치기 = row 추가! : bind_rows(df1, df2) / 변수명이 다르면  # df <- rename(df, 새로운 변수명 = 기존 변수명) 
group_a <- data.frame (id = c(1, 2, 3, 4, 5),
                       test = c(60, 80, 70, 90, 85))
group_a 

group_b <- data.frame (id = c(6, 7, 8, 9, 10),
                       test = c(70, 83, 65, 95, 80 ))
group_b

total <- bind_rows(group_a, group_b)
total

# 변수명이 다를경우 rename() 먼저 
a <-  data.frame (id = c(1, 2, 3, 4, 5),
                  test = c(60, 80, 70, 90, 85))
a
b <-  data.frame (id = c(6, 7, 8, 9, 10),
                  final_test = c(70, 83, 65, 95, 80 ))
b
all <- bind_rows (a, b)    # 변수명이 다를경우 총 id / test / final_test 로 3개의 열이 나옴! (없으면 na처리)
all

b <- rename(b, test = final_test)
b

all <- bind_rows(a,b)
all


# 연습문제

str(mpg)
View(mpg)

fuel <- data.frame (fl = c("c", "d", "e", "p", "r"),
                    price_fl = c(2.35, 2.38, 2.11, 2.76, 2.22),
                    stringsAsFactors = F)    # factors(범주형)아니고 values니까 F로! (그래야 chr 문자형으로 인식)
fuel
str(fuel)

mpg1<- left_join(mpg, fuel, by = "fl") 
mpg1

mpg1 %>% 
  select(model, fl, price_fl) %>% 
  head(5)

# 연습문제
midwest <- as.data.frame(ggplot2::midwest)
View(midwest)
summary(midwest)
str(midwest)

mid1<- midwest
mid1 %>% 
  mutate(child_per = (poptotal-popadults)/poptotal * 100,
         adult_per = (popadults/ poptotal) * 100,
         test = ifelse(child_per >= 40, "Large", ifelse(child_per >=30, "Middle", "small"))) %>% 
  select (county, poptotal, popadults, child_per, adult_per, test) %>% 
  arrange(desc(child_per)) %>% 
  group_by(test) %>% 
  summarise(n=n())


mid2 <- midwest
mid2 %>%  
  mutate(asian_per = (popasian/poptotal) * 100) %>% 
  select(county,state,popasian, poptotal, asian_per) %>% 
  arrange(asian_per) %>%         #여기서 arrange해야 sorting 됨! (default: 낮-> 높)
  head(10)         








##### ch 7 데이터 정제 (데이터 오류 수정 / 처리) #####

# 결측치 : Missing Value. 문자형은 <NA>, 숫자형은 NA  / 결측치 있으면 함수 계산 불가! 
# df생성할때 값에  "NA" 라고 넣으면 Value로 인식 

# 결측치 찾기

df <- data.frame(sex = c("M", "F", NA, "M", "F"),    
                 score = c(5, 4, 3, 4, NA))
df

# 결측치 확인하기 
# is.na(df)  -> is it na? 결측치 >>> T(True) /   Value(데이터 값있음) >>> F(False)
# table(is.na(df)) -> T,F의 갯수 확인 
is.na(df)    
table(is.na(df))   # df전체의 결측치 갯수 확인 
table(is.na(df$sex))  #df의 특정 변수 안의 결측치 갯수 확인  >>> F:4, T:1 -> 결측치 1개 
table(is.na(df$score)) 

mean(df$score)   # >>> NA 
sum(df$score)    # >>> NA

# 결측치 제거하기 
library(dplyr)
df %>% filter(is.na(score))   # score에 결측치 있는거 가져와
df %>% filter(!is.na(score))  # score에 결측치 아닌거 가져와! is 앞에 느낌표    즉 이렇게 결측치 없애기! 

df_nomiss_s <- df %>%  filter(!is.na(score))
df_nomiss_s

mean(df_nomiss_s$score)   # >>> 4
sum(df_nomiss_s$score)    # >>> 16


# 여러 변수 동시에 결측치 없는 데이터 추출하기 (na.omit(df)보다 filter(!is.na(var))추천! )
df_nomiss_all <- df %>% filter(!is.na(score) & !is.na(sex))
df_nomiss_all  # 5개에서 3개됨 

# 결측치가 하나라도 있으면 제거하기 : na.omit(df)  모든 변수에 결측치 없는 데이터 추출 (결측치가 있으면 그 행 다 삭제)
df_nomiss <- na.omit(df)
df_nomiss 

# 함수의 결측치 제외기능 이용하기
# mean() 같은 수치연산 함수는 na.rm 파라미터 지원! (지원안하는 함수는 filter로 먼저 제거하고 함수적용)
# function(df$var, na.rm = T)  : 결측치 생략하고 function 적용 / na.rm : NA remove
mean(df$score, na.rm= T)
sum(df$score, na.rm= T)

exam <- read.csv("data/csv_exam.csv")
exam [c(3, 8, 15), "math"] <-NA  # 3,8,15행의 math에 NA할당 
exam
# [ 행, 열 ] : 데이터의 위치 지정  df[c(a1, a2, ..), "variable"] <- NA 


# summarise() 에도 na.rm = T 적용가능   -> df %>% summarise(new_var = func(var, na.rm = T)) : 결측치 생략하고 함수계산
exam %>% 
  summarise(mean_math = mean(math))
exam %>% 
  summarise(mean_math = mean(math, na.rm = T),
            sum_math = sum(math, na.rm = T),
            median_math = median(math, na.rm = T))


# 결측치 대체법( = Imputation )
# 대표값(평균or최빈값 등)을 구해 모든 결측치를 하나의 값을 일괄 대체하는 법
# 통계적으로 여러 방법이 있고, 예측값을 추정해 대체하는 방법도 있음

# 평균값으로 결측치 대체하기
exam <- read.csv("data/csv_exam.csv")
exam1 <- exam
exam1 [c(3, 8, 15), "math"] <-NA  # 3,8,15행의 math에 NA할당 
exam1  # 결측치 포함한 df 
mean(exam1$math, na.rm = T)   # >>> 55.235
exam1$math <- ifelse(is.na(exam1$math), 55, exam1$math )   # 결측치면 55로 대체, 값있으면 그대로 
table(is.na(exam1$math))  #>>> false : 20 ! 결측치 없음 
exam1

# 연습 문제 
library(ggplot2)    # mpg <- as.data.frame(ggplot2::mpg) 라고 안쓰고 library만 불러도 됨(패키지 내장 데이터라서!)
library(dplyr)
mpg
mpg1 <- mpg

mpg1[c(65, 124, 131, 153, 212), "hwy"] <-NA

is.na(mpg1)
is.na(mpg1$drv)
is.na(mpg1$hwy)
table(is.na(mpg1$drv))
table(is.na(mpg1$hwy))
table(!is.na(mpg1$hwy))

mpg2 <- mpg1 %>% 
  filter(!is.na(mpg1$hwy)) %>% 
  group_by(drv) %>%                  # 여기서 hwy말고 drv로만 그룹 나눠야함! 
  summarise(hwy_mean = mean(hwy)) %>% 
  arrange(desc(hwy_mean))
mpg2



# 이상치 outlier 정제하기

# 1.이상치 제거 - 존재할 수 없는 값

# 1) 이상치 있는 df 생성 (sex = 1 or 2 /  scor e= 1~5)
outlier <- data.frame(sex = c(1, 2, 1, 3, 2, 1),
                      score = c(5, 4, 3, 4, 2, 6))
outlier

# 2) 이상치 확인하기 -> 빈도표 작성 table() 
table(outlier$sex)
table(outlier$score)

# 3) 결측 처리하기
outlier$sex <- ifelse(outlier$sex == 3, NA, outlier$sex)
outlier
outlier$score <- ifelse(outlier$score > 5, NA, outlier$score)
outlier

# 4) 분석할 때 filter() 사용하여 제외하고 분석
# 성별에 따른 평균 점수 
outlier %>% 
  filter(!is.na(sex) & !is.na(score)) %>%     #is.na(var)  <- var쓸 때 df$안써도 됨 
  group_by(sex) %>% 
  summarise(mean_score= mean(score))


# 2.이상치 제거 - 극단적인 값 
# 극단치 - 논리적으론 존재 가능하지만 극단적으로 크거나 작 
# 정상범위를 정하거나, 통계기준 이용(예: boxplot)
# boxplot - 상자 내 굵은 선: median, 
# 극단치 경계 : Q1,Q3밖 1.5IQR의 최대값 (1.5IQR= 사분위 범위(Q1~Q3의 거리)의 1.5배)

# 1) 상자그림으로 극단치 정하기 
library(ggplot2)
boxplot(mpg$hwy)

# 2) 상자그림 통계치 확인하기 
boxplot(mpg$hwy)$stats 
#12 - 극단치 경계 (1.5IQR내에 최소값) (0~25%)   - 데이터 값이 18 이하이고 1.5IQR(Q3-Q1)범위내의 최소값
#18 - 1사분위수 (25%, 상자 밑면)                - 데이터 중 25%인 값
#24 - 2사분위수 (중앙값, 상자 굵은 선)          - 데이터 중 중앙값 Median
#27 - 3사분위수 (75%, 상자 윗면)                - 데이터 중 75%인 값
#37 - 극단치 경계                               - 데이터 중 q3 + 1.5IQR(Q3-Q1)범위내의 최대값
#   >>> 12~37을 벗어나면 극단치로 분류

# 9*1.5 = 1.5IQR
# 13.5 +27 = 40.5 까지 극단치 경계이지만 그중 가까운 값인 37이 경계  그 이상은 이상치 


# 3) 결측 처리하기
mpg$hwy <- ifelse(mpg$hwy < 12| mpg$hwy > 37, NA, mpg$hwy)
table(is.na(mpg$hwy))

# 4) 결측치 제외하고 간단한 분석
mpg %>% 
  group_by (drv) %>% 
  summarise(mean_hwy = mean(hwy, na.rm = T))


# 연습문제 
mpg1 <- mpg
mpg1[c(10, 14, 58, 93), "drv"] <- "k"
mpg1[c(29, 43, 129, 203), "cty"] <- c(3, 4, 39, 42)

table(mpg1$drv)
mpg1$drv <- ifelse(mpg1$drv == "k", NA, mpg1$drv)     
# %in% 사용하려면 -> mpg1$drv <-ifelse(mpg1$drv %in% c("4", "f", "r"), mpg1$drv, NA)
table(mpg1$drv)
table(is.na(mpg1$drv)) #>>> true =4 나옴! k 4개를 결측처리함

boxplot(mpg1$cty)
boxplot(mpg1$cty)$stats  #>>> 9미만26초과하면 이상치 

mpg1$cty <- ifelse(mpg1$cty< 9 | mpg1$cty > 26, NA, mpg1$cty) 
boxplot(mpg1$cty)

mpg1 %>% 
  filter(!is.na(mpg1$cty) & !is.na(mpg1$drv)) %>%    # 여기서 !is.na(mpg1$drv) 안하면 4행에 na 나옴! 
  group_by(drv) %>% 
  summarise(mean_cty = mean(cty))





##### ch 8 그래프 만들기 #####

# 그래프 저장하기 plots - image, pdf, clipboard 가능 

library(ggplot2)
library(dplyr)
View(mpg)
# qplot은 데이터 전처리 단계에서 빠르게 확인할 때 사용, ggplot은 세부요소 설정 가능 
# ggplot layer 구조 (1: 배경(축) + 2: 그래프 추가(점, 선, 막대) + 3: 설정추가(축 범위, 색, 표식))


# 산점도 그리기  geom_point(size=, color =' ') + xlim(a,b) + ylim(c,d)
# 산점도 : scater plot, 두개의 변수의 관계를 점으로 표현 
# ggplot(data=df, aes(x= var1, y=var2, fill = var1)) : x,y축 변수지정, 배경 생성, fill = var1 변수별 색상 구분 
#  aes : 데이터 속성(aesthetics 미학요소)
# ggplot 패키지는 + 로 함수 연결 (dplyr은 %>% 로 연결 )
# geom_point : 산점도 그리는 함수 (관측치들 점으로) 
# 축 범위 설정 : xlim(a,b) ylim(c,d)  x축:a~b범위 

ggplot(data = mpg, aes(x = displ, y = hwy))   

ggplot(data = mpg, aes(x = displ, y = hwy)) + 
  geom_point()   

ggplot(data = mpg, aes(x = displ, y = hwy)) + 
  geom_point() + 
  xlim(3, 6) +
  ylim(10,30)                             
# Warning message: Removed 105 rows containing missing values (geom_point).  <- 축 바뀌어서 

# 연습문제 

ggplot (data = mpg, aes(x = cty, y = hwy)) +
  geom_point(size = 3, color = 'blue') 

View(midwest)

ggplot (data = midwest, aes(x= poptotal, y=popasian )) +
  geom_point() +
  xlim(0, 500000) +
  ylim(0, 10000)


# 막대그래프 - 집단 간 차이 표현 bar chart 
# 1. 평균 막대 그래프 만들기 + geom_col() 요약정보(평균표)를 만들고 생성 
# geom_col(position = "dodge") -> 파라미터 따로 생성 

# 1) 집단별 평균표 만들기
library(dplyr)
df_mpg <- mpg %>% 
  group_by(drv) %>% 
  summarise(mean_hwy = mean(hwy))
df_mpg
# 2) 그래프 생성하기    # ggplot(data=df, aes(x=var1, y=var2)) + geom_col() : 막대그래프 
ggplot(data = df_mpg, aes(x = drv, y = mean_hwy)) +
  geom_col()
# 3) 크기순 정렬   # reorder() : 막대를 값의 크기 순으로 정렬,  - 붙이면 내림차순(큰->작)
# ggplot(data=df, aes(x= reorder(var1, (-)정렬기준변수), y= 정렬기준변수)) + geom_col()
ggplot(data= df_mpg, aes(x= reorder(drv, -mean_hwy), y= mean_hwy)) +
  geom_col()
# 숫자+문자 섞인 변수는 숫자 오름차순(0~), 알파벳 오름차순(a~)으로 정렬됨 


# 2. 빈도 막대 그래프 만들기 : 값의 개수(=빈도)로 막대의 길이 표현 x축만 지정 
# ggplot(data=df, aes(x=var) + geom_bar()   <- y축은 count! 원자료 이용 

# x 범주형 drv: f, r, 4 3가지 종류 
ggplot(data = mpg, aes(x = drv)) +
  geom_bar() 
# x 연속형 hwy : 12~44의 값 
ggplot(data = mpg, aes(x = hwy)) +
  geom_bar()


# 연습문제
library(ggplot2)
mpg1 <- mpg
View(mpg1)
mpg2<- mpg1 %>% 
  filter (class == "suv") %>% 
  group_by(manufacturer) %>% 
  summarise(mean_cty = mean(cty)) %>% 
  arrange(desc(mean_cty)) %>% 
  head(5)
mpg2

ggplot(data = mpg2, aes (x= reorder(manufacturer, -mean_cty), y= mean_cty) ) +
  geom_col()

ggplot(data= mpg1, aes(x=class)) +
  geom_bar()


# 선 그래프 - line chart. 시계열 그래프(time series chart)  
# ggplot(data = df, aes(x= var1, y=var2 )) + geom_line()
economics <- as.data.frame(ggplot2::economics)
eco <- economics
View(eco)
ggplot(data = eco, aes(x = date, y = unemploy)) +
  geom_line()

# 연습문제
ggplot (data= eco, aes(x=date, y=psavert)) +
  geom_line()
# 시계열 그래프 기간 정하기
a <- ggplot(data=eco, aes(x=date, y=psavert)) +
  geom_line()
  # set axis limits c(min, max)   # date 타입이라 따로 지정하고 합쳐줘야 함
min <- as.Date("2002-1-1")
max <- as.Date("2010-1-1")
a + scale_x_date(limits = c(min,max))          

# 선그래프 + (평균 선)수직선 그리기
ggplot(data=eco, aes(x = date, y = psavert)) +
  geom_line() +
  geom_hline(yintercept=mean(eco$psavert))


# 상자 그림 box plot : 데이터의 분포를 직사각형 상자모양으로 표현
# ggplot(data=df, aes(x=var1, y=var2)) + geom_boxplot()
ggplot(data=mpg, aes(x=drv, y=hwy)) +
  geom_boxplot()

# 연습문제
mpg1 <- mpg
mpg2 <- mpg1 %>% 
  filter(class %in% c("compact", "subcompact", "suv")) 
mpg2
ggplot(data= mpg2, aes(x=class, y=cty)) +
  geom_boxplot()


# ggplot2 함수
# geom_point()  산점도
# geom_col()  막대 그래프 - 요약표
# geom_bar()  막대 그래프 - 원 자료
# geom_line()  선 그래프
# geom_boxplot() 상자그림(var 여러개 )



# 복습 
# 0.
getwd()
library(readxl)
library(ggplot2)

# 1.
data1 <- read.csv("data/2013년_프로야구선수_성적.csv")
data1
View(data1)

data2<-data1 %>% 
  select(팀, 경기) %>% 
  group_by(팀) %>% 
  summarise(mean_play = mean(경기))
data2

# 2.
ggplot(data = data2, aes(x=팀, y=mean_play)) +
  geom_col()

# 3.
grade <- read.csv("data/학생별국어성적_new.txt")
grade
View(grade)

# 4.
ggplot(data=grade, aes(x = 이름, y = 점수 )) +
  geom_point()

# 5. 
ggplot(data = grade, aes(x = 이름, y = 점수)) +
  geom_col()




##### ch 9 데이터 분석 프로젝트 #####

# 9-1. 한국복지패널데이터 분석 준비 
# install.packages("foreign")   #spss/sas/stata 파일 불러오기 
getwd()
library(foreign)
library(dplyr)           # 전처리 
library(ggplot2)
library(readxl)

raw_welfare <- read.spss(file = "data/Koweps_hpc10_2015_beta1.sav", to.data.frame = T)   # spss파일을 df로 불러오기
View(raw_welfare)
wel <- raw_welfare

head(wel)
tail(wel)
View(wel)
dim(wel)
str(wel)
summary(wel)

wel <- rename (wel, 
               sex = h10_g3,
               birth = h10_g4,
               marriage = h10_g10,
               religion = h10_g11,
               income = p1002_8aq1,
               code_job = h10_eco9,
               code_region = h10_reg7)
head(wel)
wel %>% 
  select(sex,birth, marriage, religion, income, code_job, code_region) %>% 
  head(20)

# 그래프 색상은 ggplot(,fill = var)
# 파라미터 표현은 +geom_col(position = "dodge")
# ?geom_col

# 분석 절차 1) 변수 검토 및 전처리 2) 변수 간 관계분석 

# 9-2. 성별에 따른 월급차이 

# 성별 검토 전처리
class(wel$sex)  # numeric
table(wel$sex)  # 1(남): 7578, 2(여):9086, 9(이상치): 0
wel$sex <- ifelse(wel$sex == 9, NA, wel$sex)  # 이상치를 결측 처리 
table(is.na(wel$sex))  #결측치없음 -> false 16664

wel$sex <- ifelse(wel$sex == 1, "male", "female")
table(wel$sex)  # 1,2를 male, female로 변경
qplot(wel$sex)  

# 월급 검토 전처리
class(wel$income)
table(wel$income)
summary(wel$income)
qplot(wel$income)

qplot(wel$income) + xlim(0,1000)
summary(wel$income)

wel$income <- ifelse(wel$income %in% c(0,9999), NA, wel$income) #월급이 없는 사람 결측치로 
table(is.na(wel$income))  # <- 결측치 true : 12044개

table(wel$sex)

# 성별에 따른 월급차이 분석 - 평균표 만들기
sex_income <- wel %>% 
  filter(!is.na(income)) %>%   # 결측치 제외! 
  group_by(sex) %>% 
  summarise(mean_income = mean(income))
sex_income

ggplot(data=sex_income, aes(x = sex, y = mean_income )) +
  geom_col()


# 9-3. 나이와 월급의 관계

class(wel$birth)
summary(wel$birth)
qplot(wel$birth)
summary(wel$birth)

table(is.na(wel$birth))
wel$birth <- ifelse(wel$birth == 9999, NA, wel$birth)    #무응답 : 9999
table(is.na(wel$birth))

wel$age <- 2015 - wel$birth +1  #age 파생변수 만들기 
summary(wel$age)
qplot(wel$age)
table(wel$age)

age_income <- wel %>% 
  filter(!is.na(income)) %>% 
  group_by(age) %>% 
  summarise(mean_income = mean(income))
head(age_income)

ggplot(data = age_income, aes(x = age, y = mean_income)) +
  geom_line()


# 9-4. 연령대에 따른 월급 차이
# 파생변수 만들기 (초/중/노년)
wel <- wel %>% 
  mutate(ageg = ifelse(age < 30, "Young",
                       ifelse(age <= 59, "Middle", "Old")))
table(wel$ageg)
qplot(wel$ageg)

ageg_income <- wel %>% 
  filter(!is.na(income)) %>% 
  group_by(ageg) %>% 
  summarise(mean_income = mean(income))
ageg_income

ggplot(data = ageg_income, aes(x = ageg, y = mean_income)) +
  geom_col()

ggplot(data= ageg_income, aes(x = ageg, y = mean_income)) +
  geom_col() +
  scale_x_discrete(limits = c("Young", "Middle", "Old"))   # 초/중/노년으로 범주형 변수 정렬 


# 9-5. 연령대 및 성별 월급 차이 (변수 3개)

sex_income <-wel %>% 
  filter(!is.na(income)) %>% 
  group_by(ageg,sex) %>%    # 나이세대별, 성별, summarise의 mean_income 
  summarise(mean_income = mean(income))
sex_income

ggplot(data= sex_income, aes(x = ageg, y = mean_income, fill=sex)) +
  geom_col()+
  scale_x_discrete(limits = c("Young", "Middle", "Old"))   #문자형은 그냥 쓰면 됨 scale_x_discret()

ggplot(data = sex_income, aes(x = ageg, y = mean_income, fill = sex)) +
  geom_col(position = "dodge") +               # position파라미터를 "dodge"로 설정해 막대 분리! 
  scale_x_discrete(limits = c("Young", "Middle", "Old"))

# 나이 및 성별 월급차이 분석하기
sex_age <- wel %>% 
  filter(!is.na(income)) %>% 
  group_by(age, sex) %>% 
  summarise(mean_income = mean(income))
head(sex_age, 20)
ggplot(data=sex_age, aes(x = age, y= mean_income, col = sex)) +
  geom_line()


# 9-6. 직업별 월급차이

# 전처리
getwd()
library(foreign)
library(dplyr)          
library(ggplot2)
library(readxl)

#raw_welfare <- read.spss(file = "data/Koweps_hpc10_2015_beta1.sav", to.data.frame = T)   # spss파일을 df로 불러오기
#View(raw_welfare)
#wel <- raw_welfare

# 직업 변수 전처리
class(wel$code_job)
table(wel$code_job)

library(readxl)
list_job <- read_excel("data/Koweps_Codebook.xlsx", col_names = T, sheet = 2)
head(list_job)
dim(list_job)


wel <- left_join(wel, list_job, id = "code_job")  #기준 변수인 code_job으로 
wel %>% 
  filter(!is.na(code_job)) %>% 
  select(code_job, job) %>% 
  head(10)

job_income <- wel %>% 
  filter(!is.na(job) & !is.na(income)) %>% 
  group_by(job) %>% 
  summarise(mean_income = mean(income))
head(job_income)

top10 <- job_income %>%   # 상위10
  arrange(desc(mean_income)) %>% 
  head(10)
top10

ggplot (data= top10, aes(x=reorder(job, mean_income), y= mean_income, fill = job)) +
  geom_col(position = "dodge") +
  coord_flip()       # x,y축 변경 


bottom10 <- job_income %>% 
  arrange(mean_income) %>% 
  head(10)

bottom10
ggplot(data = bottom10, aes(x = reorder(job, -mean_income), y = mean_income, fill = job)) +
  geom_col() +
  coord_flip() +
  ylim(0,850)  #top10과 비교하기 위해 y축 같게 

ggplot(data = bottom10, aes(x = reorder(job, -mean_income), y = mean_income, fill = job)) +
  geom_col() +
  coord_flip() +
  ylim(0,850)  #top10과 비교하기 위해 y축 같게 



# 9-7. 성별 직업 빈도 
# 남성 빈도수 높은 상위 10   빈도(개수)이므로 summarise에  n 지정하여 개수를 알아내야 함! 
job_male <- wel %>% 
  filter(!is.na(job) & sex == "male") %>% 
  group_by(job) %>% 
  summarise(n= n()) %>% 
  arrange(desc(n)) %>%    # 빈도니까 n의 갯수로 정렬 
  head(10)
job_male

# 여성 빈도수 높은 상위 10
job_female <- wel %>% 
  filter(!is.na(job) & sex == "female") %>% 
  group_by(job) %>% 
  summarise(n = n()) %>% 
  arrange(desc(n)) %>%  
  head(10)
job_female

# 성별 직업 빈도표 그래프 
ggplot(data = job_male, aes(x = reorder(job, n), y = n, fill = job)) +
  geom_col() +
  coord_flip()

ggplot(data = job_female, aes(x = reorder(job, n), y = n, fill = job)) +
  geom_col() +
  coord_flip()

# 그래프 합치기 - 찾기 



# 9-8. 종교 유무에 따른 이혼율(비율)
# 변수 검토 
class(wel$religion)
table(wel$religion)   #1:종교 있음, 2:종교 없음, 9: 모름/무응답

wel$religion <- ifelse(wel$religion == 1, "Y", "N")
table(wel$religion)

qplot(wel$religion)      
# 종교 유무에 따른 이혼율 
class(wel$marriage)   #numeric을 밑에서 char로 바꿈 
table(wel$marriage)   # 0:비해당(18세미만), 1:배우자ㅇ, 2:사별, 3: 이혼, 4:별거, 5:미혼(18세이상, 미혼모포함), 6:기타(사망등)

wel$group_marriage <- ifelse(wel$marriage == 1, "Marriage", 
                             ifelse(wel$marriage == 3, "Divorce", NA))
table(wel$group_marriage)   #결혼: 8431, 이혼: 712
class(wel$group_marriage)   # class = char 
table(is.na(wel$group_marriage)) # 전체 결측치(결혼, 이혼 제외 나머지) : true 7521 -> 결측치!
qplot(wel$group_marriage)

# 반올림 표현하기 round(var, x)
# dplyr은 tibble형태로 나와서 가독성이 좋은 반올림을 알아서 함 
# -> %>% as.data.frame()을 추가하면 round(var, x) 적용 가능  

# 종교 유무에 따른 이혼율 분석 (mutate2번 사용)
religion_marriage <- wel %>% 
  filter(!is.na(group_marriage)) %>% 
  group_by(religion, group_marriage) %>% 
  summarise(n = n()) %>% 
  mutate(total_group = sum(n)) %>%    # 비율을 구하기 위해 n의 합계도 나타냄! 
  mutate(percent = round(n/total_group*100, 1)) 
religion_marriage
# 위와 같은 gr (count() 사용, mutate1번)
religion_marriage_1 <- wel %>% 
  filter(!is.na(group_marriage)) %>% 
  count(religion, group_marriage) %>% 
  group_by(religion) %>% 
  mutate(percent = round(n/sum(n)* 100, 1 ))
religion_marriage_1  
#count(var1, var2..)  : 집단별 빈도 구하기 함수 / 쓰면 결과에 total_group은 안나옴(sum 자동계산하기 때문)


# a <- df %>% filter() %>% (  select() %>%  ) count() %>% group_by() %>% mutate() 순서로 쓰기 권장 

# 이혼 추출
divorce <- religion_marriage %>% 
  filter(group_marriage == "Divorce") %>% 
  select(religion, percent)

divorce

ggplot(data= divorce, aes(x = religion, y = percent, fill = religion)) +
  geom_col()

# 연령대 및 종교 유무에 따른 이혼율 
# 1) 연령대별 이혼율 표 (아직 종교 없음)
ageg_marriage <- wel %>% 
  filter(!is.na(group_marriage)) %>% 
  group_by(ageg, group_marriage) %>%   #나이 연령대 (초/중/노)-> 결혼여부(이혼/결혼) -> 여부별 개수, 퍼센트
  summarise(n = n()) %>% 
  mutate(total_group = sum(n)) %>% 
  mutate(percent = round(n/sum(n)*100, 1))
ageg_marriage
# 위와 같은 gr
ageg_marriage_1 <- wel %>% 
  filter(!is.na(group_marriage)) %>% 
  count(ageg, group_marriage) %>% 
  group_by(ageg) %>% 
  mutate(percent = round(n/sum(n)*100, 1)) 
ageg_marriage_1

# 2) 연령대별 이혼율 그래프  - 초년 제외, 이혼추출
ageg_divorce <- ageg_marriage %>% 
  filter(ageg != "Young" & group_marriage == "Divorce") %>% 
  select(ageg, percent)
ageg_divorce

ggplot(data= ageg_divorce, aes(x = ageg, y = percent, fill = ageg )) +
  geom_col()

# wel$group_marriage   <- NA, Marriage, Divorce 로 나눠놓음 

# 3) 연령대별 종교 유무에 따른 이혼율 표 만들기 : group_by를 통해 한번에 표 만들기 , 초년 제외
ageg_religion_marriage <- wel %>% 
  filter(!is.na(group_marriage) & ageg != "Young" ) %>% 
  group_by(ageg, religion, group_marriage) %>% 
  summarise( n = n()) %>% 
  mutate(total_group = sum(n)) %>% 
  mutate(percent = round(n/total_group*100, 1))
ageg_religion_marriage   # -> 나이 연령(중/노년) , 종교, 결혼여부, n, sum(n), percent 나옴
# 위와 같은 gr
ageg_religion_marriage_1 <- wel %>% 
  filter(!is.na(group_marriage) & ageg != "Young") %>% 
  count(ageg, religion, group_marriage) %>% 
  group_by(ageg, religion) %>% 
  mutate(percent = round(n/sum(n)*100,1))
ageg_religion_marriage_1

# 연령대 및 종교 유무별 이혼율 표 만들기
df_divorce <- ageg_religion_marriage %>% 
  filter(group_marriage == "Divorce") %>% 
  select(ageg, religion, percent)
df_divorce   # ageg, religion, percent
divorce      # religion, percent 

# 4) 연령대별 종교 유무에 따른 이혼율 그래프 만들기
ggplot(data = df_divorce, aes(x = ageg, y = percent, fill = religion)) +
  geom_col(position = "dodge")



# 연습문제

# 1. 
marry <- wel %>% 
  filter(!is.na(group_marriage)) %>% 
  group_by(group_marriage, income) %>% 
  summarise(mean_income = mean(income))
marry

ggplot(data= marry, aes(x = group_marriage, y= mean_income, fill = group_marriage)) +
  geom_col(position = "dodge")


# 2. 

wel$group_marriage <- ifelse(wel$marriage == 1, "Marriage", 
                             ifelse(wel$marriage == 3, "Divorce", NA))
wel$group_marriage

marry_m_income <- wel %>% 
  filter(!is.na(group_marriage) & !is.na(income) & sex == "male") %>% 
  group_by(group_marriage,sex) %>% 
  summarise(mean_income = mean(income))
marry_m_income

marry_f_income <- wel %>% 
  filter(!is.na(group_marriage) & !is.na(income) & sex == "female") %>% 
  group_by(group_marriage, sex) %>% 
  summarise(mean_income = mean(income))
marry_f_income

ggplot(data= marry_m_income, aes(x = group_marriage, y = mean_income, fill = group_marriage)) +
  geom_col(position = "dodge")

ggplot(data = marry_f_income, aes(x = group_marriage, y = mean_income, fill = group_marriage)) +
  geom_col(position = "dodge")


marry_income_sex  <- wel %>% 
  filter(!is.na(group_marriage) & !is.na(income) & !is.na(sex)) %>% 
  group_by(group_marriage,sex) %>% 
  summarise(mean_income = mean(income))
marry_income_sex

ggplot(data=marry_income_sex, aes(x = group_marriage, y = mean_income, fill = sex)) +
  geom_col(position = "dodge")



# 9-9. 지역별 연령대 비율 - 노년층이 많은 지역은 어디? 
# 지역 rename 하고 wel에 추가
class(wel$code_region)
table(wel$code_region)
#code_region rename 
list_region <- data.frame(code_region = c(1:7),
                          region = c("서울",
                                     "수도권(인천/경기)",
                                     "부산/경남/울산",
                                     "대구/경북",
                                     "대전/충남",
                                     "강원/충북",
                                     "광주/전남/전북/제주도"))
list_region
# 새로만든 list_region을 wel에 추가
wel <- left_join(wel, list_region, id = "code_region")
wel %>% 
  select(code_region, region) %>% 
  head(20)

table(is.na(wel$region))  #<- 결측치 없음

# 1)지역별 연령대 비율표 만들기
region_ageg <- wel %>% 
  group_by(region, ageg) %>% 
  summarise(n= n()) %>% 
  mutate(total_group = sum(n)) %>% 
  mutate(percent = round(n/total_group*100,2))
region_ageg
# 위와 같은 gr  count 하고 group_by
region_ageg_1 <- wel %>% 
  filter(!is.na(region)) %>% 
  count(region,ageg) %>% 
  group_by(region) %>%     # count에서 region, ageg 빈도 했으니 여기선 중요한 지역별만 체크해주면됨  
  mutate(percent = round(n/sum(n)*100, 2))   # count해서 
region_ageg_1         

# 2) 그래프 그리기
ggplot(data = region_ageg, aes(x = region, y = percent, fill = ageg)) +
  geom_col(position = "dodge") +
  coord_flip()

# 3) 노년픙 비율이 높은 순으로 막대 정렬 
list_order_old <- region_ageg %>% 
  filter(ageg == "Old") %>% 
  arrange(percent)
list_order_old
# 파생변수(지역명 순서 변수) 만들기
order <- list_order_old$region
order

ggplot(data = region_ageg, aes(x = region, y = percent, fill = ageg)) +
  geom_col(position = "dodge") +
  coord_flip() + 
  scale_x_discrete(limits = order)

# 4) 연령대 순으로 막대 색깔 나열하기

class(region_ageg$ageg)
levels(region_ageg$ageg)  # char여서 null로 나옴 (level이 없는것 )
# character를 factor로 바꿔주면 level생성 가능
region_ageg$ageg <- factor(region_ageg$ageg, level = c("Old", "Middle", "Young"))
class(region_ageg$ageg)
levels(region_ageg$ageg)

ggplot(data = region_ageg, aes(x = region, y = percent, fill = ageg)) +   # fill을 연령대로 하는 거니까 
  geom_col() +        #   geom_col(position = "dodge") 넣으면 한줄로 안나옴, fill= ageg로 알아서 나옴 
  coord_flip() + 
  scale_x_discrete(limits = order)


##### ch 10 텍스트 마이닝 #####

# Text Mining 문자로된 데이터에서 가치있는 정보를 얻어 분석 ( sns, 웹사이트 분석) 
# 1) 형태소분석(morphology analysis) : 문장을 구성하는 어절들이 어떤 품사로 되어있는지 파악
# 2) 품사파악, 명사/동사/형용사등 의미를 지닌 품사의 단어들을 추출해 얼마나 많이 등장한지 확인 
# 3) 빈도표 만들고 시각화 

# 텍스트 마이닝 준비 
# 1) 한글 자연어 분석 패키지 : KoNLP, 자바필요 

# install.packages("rJava")
# install.packages("memoise")
# install.packages("KoNLP")

library(KoNLP)
library(dplyr)

Sys.setenv(JAVA_HOME = "C:/Program Files/Java/jre1.8.0_211/")  # 자바 폴더 경로 설정 
useNIADic()


# 10-1.힙합가사 텍스트 마이닝 
txt <- readLines("data/hiphop.txt")
head(txt)

#install.packages("stringr")
library(stringr)

# 특수문자 제거
txt <-str_replace_all(txt, "\\W"," ")   #공백으로 변환!  
# \\W: 정규 표현식(Regular Expression) : 문장의 내용중 이메일주소,전화번호 처럼 특정한 규칙으로 된 부분 추출가능
# str_replace_all(string, pattern, replace)  -> stringr 패키지 

# <가장 많이 사용된 단어 알아보기> 
extractNoun("대한민국의 영토는 한반도와 그 부속도서로 한다")

# 1. 명사 추출 &  df로 변환하고 변수명 수정
# 1)가사에서 명사 추출
nouns <- extractNoun(txt)
head(nouns)
# 2)추출한 명사 리스트를 문자열 벡터로 변환, 단어별 빈도표 생성    # list 아니고 벡터로 만들! df로 만들기 위해
wordcount <- table(unlist(nouns))   #table: 빈도수 계산 
wordcount
# 3) 데이터 프레임으로 변환    #문자열이야! factor아님! 
head(wordcount)
df_word <- as.data.frame(wordcount, stringsAsFactors = F)
# 4)변수명 수정 
df_word <- rename(df_word,
                  word = Var1,
                  freq = Freq)
# 2. 자주 사용된 단어 빈도표 만들기
df_word <-filter(df_word, nchar(word) >=2)   # nchar(): 두 글자 이상 단어 추출   nchar(var1) 산술연산자 a

# ?nchar
top_20 <- df_word %>% 
  arrange(desc(freq)) %>% 
  head(20)
top_20

# as.로 시작하는 함수는 변수의 타입을 바꿈
# as.data.frame() , as.Date(), as.factor() ...

#sapply(df, func()) : 함수는 모든 행에 함수를 적용할때 사용


# <워드 클라우드 만들기> : 단어의 빈도를 구름모양으로 표현 
#install.packages("wordcloud")

library(wordcloud)
library(RColorBrewer)

pal <- brewer.pal(8,"Dark2")

?brewer

set.seed(1234)   
#난수(random variable 무작위로 생성) 고정하기 
# wordcloud()는 실행할 때마다 난수를 이용해 다른모양으로 만들어냄! 즉 실행전에 고정하기! 고정안하면 실행마다 색이 변함 

wordcloud(words = df_word$word,  # 단어선택
          freq = df_word$freq,   # 빈도
          min.freq = 2,          # 최소 단어 수
          max.freq = 200,        # 표현 단어 수
          random.order = F,      # 고빈도 단어 중앙 배치  
          rot.per = .1,          # 회전 단어 비율 (90도 회전)  회전되는 단어의 빈도
          scale = c(4, 0.3),      # 단어 크기 범위 
          colors = pal)          # 색상
# random.color=T : 실행시마다 단어의 색 변화하도록 함          
# minSize 시각화할 최소 빈도 수 설정
# size 배수 기준 워드클라우드 크기 변경
# ratateRation 회전율
# backgroundColor 배경색
# figPath 이미지 

# ?wordcloud  help이용 

# 단어 색상 바꾸기
pal <- brewer.pal(9, "Blues") [5:9]  # 색상 목록 생성 
set.seed(1234)

wordcloud(words = df_word$word,
          freq = df_word$freq,
          min.freq = 2,
          max.words = 200,
          random.order = T,
          rot.per = .1,
          scale = c(4, 0.3),
          colors = pal)



display.brewer.all(n=10, exact.n=FALSE)

brewer.pal.info
brewer.pal.info["Blues",]
brewer.pal.info["Blues",]$maxcolors



# 10-2. 국정원 트윗 텍스트 마이닝

# 1) 데이터 로드, 변수명수정, 특문제거 
twitter <- read.csv("data/twitter.csv", 
                 header = T, 
                 stringsAsFactors = F,
                 fileEncoding = "UTF-8")
head(twitter)
twit <- twitter
twit <- rename(twit, 
               no = 번호,
               id = 계정이름,
               date = 작성일,
               tw = 내용)
head(twit)

twit$tw <- str_replace_all(twit$tw, "\\W", " ")
head(twit)

# 2) 단어 빈도표 만들기
nouns <- extractNoun(twit$tw)
wordcount <- table(unlist(nouns))
df_word <- as.data.frame(wordcount, stringsAsFactors = F)
df_word <- rename(df_word,
                  word = Var1,
                  freq = Freq)                                                                                                 
df_word <- filter(df_word, nchar(word)>= 2)
top20 <- df_word %>% 
  arrange(desc(freq)) %>% 
  head(20)
top20

# 3) 단어 빈도 막대 그래프 만들기
library(ggplot2)
order <- arrange(top20, freq)$word    # 빈도 순서 변수 생성

ggplot(data= top20, aes(x = word, y = freq)) +
  ylim(0,2500) +
  geom_col() +
  coord_flip() +
  scale_x_discrete(limit = order) +          # 빈도순 막대 정렬
  geom_text(aes(label = freq), hjust = - 0.3) # 빈도 표시 +하면 왼쪽, -하면 오른쪽으로    vjust 하면 세로 -1~0까지 정의가능

# 4) 워드 클라우드 만들기
pal <- brewer.pal(8, "Dark2")
set.seed(1234)

wordcloud(words = df_word$word,
          freq = df_word$freq,
          min.freq = 10,         # 최소단어 빈도
          max.words = 200,       # 표현 단어 수 
          random.order = F,
          rot.per = .1,
          scale = c(6, 0.2),     #폰트 크기 c(max, min)
          colors = pal )

# 4-1) 워드 클라우드 색상 변경
pal <- brewer.pal(9, "Blues")[5:9]
set.seed(1234)

wordcloud(words = df_word$word,
          freq = df_word$freq,
          min.freq = 10,         # 최소단어 빈도
          max.words = 200,       # 표현 단어 수 
          random.order = F,
          rot.per = .1,
          scale = c(6, 0.2),
          colors = pal )

# install.packages("wordcloud2")  - 이것도 찾아보기 
library(wordcloud2)
head(demoFreq)
wordcloud2(demoFreq, size = 1.6, color = "random-dark")  # 괄호안에 설정 더 넣을 수 있음 






# 연습문제 seoul_new활용 
# 0)
getwd()
library(KoNLP)
library(wordcloud)
library(RColorBrewer)
useSejongDic()


# 1) 불러오고 단어 추출 
data1 <- readLines("data/seoul_new.txt")   # 한글 깨지면 저장할때 인코딩을 UTF-8로 하면 됨 
head(data1)
data2 <- sapply(data1, extractNoun, USE.NAMES = F )   #sapply() 여러건의 데이터를 한꺼번에 저장 #F로 주면 명사 추출한것만 나옴 
head(data2,30)

head(unlist(data2),30)
data3 <- unlist(data2)
head(data3,30)

# 2) 불필요한 내용 삭제  df <-  gsub("변경전 글자", "변경후 글자", df)

data3 <- gsub("\\d+", "",data3)
data3 <- gsub("서울시", "",data3)
data3 <- gsub("서울", "",data3)
data3 <- gsub("요청", "",data3)
data3 <- gsub("제안", "",data3)
data3 <- gsub(" ", "",data3)
data3 <- gsub("-", "",data3)
head(data3,30)

# 3) 공백제거된 파일 저장, 불러오기, 데이터 확인 
write(unlist(data3), "data/seoul_2.txt")   # 파일 저장 
data4 <- read.table("data/seoul_2.txt")    # 파일 불러오기
data4
nrow(data4)       # 데이터가 몇 건있는지 조회 (row의 개수)

# 4) 빈도표 만들고 sorting
wordcount <- table(data4)
wordcount
head(sort(wordcount, decreasing = T), 20)

# 5) 한번더 불필요한 내용 삭제 df <-  gsub("변경전 글자", "변경후 글자", df)
data3 <- gsub("OO", "", data3)
data3 <- gsub("개선", "", data3)
data3 <- gsub("문제", "", data3)
data3 <- gsub("관리", "", data3)
data3 <- gsub("민원", "", data3)
data3 <- gsub("이용", "", data3)
data3 <- gsub("관련", "", data3)
data3 <- gsub("시장", "", data3)
data3

# 6) 중간중간 저장 
write(unlist(data3), "data/seoul_3.txt")
data4 <- read.table("data/seoul_3.txt")
head(data4,30)
wordcount <- table(data4)
head(sort(wordcount, decreasing = T), 20)

# 7) 그리기
library(RColorBrewer)
pal <- brewer.pal(9, "Set3")



wordcloud(names(wordcount), 
          freq = wordcount,
          scale = c(5, 1),
          rot.per = 0.25,
          min.freq = 1,
          random.order = F,
          random.color = T,
          colors = pal)
# legend는 범례를 만들어주는 함수
# ?legend
legend(0.3, 1, "서울시 응답소 요청사항 분석", 
       cex = 0.8,
       fill = NA,
       border = NA,
       bg = "white",
       text.col = "red",
       text.font = 2,
       box.col = "red")



# 1-0. 
getwd()
library(readxl) 
library(ggplot2)

data <- read_excel("data/dust.xlsx", col_names = T)
View(data)
head(data)

data1 <-data
data1 %>% 
  filter(!is.na(data1$finedust)) %>% 
  group_by(area) %>% 
  summarise(mean_dust = mean(finedust)) %>% 
  arrange(mean_dust) %>% 
  tail(1)   #영등포구
#head(1)   #마포구

data2 <- data %>% 
  filter(area %in% c("마포구", "영등포구")) %>% 
  group_by(area) %>% 
  summarise(mean_dust = mean(finedust)) %>% 
  arrange(mean_dust)
data2


data3 <- data %>% 
  filter(area == "마포구" | area == "영등포구")
data3

# 1-1.
ggplot(data = data2, aes(x = area, y = mean_dust, fill = area )) +
  geom_col(position = "dodge")
# 1-2.
ggplot(data = data3, aes(x = area, y = finedust )) +
  geom_boxplot()



# 2-0.
getwd()
library(KoNLP)
library(wordcloud)
library(RColorBrewer)
library(dplyr)
library(stringr)
useNIADic()

data1 <- readLines("data/jeju.txt")
data1 <- str_replace_all(data1, "\\W", " ")
nouns <- extractNoun(data1)
head(nouns)

wordcount <- table(unlist(nouns))
df_word <- as.data.frame(wordcount, stringsAsFactors = F)
df_word <- rename(df_word,
                  word = Var1,
                  freq = Freq)                                  

df_word <- filter(df_word, nchar(word)>= 2)
top30 <- df_word %>% 
  arrange(desc(freq)) %>% 
  head(30)
top30

# 2-1.

library(ggplot2)
order <- arrange(top30, freq)$word  

ggplot(data= top30, aes(x = word, y = freq, fill = word)) +
  ylim(0,100) +
  geom_col() +
  coord_flip() +
  scale_x_discrete(limit = order) +          
  geom_text(aes(label = freq), hjust = - 0.3) 


pal <- brewer.pal(9, "Blues")[5:9]
set.seed(1234)

# 2-2.

wordcloud(words = df_word$word,
          freq = df_word$freq,
          min.freq = 10,         
          max.words = 200,       
          random.order = F,
          rot.per = .1,
          scale = c(6, 0.1),
          colors = pal )


library(wordcloud2)
wordcloud2(df_word, size = 1.6, color = "random-dark")





##### ch 11 지도 시각화 #####

# remove.packages(c("package1","package2",...)) 한번에 패키지 지우기
# install.packages(c("package1","package2",...)) 한번에 다운 


# 11-1.미국주별 범죄율 단계 구분도 만들기
# 단계 구분도(Choropleth Map) : 지역별 통계치를 색깔차이로 표현함 
# install.packages("ggiraphExtra")
library(ggiraphExtra)

str(USArrests)  # USArrests : r내장 데이터
head(USArrests)

library(tibble)   # dplyr설치하면 자동으로 생성됨, 지역명 변수가 없어서 새로운 df로

crime <- rownames_to_column(USArrests, var = "state")
crime$state <- tolower(crime$state)  #state의 데이터를 소문자로 변환 

str(crime)

?tolower

# 미국 지도 데이터 준비(위경도 포함된 지도 )
library(ggplot2)
# install.packages("maps")  r내장인데 없.. -> 위도, 경도 정보가 있는 데이터 있음 state 미국 주별
library(maps)
states_map <- map_data("state")   # df형태로 불러옴 (ggplot2의 함수)
str(states_map)

# 단계 구분도 만들기
# ggChoropleth(data = df, aes(fill = var1, mpa_id = 표현할 기준 변수), map = 배경에 쓸 지도)
# install.packages("mapproj") 얘도 필요 
?ggChoropleth
library(mapproj)
ggChoropleth(data= crime,                  # 지도에 표현할 데이터
             aes(fill = Murder, map_id = state), # fill 색깔로 표현할 변수 , 지역기준 변수
             map = states_map)             # 지도 데이터 

# 인터랙티브 단계 구분도 만들기 (interactive = T로 설정하면 마우스 우직임에 반응함)
ggChoropleth(data= crime,                
             aes(fill = Murder, map_id = state),
             map = states_map,
             interactive = T) 
# viewer - export - save as Web page -> html 포맷으로 저장가능 확대/축소도 됨 
?interactive

library(RColorBrewer)
ggChoropleth(data = crime,
             aes(fill = c(Murder,Assault), map_id = state),    # 두개 하면 두개 나옴 
             map = states_map,
             color = "grey50",   # default값 
             palette="OrRd",     # default값     
             title="US",
             interactive = T)


# 11-2.대한민국 시도별 인구, 결핵 환자수 단계 구분도 만들기 


# install.packages("stringi")
# install.packages("devtools")
# devtools::install_github("cardiomoon/kormaps2014")  # 깃허브에 공유된 패키지 다운
library(kormaps2014)

# korpop1 2015시도별 데이터, 2시군구별, 3읍면동별 

str(changeCode(korpop1))   # korpop utf-8로 인코딩 되어있어서 changeCode하고 str하면 안깨짐 

library(dplyr)
korpop1 <- rename(korpop1,
                  pop = 총인구_명,
                  name = 행정구역별_읍면동)

str(changeCode(kormap1))    # kormap1에 위경도 정보 있음 

ggChoropleth(data= korpop1,
             aes(fill = pop,
                 map_id = code,     # 코드 지정했는데 (밑)
                 tooltip = name),   # 코드 대신 지역명이 표시되도록! 
             map = kormap1,
             interactive = T)
# 단계 구분도 한글이 깨질 경우
# options(encoding = "UTF=8") 실행하기  R재시작하면 사라짐 
# 이후 CP949인코딩 방식으로 저장된 한글 파일 이용시 options(encoding = "CP949") 원래대로 사용 

str(changeCode(tbc))

ggChoropleth(data = tbc,
             aes (fill = NewPts,
                  map_id = code,
                  tooltips = name),
             map = kormap1,
             interactive = T)


# 실습 : 구글지도 활용하는 ggmap 패키지

# install.packages("ggmap")
# 서울의 위치정보 가져온후 gg_seoul 변수에 할당   -> 지금은 구글에서 막음! 안됨! api필요 
library(devtools)
library(ggmap) 

register_google(key = 'AIzaSyBL5rflS0gXwb5MAgi441UT_of6WCwXppU') # GCP참조 

gg_seoul_1 <- get_googlemap("seoul", maptype = "terrain")   # 지역 가져오기 
ggmap(gg_seoul_1)

gg_seoul_2 <- get_googlemap("seoul", zoom = 6,  maptype = "hybrid") 
ggmap(gg_seoul_2)

gg_seoul <- get_googlemap("seoul", zoom = 6,  maptype = "roadmap") 
ggmap(gg_seoul)

?ggmap

# get_googlemap 의 함수
# center :지도의 중심 좌표
# maptype = terrain, satellite, roadmap, hybrid 중 하나 
# zoom :3 대륙, 21 빌딩 (기본값은 10 도시)
# size : 기본값 640*640

# geocode() : 위도와 경도 정보를 저장 
# geom_point() : 점으로 표시! 밑에 나옴
# ggmap(df) : 실제 지도를 그려줌  

# 지도상에 공영주차장 위치 표시하기 

getwd()
library(ggmap)
library(stringr)

loc <- read.csv("data/서울_강동구_공영주차장_위경도.csv", header =T)
head(loc)

kd <- get_map("Amsa-dong", zoom = 13, maptype = "roadmap")
kor.map <- ggmap(kd) +geom_point(data= loc, aes(x= LON, y=LAT), size = 3, alpha = 0.7, color="red")  #alpha 연해짐 
kor.map + geom_text(data = loc, aes(x = LON, y = LAT + 0.001, label = 주차장명), size = 3)
ggsave("data/kd.png", dpi=500)    # 지도 저장 


# 지도상에 도서관 위치 표시 

lib <- read.csv("data/지역별장애인도서관정보.csv", header = T)
head(lib)

kor <- get_map("seoul", zoom = 11, maptype = "roadmap")  # 지도 가져오기 

kor.map <- ggmap(kor) +geom_point(data= lib, aes(x= LON, y=LAT), size = 3, alpha = 0.7, color="red")
kor.map  + geom_text(data = lib, aes(x= LON, y= LAT+0.01, label = 자치구명), size =3)
ggsave("data/lib_loc.png", dpi = 500)

# 지도상 지역별 인구 표시 
?grid

library(grid)
pop <- read.csv ("data/지역별인구현황_2014_4월기준.csv", header = T)
head(pop)

lon <- pop$LON
lat <- pop$LAT
data <- pop$총인구수
df <- data.frame(lon,lat,data)
df

map1 <- get_map("Jeonju", zoom = 7, maptype = "roadmap") # 서울하면 제주 사라짐! 
map1 <- ggmap(map1)
map1
map1 + geom_point(aes(x = lon, y = lat, colour = data, size = data), data = df)
ggsave("data/pop.png", scale=1, width = 7, height = 4, dpi = 1000)



# 2호선 위치정보
library(ggplot2)
library(ggmap)
sub2 <- read.csv ("data/서울지하철2호선위경도정보.csv", header = T)
head(sub2)
center <- c(mean(sub2$LON), mean(sub2$LAT))
kor <- get_map(center, zoom =11, maptype = "roadmap")
kor.map <- ggmap(kor) + geom_point(data = sub2, aes(x = LON, y = LAT), size = 3, alpha = 0.7)
kor.map +geom_text(data = sub2, aes(x = LON, y = LAT+0.005, label = 역명), size = 3)
ggsave("data/line2.png", dpi = 500)


# 2개의 라인 같이 표시하기 2,3호선
sub2 <- read.csv ("data/서울지하철2호선위경도정보.csv", header = T)
sub3 <- read.csv ("data/서울지하철3호선역위경도정보.csv", header = T)
center <- c(mean(sub2$LON), mean(sub2$LAT))

kor <-get_map(center, zoom = 11, maptype = "roadmap")
kor.map <- ggmap(kor) + geom_point(data= sub2, aes(x= LON, y= LAT), size = 3, color = "green") +
  geom_point(data= sub3, aes(x= LON, y= LAT), size = 3, color = "red" )

kor.map +geom_text(data= sub2, aes(x = LON, y = LAT+0.005, label = 역명), size = 3) +
  geom_text(data= sub3, aes(x = LON, y = LAT+0.005, label = 역명), size = 3)


# leaflet
install.packages("leaflet")
library(leaflet)

m <- leaflet() %>% 
  addTiles() %>%    # add default OpenStreetMap map tiles
  addMarkers(lng = 174.768, lat = -36.852,
             popup = "The birthplace of R")
m  # print the map 






library(ggplot2)
library(ggmap)
library(dplyr)
register_google(key = 'AIzaSyBL5rflS0gXwb5MAgi441UT_of6WCwXppU')

food <- read.csv("C:/Work_r/data/강원도으뜸음식점.csv",header = T )
head(food)

food1 <- rename(food, 
                name = 업소명,
                lon = 경도,
                lat = 위도)

head(food1)
# View(food1)
str(food1)

center <- c(mean(food1$lon), mean(food1$lat))
kor <- get_map(center, zoom = 9, maptype = "roadmap")
ggmap(kor)

kor.map <- ggmap(kor) + geom_point(data= food1, aes(x = lon, y = lat), size = 3, alpha = 0.7, color = "blue" )
kor.map + geom_text(data = food1, aes(x = lon, y = lat+0.005, label = name), size = 3)

library(leaflet)

?leaflet

name <- food1$name
name
lon <- food1$lon
lon
lat <- food1$lat
lat

df <- data.frame(name,lon,lat)
df

df_1 <- leaflet() %>% 
  addTiles() %>% 
  addMarkers(lng = food1$lon, lat= food1$lat, data= df, popup = name)
df_1






##### ch 12 인터랙티브 그래프 #####

# 12-1. plotly 패키지로 인터랙티브 그래프 만들기
# interactive graph : 마우스 움직임에 반응하여 실시간으로 형태가 변하는 그래프, html포맷으로 저장하면 웹을 이용해 그래프 조작 가능

# 인터랙티브 그래프 그리기 
# install.packages("plotly")
library(plotly)

library(ggplot2)
p <- ggplot(data = mpg, aes(x = displ, y = hwy, col = drv)) +
  geom_point()
p

str(mpg)
ggplotly(p)   # 커서 올리면 값 나타남, 드래그하면 확대가능, 더블클릭 원래크기로


# viewer 창 - export - save as web pages 
# 인터랙티브 막대 그래프 만들기 
b <- ggplot(data = diamonds, aes(x = cut, fill = clarity)) +
  geom_bar(position = "dodge")  # geom_bar할거니까 x만 지정해주면 됨 
ggplotly(b)

?ggplotly



# 연습문제 
score <- read.csv("data/학생별과목별성적_국영수_new.csv", header = T)

str(score)
head(score)
View(score)

a <- ggplot(data = score, aes(x = 이름, y = 점수, col = 과목)) +
  geom_point()
ggplotly(a)


b <- score
b <- ggplot(data= b, aes(x = 이름, y = 점수, fill = 과목)) +
  geom_col(position = "dodge")+
  coord_flip()
ggplotly(b)


c <- score %>% 
  group_by(이름) %>% 
  summarise(mean = mean(점수)) %>% 
  arrange(desc(mean)) 
c
View(c)
c1 <- ggplot(data = c, aes(x = 이름, y = mean, fill = 이름)) +
  geom_col(position = "dodge")
ggplotly(c1)



# 12-2.dygraphs 패키지로 인터랙티브 시계열 그래프 그리기
# interactive time series graph

# install.packages("dygraphs")
library(dygraphs)

eco <- ggplot2::economics
head(eco)

# 시계열 데이터가 시간순서의 속성을 지니는 xts데이터 타입이어야 함 !  
# xts(df$var, order.by=시간관련 var) 사용! 
?xts
library(xts)

# 실업률 시계열 인터랙티브
eco1 <- xts(eco$unemploy, order.by = eco$date)   #unemploy를 date를 활용해 xts 데이터로 변경! 
head(eco1)
dygraph(eco1) 

#날짜 범위 선택기능 추가하기(그래프 아래에 새로 추가)
dygraph(eco1) %>% 
  dyRangeSelector()


# 인구수 시계열 인터랙티브
eco3  <- xts(eco$pop, order.by = eco$date)
head(eco3)
dygraph(eco3) %>% 
  dyRangeSelector()


# 여러값 표현하기

#저축률
eco_a <- xts(eco$psavert, order.by = eco$date) 
#실업자수 (psavert가 100만명 단위, unemploy는 천명단위, 즉 나눠서 100만명 단위로 수정  )
eco_b <- xts(eco$unemploy/1000, order.by = eco$date)

eco2 <- cbind(eco_a, eco_b)  # 데이터 결합
head(eco2)
colnames(eco2) <- c("psavert", "unemploy")  
# eco_a, eco_b라고 되어있는걸 바꿔줌 / xts타입이라 rename안됨 ->  colnames() 로 변수명 수정
head(eco2)
dygraph(eco2) %>% 
  dyRangeSelector()

getwd()

# 연습문제 항공사
plane <- read.csv("data/international-airline-passengers.csv", header = T)
head(plane)
View(plane)
p <- plane
p1 <- rename (p, 
              month = Month,
              count = International.airline.passengers..monthly.totals.in.thousands..Jan.49...Dec.60 )

p1 <- p1 %>% 
  filter(!is.na(count))
tail(p1)

p1$month <- paste(p1$month, "-01")
p1$month <- str_remove_all(p1$month," ")

head(p1)
str(p1)
p2 <- xts(p1$count, order.by = p1$month)



# or 카페 참조
# install.packages("anytime")
# 카페 참조
 






##### ch 13 통계 분석 기법을 이용한 가설검정 #####
# 13-1. 통계분석 
# 통계 분석 
# 1)기술통계 : 데이터 요약, 설명
# 2)추론통계 : 우연히 발생할 확률 계산 - 우연히 나타날 확률이 크면 유의하지않다.

# 통계적 가설검정 

# 1.가설설정 : 연구가설 제시, 귀무가설로 검정
# 귀무가설 H0: 두 변수간의 관계가 없다/ 차이가 없다. 부정적 형태 진술 
# 대립(연구)가설 H1: 차이가 있다/ 효과가 있다. 긍정적 형태 진술 - 논문에서 제시
#  예) H1: 신약이 효과가 있다. p-value =0.03  100마리 실험 
# 사회과학: 임계값 a=0.05(p<0.05, 5%미만,96마리 이상 효과) -> 귀무가설 기각 
# 의/생명: 임계값 a=0.01(99마리 이상 효과 필요) ->귀무가설 채택 

# 2. 임계값(a,alpha)
# 유의수준: a=0.05(p<0.05)기준!  95%신뢰도 
# a > p : 귀무가설 기각   // a <= p : 귀무가설 채택 
# 예) h0: 효과가 없다. p-value:0.04/ a=0.05 -> 귀무가설 기각 (=효과가 있다)

# 3.툴 선택,  4.데이터 수집,  5. 데이터 코딩(데이터입력, 데이터 전처리)  6. 통계분석 수행  7. 결과분석


# 13-2. t검정 - 두 집단의 평균 비교  t.test()
library(ggplot2)
library(dplyr)

# compact /suv 차의 도시연비 t검정
mpg_diff <- mpg %>% 
  select(class, cty) %>% 
  filter(class %in% c("compact","suv"))
head(mpg_diff)
table(mpg_diff$class)

t.test(data= mpg_diff, cty~class, var.equal = T)      # class에 따른 cty평균차이 
# t.test(data=df, var1~var2, var.equal = T/F) ~ 는 by : var1 by var2 : var2에 따른 var1의 차이 

# Two Sample t-test
# data:  cty by class       
# t = 11.917, df = 107, p-value < 2.2e-16
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#  5.525180 7.730139
# sample estimates:
#  mean in group compact     mean in group suv   # class에 따른 cty평균 
# 20.12766              13.50000 

# p-value : 2.2 * 10**16  < a=0.05   -> 귀무가설 기각
# => cty는 class(자동차종류)에 따라 차이가 있다. (=두 집단의 통계적 차이가 유의하다)


# 일반휘발유와 고급휘발유의 도시연비 t검정
mpg_diff2 <- mpg %>% 
  select(fl, cty) %>% 
  filter(fl %in% c("r", "p"))   # r regular, p premium
table(mpg_diff2$fl)

t.test(data=mpg_diff2, cty~fl, var.equal = T)  
# fl에 따른 cty 평균차이 
# p-value : 0.2875 > 0.05 두집단의 차이가 통계적으로 유의하지 않다. ( fl에 따른 차이가 없다. 우연히 28.75%나 발생)
# 평균값도 p: 17.36 / r: 16.73 많이 차이 없 


# 13-3. 상관분석 - 두 변수의 관계성 분석 
# 두 연속변수가 서로 관련이 있는지 검정 / 상관계수 (correlation coefficient) -1<cor<1 |1|에 가까울수록 관련성이 높 

# 실업자 수와 개인 소비 지출pce 의 상관관계
eco <- as.data.frame(ggplot2::economics)
cor.test(eco$unemploy, eco$pce)

# Pearson's product-moment correlation

# data:  eco$unemploy and eco$pce
# t = 18.63, df = 572, p-value < 2.2e-16
# alternative hypothesis: true correlation is not equal to 0
# 95 percent confidence interval:
#  0.5608868 0.6630124
# sample estimates:
#      cor 
# 0.6145176 

# p-value < 0.05 : 두개의 변수의 상관이 통계적으로 유의하다. cor=0.6145 -> 정비례관계


# 상관행렬 히트맵 만들기
# 상관행렬(correlation matrix) : 모든 변수의 상관관계를 나타낼때 cor(df)
# 상관행렬 히트맵 : corrplot(상관행렬)
# bit.ly/2r2h2x   참조하기! 


# mtcars 데이터를 이용 (32종 자동차의 11개 속성)

head(mtcars)

car_cor <- cor(mtcars)
car_cor
round(car_cor,2)

# install.packages("corrplot")
library(corrplot)
corrplot(car_cor)  # 양수: 파란색, 음수:빨간색, 상관계수 크면 원의 크기 큼 
corrplot(car_cor, method = "number")  # 상관계수 원대신 숫자로 표현 
# method = number, pie ... 많음 
?colorRampPalette
?corrplot


col<- colorRampPalette(c("#BB4444","#EE9988" ,"#FFFFFF", "#77AADD", "#4477AA"))
corrplot(car_cor,
         method = "color",     # 색깔로 표현 
         col = col(200),       # 색 200개 선정
         type = "lower",       # 왼쪽아래만
         order = "hclust",     # 유사한 상관계수끼리 군집화
         addCodf.col ="black", # 상관계수 색깔
         tl.col = "black",     # 변수명 색
         tl.srt = 45,          # 변수명 45도 기울임
         diag =F)              # 대각행렬 제외 


library(RColorBrewer)
col1<- brewer.pal(6,"Greens")
# col1    #  숫자로 볼수 있음
pal <- colorRampPalette(col1)
corrplot(car_cor,
         method = "color",
         col = pal(20),
         type = "lower",
         order = "hclust",
         addCodf.col ="green",
         tl.col = "black",
         tl.srt = 45,
         diag =F)

# 사이트 참조 (위에보다 쉽게!) 
corrplot(car_cor, type="lower", order="hclust", 
         col=brewer.pal(n=8, name="RdYlBu"))



# 연습문제 

library(readxl)
library(dplyr)
library(ggplot2)
d <- read_excel("data/dust_2018.xlsx", col_names = T)
head(d)
# View(d)


d1 <- d %>% 
  filter(area %in% c("구로구", "강남구"))
head(d1)
d1<- d1 %>% 
  filter(!is.na(finedust))
table(is.na(d1$finedust))

boxplot(d1$finedust)$stats


ggplot(data= d1, aes(x = area, y = finedust)) +
  geom_boxplot()

table(is.na(d1$finedust))

a <- d1 %>% 
  filter (area == "구로구")
head(a)
a<- data.frame(a)
head(a)
a1  <- a %>% 
  mutate(dust = ifelse(finedust <5, mean(finedust),
                       ifelse(finedust > 93, mean(finedust), finedust)))

head(a1)
boxplot(a1$dust)$stats


b <- d1 %>% 
  filter (area == "강남구")
head(b)
b <- data.frame(b)
head(b)
# View(b)

b1 <- b %>% 
  mutate(dust = ifelse(finedust < 3, mean(finedust),
                       ifelse(finedust > 85, mean(finedust), finedust)))

head(b1)  
boxplot(b1$dust)$stats

# View(b1)

total <- bind_rows(a1,b1)
head(total)

ggplot(data= total, aes(x = area, y = dust , fill = area)) + 
  geom_boxplot()



# 2. 결측치 제거, 이상치 평균값 대체한 df 

t.test(data= total, finedust~area, var.equal = T)

# p-value = 3.598e-05  < a=0.05 이므로 귀무가설 기각 
# -> (H0: 두 집단의 차이가 없다.)
# 강남구와 구로구 집단의 통계적 차이가 유의하다 
# -> (지역에 따라 차이가 있다 강남구 평균: 34.94 / 구로구 평균 : 42.50)







##### ch 14 r markdown 으로 데이터 분석 보고서 만들기 #####
# 재현성이 있는 신뢰있는 문서 만들기 (동일한 분석결과가 나오도록)

# file - new file - r markdown 
# 포맷선택(html, pdf, word) 
# 뜨개질모양(knit) rmd(r markdown)파일과 포맷선택한 파일로 둘다 저장됨
# 그 옆에 톱니모양 - preview in viewer pane 
# 특수문자 활용해서 문서양식 결정  책 313~쪽 참조 /reference폴더에 sheet확인 




##### ch 15 r 내장함수, 변수 타입과 데이터 구조 #####


