# 출처 : http://www.mysqltutorial.org/
# error code 1175 해결방법 
SET SQL_SAFE_UPDATES = 0;  # or preference 에서 safe update ~ 라고 써있는거 해체하고 재접


/* 샘플 데이터 가져오기 
 mysql client server(prompt) 에서
 
 mysql -u root -p (enter the password)
 source c:/temp/mysqlsampledatabase.sql
 show databases; 
 */

USE classicmodels;
/* 
SELECT 	 3
FROM	 1
WHERE 	 2
GROUP BY 4
HAVING	 5
ORDER BY 6
LIMIT 	 7

*/


/* SELECT */ 
SELECT *
	FROM customers;

# SELECT select_list
#	FROM table_name;

SELECT lastname
	FROM employees;
    
SELECT lastname, firstname, jobtitle
	FROM employees;
    
SELECT *
FROM employees;

/* ORDER BY 
# SELECT select_list
# FROM table_name
# ORDER BY column1 [ase|desc] ... ;  default=asc
*/

select contactLastname, contactFirstname
from customers
order by contactLastname;

select contactLastname, contactFirstname
from customers
order by contactLastname desc;

select contactLastname, contactFirstname
from customers
order by contactLastname desc, contactFirstname asc ;
	
select orderNumber, orderlinenumber, quantityOrdered*priceEach
from orderdetails
order by quantityOrdered*priceEach desc ;

select 
	orderNumber, 
    orderlinenumber, 
    quantityOrdered*priceEach as subtotal
from orderdetails
order by subtotal desc ;

# 원하는 결과(status)에 따라서 정렬
select ordernumber, status
from orders
order by FIELD(status, 
'In process',
'On Hold',
'Cancelled',
'Resolved',
'Disputed',
'Shipped');
 
/* WHERE 
select select_list
FROM table_name
where search_condition;	  <- AND / OR / NOT

*/ 

SELECT lastname, firstname, jobtitle, officeCode
FROM employees
WHERE jobtitle = 'Sales Rep' AND # AND : 둘다 만족 
	officeCode = 1;

SELECT lastname, firstname, jobtitle, officeCode
FROM employees
WHERE jobtitle = 'Sales Rep' OR	 # OR: 둘중 하나 만족하면 print
	officeCode =1
ORDER BY officeCode ASC, jobtitle ;

# WHERE value BETWEEN low AND high
SELECT firstname, lastname, officeCode
FROM employees
WHERE officeCode BETWEEN 1 AND 3 
ORDER BY officeCode;

# WHERE value LIKE  ' _ % ' 밑에 더 설명함 
SELECT firstname, lastname
FROM employees
WHERE lastname LIKE '%son' # % :0개 이상 일치할때    
ORDER BY firstname;

SELECT firstname, lastname
FROM employees
WHERE lastname LIKE '_%s'    #_ : 단일문자 자리에 씀!
ORDER BY firstname;


SELECT firstname, lastname
FROM employees
WHERE lastname LIKE 'J_n%'   
ORDER BY firstname;

# WHERE value IN (value1, value2,..)
SELECT firstname, lastname, officecode
FROM employees
WHERE officecode IN (1,2,3)
ORDER BY officecode DESC;

# WHERE valus IS NULL  # null뽑기 
SELECT lastname, firstname, reportsto
FROM employees
WHERE reportsto IS NULL;

/* WHERE + 비교 연산자 
= 
!= (or <>) 
< 일반적으로 숫자, 날짜, 시간데이터와 사용
>
<=, >=
*/ 

SELECT lastname, firstname, jobtitle
FROM employees
WHERE jobtitle != 'Sales Rep';

SELECT lastname, firstname, officecode
FROM employees
WHERE officecode > 5 OR 
	officecode <= 4 
ORDER BY officecode DESC;


/* SELECT DISTINCT : 중복 행 제거 
SELECT DISTINCT select_list
FROM table_name; 
*/

SELECT lastname 
FROM employees
ORDER BY lastname;
# -> 중복 있음

SELECT DISTINCT lastname
FROM employees
ORDER BY lastname;

# 다 나옴 
SELECT state
FROM customers;
# 중복제거 하고 다 표시 null도 있음 
SELECT DISTINCT state
FROM customers;

# state - city 의 중복 조합 제공 
SELECT  state, city
FROM customers
WHERE state IS NOT NULL
ORDER BY state, city;
# 고유한 state - city 의 조합 제공 
SELECT DISTINCT state, city
FROM customers
WHERE state IS NOT NULL
ORDER BY state, city;

# DISCTINCT + GROUP BY
SELECT state
FROM customers
GROUP BY state;
# 위와 비슷한 결과 나타냄 
SELECT DISTINCT state
FROM customers;

SELECT DISTINCT state
FROM customers
ORDER BY state;

# DISTINCT + 집계함수 (SUM, AVG, COUNT..)
SELECT COUNT(DISTINCT state)
FROM customers
WHERE country = 'usa';

# DISTINCT + LIMIT : 고유한 행 수를 찾으면 즉시 검색 중지 
SELECT DISTINCT state
FROM customers
WHERE state IS NOT NULL
LIMIT 5;
 
 
 /*  연산자
 # AND 
 bool_expression1 AND bool_expression

		TRUE	FALSE	 NULL
 TRUE	true	false	 null
 FALSE  false	false	 false
 NULL	null	false	 null
 
 AND : (SELECT, UPDATE, DELETE)가 있는 WHERE절에서 많이 사용 / INNER JOIN, LEFT JOIN
 
 */
 
 SELECT 1 = 0 AND 1/0;  # >>> 0 : False   / 1 : True 
 
 SELECT customername, country, state
 FROM customers
 WHERE country ='usa' and state ='ca';
 
 SELECT customername, country, state, creditlimit
 FROM customers
 WHERE country='usa' 
	AND state='ca'
    AND creditlimit >= 100000;
    
    
# OR
SELECT 1 = 1 OR 1/0 ; # >>> 1 뒤에 평가 안해 어짜피
# AND 를 먼저 시행, 그다음 OR
SELECT true OR false AND false; #>>> 1  
# 순서를 바꿔서 시행하려면
SELECT (true OR false) AND false;  #>>> 0 

SELECT customername, country
FROM customers
WHERE country ='usa' 
	OR country = 'france';
    
SELECT customername, country, creditlimit
FROM customers
WHERE (country='usa' OR country='france') 
	AND creditlimit >= 10000;
    
# and 먼저 그다음 or
SELECT customername, country, creditlimit
FROM customers
WHERE country='usa' 
	OR country='france' 
	AND creditlimit >= 10000;

# IN ( OR과 비슷) 
/* SELECT column1, columnn2,...
FROM table_name
WHERE (expr|column_1) IN ('value1','value2',...);
*/

SELECT officecode, city, phone, country
FROM offices
WHERE country IN ('usa','france');
# OR을 사용한경우 (길어짐) 
SELECT officecode, city, phone, country
FROM offices
WHERE country='usa' OR country='france';

# 미국과 프랑스가 아닌 곳을 찾기
SELECT officecode, city, phone
FROM offices
WHERE country NOT IN ('usa','france');

# 하위 쿼리 to find the orders whose total values are greater than 60,000
SELECT ordernumber, customernumber, status, shippeddate
FROM orders
WHERE ordernumber
	IN (SELECT ordernumber 
		FROM orderdetails
        GROUP BY ordernumber
        HAVING SUM(quantityordered * priceeach) > 60000) ;

# BETWEEN 
/* expr [NOT] BETWEEN start_expr AND end_expr; 
expr , start_expr, end_expr은 데이터 형식이 같아야함 
*/

SELECT productcode, productname, buyprice
FROM products
WHERE buyprice BETWEEN 90 AND 100;
# WHERE buyprice >= 90 AND buyprice <= 100 ; 과 동일

SELECT productcode, productname, buyprice
FROM products
WHERE buyprice NOT BETWEEN 20 AND 100;
# WHERE buyPrice < 20 OR buyPrice > 100; 과 동일 

# BETWEEN 날짜 값과 연산자를 사용할 때는 CAST() 함수를 사용해서 DATE로 명시적으로 바꾼다음에 해야함
SELECT ordernumber, requireddate, status
FROM orders
WHERE requireddate  
	BETWEEN CAST('2003-01-01' AS DATE) AND CAST('2003-02-01' AS DATE) ;


# LIKE 
/* expression LIKE pattern ESCAPE escape_character 
(select, delete, update)를 포함한 where 절에서 많이 사용 for filtering 
% : 0개이상의 문자열과 일치	s% : sun, six, super ...
_ : 단일 문자와 일치 		se_ : sea, see, seven...
*/

SELECT employeenumber, lastname, firstname
FROM employees
WHERE firstname LIKE 'a%';

SELECT employeenumber, lastname, firstname
FROM employees
WHERE lastname LIKE '%on';

SELECT employeenumber, lastname, firstname
FROM employees
WHERE lastname LIKE '%on%'; # 중간에 들어간것도 포함할경우 

SELECT employeenumber, lastname, firstname
FROM employees
WHERE firstname LIKE 'T_m';

SELECT employeenumber, lastname, firstname
FROM employees
WHERE lastname NOT LIKE 'B%';  # B로 시작하지 않는 사람 검색 / 대소문자 구분 안함! 

# % 나 _가 포함된 문자열이 있을 경우 \로 escape 
SELECT productcode, productname
FROM products
WHERE productcode LIKE '%\_20%';
#WHERE productcode LIKE '%$_20%' ESCAPE '$'; # 해도 똑같음 

/* LIMIT  - order by와 같이 쓰는걸 추천 
SELECT select_list
FROM table_name
ORDER BY oreder_expression
LIMIT [offset,] row_count;  

offset: 0으로 시작!  시작
row_count=최대행수  끝
둘다 >= 0 

LIMIT row_count;
LIMIT 0, row_count;
LIMIT row_count OFFSET offset  -> PostgreSQL 과 호환성을 위해 제공함 
*/

SELECT customernumber, customername, creditlimit
FROM customers
ORDER BY creditlimit # DESC
LIMIT 5;

SELECT customernumber, customername, creditlimit
FROM customers
ORDER BY creditlimit, customernumber
LIMIT 5;

SELECT COUNT(*) FROM customers;  # 총 행의 갯수 반환 122
SELECT customernumber, customername
FROM customers
ORDER BY customername
#LIMIT 10; # 한페이지에 10개 씩 있다고 한다면 0~9까지(1행~10행)
LIMIT 10,10;  # 10~19번까지  

# NULL
SELECT 1 IS NULL, 0 IS NULL, NULL IS NULL;
SELECT 1 IS NOT NULL;

SELECT customername, country, salesrepemployeenumber
FROM customers
WHERE salesrepemployeenumber IS NOT NULL # IS NULL
ORDER BY customername;

# NULL 특수기능
CREATE TABLE IF NOT EXISTS projects(
	id INT AUTO_INCREMENT,
    title VARCHAR(255),
    begin_date DATE NOT NULL,
    complete_date DATE NOT NULL,
    PRIMARY KEY(id)
    );
INSERT INTO projects(title, begin_date, complete_date)
VALUES('New CRM','2020-01-01','0000-00-00'),
	('ERP Future','2020-01-01','0000-00-00'),
    ('VR','2020-01-01','2030-01-01');

SELECT * FROM projects
WHERE complete_date IS NULL ;

# @@sql_auto_is_null 이 1로 INSERT되면 IS NULL연산자를 사용해서 명령문을 실행한후 생성된 열의 값을 얻을 수 있음 
# default는 0
SET @@sql_auto_is_null=1;
INSERT INTO projects(title, begin_date, complete_date)
VALUES('MRP III','2010-01-01','2020-12-31');

SELECT id
FROM projects
WHERE id IS NULL;  # IS NULL인 index반환 

SELECT *
FROM projects;


# IS NULL 최적화
# null인 행 다 반환
SELECT customernumber, salesrepemployeenumber
FROM customers
WHERE salesrepemployeenumber IS NULL;

# null인 데이터에 관한 설명 반환 EXPLAIN SELECT 
EXPLAIN SELECT customernumber, salesrepemployeenumber
FROM customers
WHERE salesrepemployeenumber IS NULL;

# 조합 최적화도 가능 
EXPLAIN SELECT customernumber, salesrepemployeenumber
FROM customers
WHERE salesrepemployeenumber = 1370 OR
	salesrepemployeenumber IS NULL;

/* Alias 별명 
SELECT [column_1|expression] AS descriptive_name
FROM table_name;

SELECT [column_1|expression] AS 'descriptive name' # 빈칸 하려면 '' 사용
FROM table_name;

*/
# CONCAT_WS() : 연결! 
SELECT CONCAT_WS(', ', lastname, firstname) 
FROM employees;

SELECT CONCAT_WS(', ', lastname, firstname) AS 'Full name'
FROM employees
ORDER BY 'full name';

SELECT ordernumber 'order no.',
	SUM(priceeach*quantityordered) total
FROM orderdetails
GROUP BY 'order no.'
HAVING total > 60000;

# TABLE의 AS 
# table_name as  table_alias

SELECT *
FROM employees e;

# table_alias.column_name 으로 불러올수 있음 
SELECT e.firstname, e.lastname
FROM employees e
ORDER BY e.firstname;

# customernumber 은 2개(customers / orders )에 들어있음 -> ambiguous
# join 간단한 튜토리얼 / 별명 써야 간단해짐 
SELECT customername, COUNT(o.ordernumber) total
FROM customers c
INNER JOIN orders o 
	ON c.customernumber = o.customernumber
GROUP BY customername
ORDER BY total DESC;
 
 
 
 /* JOIN 
 ordernumber 은 orders/orderdetails에 같이 있음
FROM에서 사용 - inner /left/right/cross join (full outer join은 아직 없)
 */

# 우선 table 생성
DROP TABLE members;
DROP TABLE committees;

CREATE TABLE members(
	member_id INT AUTO_INCREMENT,
    name_ VARCHAR(100),
    PRIMARY KEY (member_id)
    );

CREATE TABLE committees (
	committee_id INT AUTO_INCREMENT,
    name_ VARCHAR(100),
    PRIMARY KEY (committee_id)
    );
    
INSERT INTO members(name_)
VALUES ('john'),('Jane'),('Mary'),('David'),('Amelia');

INSERT INTO committees(name_)
VALUES('John'),('Mary'),('Amelia'),('Joe');

SELECT * 
FROM members;

SELECT * 
FROM committees;

/* inner join  겹치는 것만 (가운데만) 
SELECT column_list
FROM table_1
INNER JOIN table_2 ON join_condition;

# 매칭할 col이름이 같으면 USING 사용해도 됨 
SELECT column_list
FROM table_1
INNER JOIN table_2 USING (column_name);

 */
 
 SELECT m.member_id, 
	m.name_ mname, 
	c.committee_id, 
    c.name_  cname
 FROM members m
 INNER JOIN committees c   
	ON c.name_ = m.name_;  # driving = driven 
# INNER JOIN committees c USING(name_); 도 가능


/* LEFT JOIN  : table1 전체 + table2에도 포함되는 정보 포함! 
# only table1이면 table2의 정보는 null로 나옴 

SELECT column_list
FROM table_1
LEFT JOIN table_2 ON join_condition;
LEFT JOIN table_2 USING(column_name); # 공통되는 col_name
[WHERE table2_alias.column IS NULL;] : only table1 만 찾아내기 
*/

# members 전체 (교집합 포함) 
SELECT m.member_id, 
	m.name_ AS m_name, 
    c.committee_id, 
    c.name_ AS c_name
FROM members m
# LEFT JOIN committees c USING(name_);
LEFT JOIN committees c 
	ON c.name_ = m.name_;
    
# only members 만 
SELECT m.member_id, 
	m.name_ AS m_name, 
    c.committee_id, 
    c.name_ AS c_name
FROM members m
LEFT JOIN committees c 
	ON c.name_ = m.name_
WHERE c.committee_id IS NULL;   


/* RIGHT JOIN : table2!  (table1의 정보도나옴) 
SELECT column_list
FROM table_1
RIGHT JOIN table_2 ON join_condition;
RIGHT JOIN tale_2 USING(column_name); 
[WHERE column_table1 IS NULL;]  # only table2! 오직 2번 테이블만 나오게 할때 
*/

# table2 전체(교집합 포함) c 전체
SELECT m.member_id, m.name_ mname,
	c.committee_id, c.name_ cname
FROM members m
#RIGHT JOIN committees c 
#	ON c.name_ = m.name_;
RIGHT JOIN committees c USING (name_);

# only c! 
SELECT m.member_id, m.name_ mname,
	c.committee_id, c.name_ cname
FROM members m
RIGHT JOIN committees c USING (name_)
WHERE m.member_id IS NULL;  # c는 맞지만 m이 아닌경우


/* CROSS JOIN 
SELECT select_list
FROM table_1
CROSS JOIN table_2;  

# 조인 조건절이 없음 table1에 n행, 2에 m행이 있으면 n*m 행 반환
# 계획 데이터를 생성하는데 유용 
*/

SELECT m.member_id, m.name_ mname,
	c.committee_id, c.name_ cname
FROM members m
CROSS JOIN committees c;

/***** JOIN 샘플 예제 
SELECT selected_list
FROM t1
INNER JOIN t2 ON join_condition1
INNER JOIN t3 ON join_condition2
...;

<products table> t1 의 columns 
: productcode, productname, productline, productscale, 
productvendor, productdesciption, quantitylnstock, buyprice, msrp

<productlines table > t2 의 columns
: productline, textdescription, htmldescription, image


*****/


### inner join 
# prodictcode, product name from <products table> -> t1
# textdescription from <productlines table> -> t2

SELECT productcode, productname, textdescription
FROM products t1
INNER JOIN productlines t2 
	ON t1.productline = t2.productline;
# INNER JOIN productlines USING (productline) ; # 기준이 되는 컬럼이 productline

SELECT t1.ordernumber, t1.status, SUM(quantityordered * priceeach) total
FROM orders t1
INNER JOIN orderdetails t2 
	ON t1.ordernumber =t2.ordernumber
# INNDER JOIN orderdetails USING(ordernumber);
GROUP BY ordernumber;

# 3개 결합 t: orders, orderdetails, products
SELECT ordernumber, orderdate, orderlinenumber, productname, quantityordered, priceeach
FROM orders
INNER JOIN orderdetails USING (ordernumber)
INNER JOIN products USING (productcode)
ORDER BY ordernumber, orderlinenumber;

# 4개  t: orders, orderdetails, customer, products
SELECT ordernumber, orderdate, customername, productname, quantityordered, priceeach, (quantityordered *priceeach) AS total
FROM orders
INNER JOIN orderdetails USING (ordernumber)
INNER JOIN products USING (productcode)
INNER JOIN customers USING (customernumber)
ORDER BY ordernumber, orderlinenumber;

# INNER JOIN + 연산자 
SELECT ordernumber, productname, msrp, priceeach, p.productcode
FROM products p
INNER JOIN orderdetails o 
	ON p.productcode = o.productcode 
    AND p.msrp > o.priceeach
WHERE p.productcode ='S10_1678';

### LEFT JOIN 
# 고객은 1개이상 주문가능, 주문은 1개고객에 속해야함
SELECT c.customernumber, customername, ordernumber, status  # 한 테이블에만 있으면 굳이 별명.컬럼 안해도됨 
FROM customers c
#LEFT JOIN orders o
#	ON o.customernumber=c.customernumber;
LEFT JOIN orders USING(customernumber);

# 부모 = 자식 으로 쓰는 습관 들이기 아니면 USING 써! 
SELECT c.customernumber, customername, ordernumber, status    
FROM customers c
LEFT JOIN orders o
	ON c.customernumber=o.customernumber;

# LEFT JOIN + IS NULL 
# 주문이 없는 고객을 찾기! 
SELECT c.customernumber, c.customername, o.ordernumber, o.status
FROM customers c
LEFT JOIN orders o USING(customernumber)
WHERE ordernumber IS NULL ;

# LEFT JOIN 3개  # 여기선 USING 사용 못...? 
SELECT lastname, firstname, customername, checknumber, am ount
FROM employees
LEFT JOIN customers 
	ON employeenumber =salesrepemployeenumber
LEFT JOIN payments
	ON payments.customernumber = customers.customernumber
ORDER BY customername, checknumber;




# WHERE vs ON
SELECT o.ordernumber, customernumber, productcode
FROM orders o
LEFT JOIN orderdetails USING(ordernumber)
WHERE ordernumber =10123;  # 에만 새당하는 4명 출력 

# on을 쓰면 4명 포함하고, 그외 null인 사람도 다~ 
SELECT o.ordernumber, customernumber, productcode
FROM orders o
LEFT JOIN orderdetails d
	ON o.ordernumber =d.ordernumber
    AND o.ordernumber =10123;
    
### RIGHT JOIN
# 영업담당자 : sales rep employee number, 각 고객은 1명의 영업담당자만 가질수 있음 
# 즉 null 값은 영업담당자가 없는 것 
SELECT employeenumber, customernumber
FROM customers
RIGHT JOIN employees ON employeenumber = salesrepemployeenumber
ORDER BY employeenumber;

# null만 뽑아내기 
SELECT employeenumber, customernumber
FROM customers
RIGHT JOIN employees ON employeenumber = salesrepemployeenumber
WHERE customernumber IS NULL
ORDER BY employeenumber;

### CROSS JOIN - on/using 절이 없음 
/* 
SELECT * FROM t1
CROSS JOIN t2
WHERE t1.id = t2.id;
*/

# 새로운 데이터 베이스 만들기
CREATE DATABASE IF NOT EXISTS testdb;
USE testdb;

# DROP TABLE products;

CREATE TABLE products (
    id INT PRIMARY KEY AUTO_INCREMENT,
    product_name VARCHAR(100),
    price DECIMAL(13,2 )
);
 
CREATE TABLE stores (
    id INT PRIMARY KEY AUTO_INCREMENT,
    store_name VARCHAR(100)
);
 
CREATE TABLE sales (
    product_id INT,
    store_id INT,
    quantity DECIMAL(13 , 2 ) NOT NULL,
    sales_date DATE NOT NULL,
    PRIMARY KEY (product_id , store_id),
    FOREIGN KEY (product_id)
        REFERENCES products (id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (store_id)
        REFERENCES stores (id)
        ON DELETE CASCADE ON UPDATE CASCADE
);

INSERT INTO products(product_name, price)
VALUES('iPhone', 699),
      ('iPad',599),
      ('Macbook Pro',1299);
 
INSERT INTO stores(store_name)
VALUES('North'),
      ('South');
 
INSERT INTO sales(store_id,product_id,quantity,sales_date)
VALUES(1,1,20,'2017-01-02'),
      (1,2,15,'2017-01-05'),
      (1,3,25,'2017-01-05'),
      (2,1,30,'2017-01-02'),
      (2,2,35,'2017-01-05');

# 각 상점 및 제품의 총 판매량 반환, 판매를 계산해서 상점별 / 제품별 그룹화
SELECT store_name, product_name, SUM(quantity * price) AS revenue
FROM sales
INNER JOIN products ON products.id =sales.product_id
INNER JOIN stores ON stores.id =sales.store_id
GROUP BY store_name, product_name;

# 특정 제품을 판매하지 않은 상점을 알고 싶을때  ->  cross join 사용해서 셀렉트안에 셀렉트 문으로 집어넣기 
SELECT store_name, product_name
FROM stores AS a 
CROSS JOIN products b ;

#  전체 코드 
SELECT b.store_name, a.product_name, IFNULL(c.revenue, 0) AS revenue  # IFNULL : 매출이 NULL인경우 함수를 사용해서 0반환
FROM products a
CROSS JOIN stores b
LEFT JOIN
    (SELECT stores.id AS store_id, products.id AS product_id,
        store_name, product_name,
        ROUND(SUM(quantity * price), 0) AS revenue
    FROM sales
    INNER JOIN products ON products.id = sales.product_id
    INNER JOIN stores ON stores.id = sales.store_id
    GROUP BY store_name , product_name) AS c ON c.store_id = b.id
        AND c.product_id = a.id
ORDER BY b.store_name;


/* SELF JOIN 
계층적으로 데이터를 쿼리하거나, 같은 테이블내에서 행과 다른행을 비교할때 사용 
단일 조회에서 동일한 테이블이름을 반복하지 않도록 별명 사용해야함! 안그러면 오류~

*/
USE classicmodels;

SELECT lastname, firstname, reportsto, employeenumber
FROM employees;

# SELF INNER  -> 관리자가 있는 사람만 나오니까 맨 위에 사장은 안나옴 

SELECT CONCAT(m.lastname, ' ', m.firstname) AS manager,
	CONCAT(e.lastname, ' ', e.firstname) AS 'direct report',
    e.jobtitle
FROM employees e
INNER JOIN employees m ON m.employeenumber = e.reportsto
ORDER BY manager;

# 그래서 SELF LEFT  #>>> jobtitle : predident 나옴! 
SELECT CONCAT(m.lastname, ' ', m.firstname) AS manager,
	CONCAT(e.lastname, ' ', e.firstname) AS 'direct report',
    e.jobtitle
FROM employees e
LEFT JOIN employees m ON m.employeenumber = e.reportsto
ORDER BY manager ;


# SELF JOIN 으로 연속행 비교 
# 같은 도시에 있는 고객목록 표시하고 싶을 때
SELECT c1.city, c1.customername, c2.customername
FROM customers c1
INNER JOIN customers c2 ON c1.city=c2.city 
AND c1.customername > c2.customername  # 동일한 고객이 포함되지 않도록 정의 
ORDER BY c1.city;


/* GROUP BY -> 그룹화
# 집계 함수(aggregate function) 사용가능 SUM, AVG, MAX, MIN, COUNT 

SELECT c1,c2,...,cn, aggregate_func(ci)
FROM table
WHERE where_conditions
GROUP BY c1, c2,...cn;

*/

SELECT status
FROM orders
GROUP BY status;  # SELECT DISTINCT status를 사용한것과 같은 결과 

# COUNT + GROUP BY
# status별 count 주문수 알아내기 (select 된 column의 count!) 

SELECT status, COUNT(*)  
FROM orders
GROUP BY status;

SELECT employeenumber, COUNT(*)
FROM employees
GROUP BY employeenumber;

# SUM + GROUP BY
SELECT status, SUM(quantityordered * priceeach) AS amount
FROM orders
INNER JOIN orderdetails USING(ordernumber)
GROUP BY status;

# 주문번호과 각 주문의 총액 반환 
SELECT ordernumber, SUM(quantityordered * priceeach)  AS total
FROM orderdetails
GROUP BY ordernumber;

# 매년 총 판매량을 가져오기 (월로 할꺼면 MONTH)
SELECT YEAR(orderdate) AS year, SUM(quantityordered * priceeach) AS total
FROM orders
INNER JOIN orderdetails USING (ordernumber)
WHERE status ='shipped'
GROUP BY YEAR(orderdate);

# GROUP BY + HAVING : 반환된 그룹을 필터링하려면 HAVING ! 
SELECT YEAR(orderdate) AS year, SUM(quantityordered * priceeach) AS total
FROM orders
INNER JOIN orderdetails USING (ordernumber)
WHERE status ='shipped'
GROUP BY YEAR(orderdate)
HAVING year >= 2004;

# GROUP BY는 원래 별명 사용할 수 없지만 mysql에는 됨! 
SELECT MONTH(orderdate) AS month, COUNT(ordernumber)
FROM orders 
GROUP BY month;

SELECT status, COUNT(*)
FROM orders 
GROUP BY status;  # 뒤에 ASC, DESC된다는데 안됨 

/* HAVING

SELECT select_list
FROM table_name
WHERE search_condition
GROUP BY group_by_expression
HAVING group_condition;

그 뒤에 order by, limit 
 */
 
SELECT ordernumber, SUM(quantityordered) AS itemcount,
	SUM(priceeach * quantityordered ) AS total
FROM orderdetails
GROUP BY ordernumber;

SELECT ordernumber, SUM(quantityordered) AS itemcount,
	SUM(priceeach * quantityordered) AS total
FROM orderdetails
GROUP BY ordernumber
# HAVING total > 10000;
HAVING total > 1000 AND itemcount > 600; # AND OR 도 됨 

SELECT a.ordernumber, status, SUM(priceeach * quantityordered) total
FROM orderdetails a
INNER JOIN orders b ON b.ordernumber = a.ordernumber 
GROUP BY ordernumber, status
HAVING status = 'shipped' AND total > 1500;


/* ROLL UP
부분합, 총합 계산 
 */


CREATE TABLE sales
SELECT productline, YEAR(orderdate) orderyear, quantityordered * priceeach ordervalue
FROM orderdetails 
INNER JOIN orders USING(ordernumber)
INNER JOIN products USING(productcode)
GROUP BY productline, YEAR(orderdate);

SELECT * FROM sales;

SELECT productline, SUM(ordervalue) totalordervalue
FROM sales
GROUP BY productline;

# ordervalue 의 합계 
SELECT SUM(ordervalue) totalordervalue
FROM sales;

# 하나의 쿼리에서 둘 시아의 그룹화를 함께 생성하려면 union all 사용 위에는 합계 하나만 나옴 
SELECT productline, SUM(ordervalue) totalordervalue
FROM sales
GROUP BY productline
UNION ALL
SELECT NULL, SUM(ordervalue) totalordervalue  # 마지막에 전체 합계에 productline col에 NULL
FROM sales;

# 위에 너무 길고 복잡해서 ROLLUP 사용 ! 똑같이 나옴! 
/* 
SELECT selected_list
FROM table_name
GROUP BY c1,c2,c3 WITH ROLLUP; 
*/

SELECT productline, SUM(ordervalue) totalordervalue
FROM sales
GROUP BY productline WITH ROLLUP;


/*
# 소계를 나타낼때 (productline 별로 소계를 나타내려 할때)
GROUP BY c1,c2,c3 WITH ROLLUP 일때  
둘이상의 columns이 있다면 계층 구조가 있다고 가정 c1 > c2 > c3  
그리고 나서 그룹화 세트를 생성 
(c1, c2,c3)
(c1, c2)
(c1)
() 

즉 productline 별로 소계 생성

*/
SELECT productline, orderyear, SUM(ordervalue) totalordervalue
FROM sales
GROUP BY productline, orderyear  # productline > orderyear  : productline별로 소계 
WITH ROLLUP;

SELECT productline, orderyear, SUM(ordervalue) totalordervalue
FROM sales
GROUP BY orderyear , productline # 위와 반대, year별로 소계 
WITH ROLLUP;


/* GROUPING () : NULL로 표시되는 소계/총계를 나타내는지 확인하려면 사용 
- 새로운 col이 생김 GROUPING(orderyear) 처럼! 그리고 소계면 1, 그냥 값이면 0으로 반환
- SELECT문의 selected_list와  HAVING/ORDER BY 절에서 사용가능 
*/

SELECT orderyear, productline, SUM(ordervalue) totalordervalue, 
	GROUPING(orderyear), GROUPING(productline)
FROM sales
GROUP BY orderyear, productline
WITH ROLLUP;

# NULL로 뜨는 소계/합계에 이름지어주기 
SELECT IF(GROUPING(orderyear), '<all years>', orderyear) orderyear,
	IF(GROUPING(productline), '<all product lines>', productline) productline,
    SUM(ordervalue) totalordervalue
FROM sales
GROUP BY orderyear, productline
WITH ROLLUP;


/* SUB QUERY 서브쿼리
괄호로 묶어야함! 
Outer Query 안에 sub query 있음(or inner query)

서브쿼리가 실행되고 결과반환된것이 외부쿼리의 입력으로 사용 
*/

USE classicmodels;
SELECT lastname, firstname
FROM employees
WHERE officecode IN (SELECT officecode FROM offices WHERE coun);

# 비교연산자가 포함된 서브쿼리
SELECT MAX(amount) FROM payments;

SELECT customernumber, checknumber, amount
FROM payments
WHERE amount = (SELECT MAX(amount) FROM payments);

SELECT customernumber, checknumber, amount 
FROM payments
WHERE amount > (SELECT AVG(amount) FROM payments);

# IN , NOT IN이 포함된 서브쿼리

# 아직주문하지 않은 고객 명단 
SELECT customername 
FROM customers
WHERE customernumber NOT IN ( SELECT DISTINCT customernumber FROM orders); 

# FROM절의 서브쿼리 
# 서브쿼리의 결과값이 임시테이블로 사용됨 -> 파생테이블 or 구체화된 서브쿼리라고 함  (as a derived table or materialized subquery)
SELECT MAX(items), MIN(items), FLOOR(AVG(items)) # 최대, 최소, 평균 품목의 개수 / FLOOR(): 소수점 이하 자리수 제거 
FROM (SELECT ordernumber, COUNT(ordernumber) AS items
	FROM orderdetails
    GROUP BY ordernumber) AS lineitems;
    
# Correlated subquery : 아우터 쿼리에 영향을 받는 쿼리 
# 내부는 모든 productline에 대해 실행되고, 외부는 그 가격보다 큰것만 필터링해서 반환 
# 평균가격보다 구매가격이 큰 제품만 선택 ! 
SELECT productname, buyprice # , p1.productline
FROM products p1
WHERE buyprice > (SELECT AVG(buyprice) FROM products WHERE productline =p1.productline);


/* EXISTS / NOT EXISTS  뒤에 더 자세히 설명되어있음 
SELECT * 
FROM table_name
WHERE EXISTS (subquery) ;

서브쿼리의 결과 : T or F 반환 
*/

SELECT ordernumber, SUM(priceeach * quantityordered) total
FROM orderdetails INNER JOIN orders USING(ordernumber) 
GROUP BY ordernumber
HAVING SUM(priceeach * quantityordered ) > 60000;  # >>> ordernumber와 total 3개 반환

# 위의 쿼리를 서브쿼리로 넣어서 해당되는 사람들의 customer 번호와 이름 반환 
SELECT customernumber, customername
FROM customers
WHERE EXISTS(SELECT ordernumber, SUM(priceeach * quantityordered) 
				FROM orderdetails
                INNER JOIN orders USING(ordernumber)
                WHERE customernumber = customers.customernumber
                GROUP BY ordernumber
                HAVING SUM(priceeach * quantityordered) > 60000);
                


/* Derived TABLE 파생테이블 -> 단어 확인  
임시테이블과 유사하지만 작성이 필요 없으므로 훨씬간단 
서브쿼리와 derived table이 비슷한 의미로 사용됨 (FROM 절에서 독립적으로 사용되면 derived table이라고 함)
별명 꼭 필요!!!!!!

SELECT column_list
FROM ( SELECT column_list FROM table1) derived_table_name  # 꼭 별명 지어줘야함
WHERE derived_table_name.c1 > 0;


*/

SELECT productcode, ROUND(SUM(quantityordered * priceeach)) sales
FROM orderdetails 
INNER JOIN orders USING (ordernumber)
WHERE YEAR(shippeddate)=2003
GROUP BY productcode
ORDER BY sales DESC
LIMIT 5;  # >>> productcode, sales 나옴 

# 위의 쿼리를 그대로 서브 쿼리로 주고, 별명 지어주고 products 테이블과 조인
SELECT productname, sales
FROM ( SELECT productcode, ROUND(SUM(quantityordered * priceeach)) sales
		FROM orderdetails 
		INNER JOIN orders USING (ordernumber)
		WHERE YEAR(shippeddate)=2003
		GROUP BY productcode
		ORDER BY sales DESC
		LIMIT 5) top5_products_2003
INNER JOIN products USING(productcode);  # 제품이름, sales나옴

# 2003년 고객을 platinum, gold, silver로 분류할때 CASE, GROUP BY 사용하기
SELECT customernumber, ROUND(SUM(quantityordered * priceeach)) AS sales,
	(CASE WHEN SUM(quantityordered * priceeach ) < 10000 THEN 'silver'
		WHEN SUM(quantityordered * priceeach ) BETWEEN 10000 AND 100000 THEN 'gold'
		WHEN SUM(quantityordered * priceeach ) > 100000 THEN 'platinum' 
	END) AS customergroup
FROM orderdetails
INNER JOIN orders USING (ordernumber)
WHERE YEAR(shippeddate) =2003 
GROUP BY customernumber ;
#GROUP BY customergroup; # 는 안됨 

# 그리고 나서 위의 쿼리를 서브쿼리로 사용하고 customer group 별로 카운트반환! 
SELECT customergroup, COUNT(cg.customergroup) AS groupcount
FROM ( SELECT customernumber, ROUND(SUM(quantityordered * priceeach)) AS sales,
		(CASE WHEN SUM(quantityordered * priceeach ) < 10000 THEN 'silver'
			WHEN SUM(quantityordered * priceeach ) BETWEEN 10000 AND 100000 THEN 'gold'
			WHEN SUM(quantityordered * priceeach ) > 100000 THEN 'platinum' 
		END) AS customergroup
	FROM orderdetails
	INNER JOIN orders USING (ordernumber)
	WHERE YEAR(shippeddate) =2003 
	GROUP BY customernumber) cg
GROUP BY cg.customergroup;

/* EXISTS 
서브쿼리에서 반환된 행의 존재여부를 테스트하는데에 사용됨
SELECT selected_list
FROM a_table
WHERE [NOT] EXISTS(subquery);

하나이상의 행이 존재하면 TRUE반환 그렇지 않으면 FALSE

*/

# 하나이상의 주문이 있는 고객을 찾음! 
SELECT customernumber, customername
FROM customers
WHERE # EXISTS(SELECT 1 FROM orders WHERE orders.customernumber = customers.customernumber);  # 주문 한사람 
	NOT EXISTS(SELECT 1 FROm orders WHERE orders.customernumber = customers.customernumber);  #  주문 안한사람 
    
# UPDATE EXISTS  샌프란시스코 직원들의 번호를 업데이트하려고 할때 
SELECT employeenumber, firstname, lastname, extension 
FROM employees
WHERE EXISTS (SELECT 1   # 아마도 TRUE가 1이니까.....겠지? 확인하기 
				FROM offices 
                WHERE city='san francisco' 
                AND offices.officecode = employees.officecode);
    

# 이제 뒤에 1 추가  -> 원래 된다는데....안됨 
# Error Code: 1175. You are using safe update mode and you tried to update a table without a WHERE that uses a KEY column.  
# To disable safe mode, toggle the option in Preferences -> SQL Editor and reconnect.


UPDATE employees 
SET extension = CONCAT(extension , '1')
WHERE EXISTS (SELECT 1     
				FROM offices 
                WHERE city='san francisco' 
                AND offices.officecode = employees.officecode) ;

# 판매주문이 없는 고객 별도로 보관할 테이블 만들기
CREATE TABLE c_archive
LIKE customers;

INSERT INTO c_archive 
SELECT * 
FROM customers 
WHERE NOT EXISTS(SELECT 1 FROM orders WHERE orders.customernumber = customers.customernumber);

SELECT *
FROM c_archive;

# DELETE EXISTS 를 사용해서 c_archive에 있는 주문이 있는exists 고객 삭제하기 인데 안돼 얘도  ㅜㅜ 
DELETE FROM customers
WHERE EXISTS (SELECT 1 FROM c_archive a WHERE a.customernumber =customers.customernumber);


# IN vs EXISTS 
# EXISTS 가 훨씬 빠르고 cost가 적음
# IN은 서브쿼리 실행하고 이걸 가지고 full scan함! (데이터가 정말 적다면 IN이 빠름) 

# 값으로 나옴 
EXPLAIN SELECT customernumber, customername
FROM customers 
WHERE EXISTS(SELECT 1 FROM orders WHERE orders.customernumber = customers.customernumber);

# 실행계획 보려면 result grid 말고 밑에 내리다 보면 있음  
SELECT customernumber, customername
FROM customers 
WHERE EXISTS(SELECT 1 FROM orders WHERE orders.customernumber = customers.customernumber);

# IN  
SELECT customernumber, customername
FROM customers
WHERE customernumber IN (SELECT customernumber FROM orders);



/* UNION : 둘이상의 쿼리 결과를 1개로 결합가능 
SELECT column_list
UNION [DISTINCT | ALL]
SELECT column_list
UNION [DISTINCT | ALL]
SELECT column_list
...

UNION 은 중복행을 알아서 제거 
*/

# DROP TABLE IF EXISTS t1;
CREATE TABLE t1 (id INT PRIMARY KEY);
CREATE TABLE t2 (id INT PRIMARY KEY);
INSERT INTO t1 VALUES (1),(2),(3);
INSERT INTO t2 VALUES (2),(3),(4);

SELECT id
FROM t1
#UNION 
UNION ALL
SELECT id
FROM t2;  # >>>> 1,2,3,4 중복제거하고 전체 나옴 

# UNION  ALL 하면 중복도 포함 

# 직원의 고객과  단일결과로 결합할때
SELECT firstname, lastname
FROM employees
UNION
SELECT contactfirstname, contactlastname
FROM customers;

# col name 바꾸려면 SELECT 에서 별명 지어주기 
SELECT CONCAT(firstname, ' ' ,lastname) fullname
FROM employees
UNION 
SELECT CONCAT(contactfirstname, ' ',contactlastname)
FROM customers
ORDER BY fullname;

# 구분 열 추가( 직원과 고객) 
SELECT CONCAT(firstname, ' ' ,lastname) fullname, 'Employee' AS contact_type
FROM employees
UNION 
SELECT CONCAT(contactfirstname, ' ',contactlastname), 'Customer' AS contact_type
FROM customers
ORDER BY fullname;


/* MINUS 
# ORACLE에서 가능 여기서는 안됨 
SELECT select_list1 
FROM table_name1
MINUS 
SELECT select_list2 
FROM table_name2;

select_list1 과 select_list2의 모양이 똑같아야함 

# MYSQL ver.
SELECT select_list
FROm table1
LEFT JOIN table2 ON join_condition
WHERE table2.col_name IS NULL;

*/
# 위에서 만든 t1,t2 table
SELECT * FROM t1; # 1 2 3
SELECT * FROM t2; # 2 3 4 


# SELECT id FROM t1 
# MINUS  안됨! 오라클에서는 됨! 
# SELECT id FROM t2;

SELECT id FROM t1 
LEFT JOIN t2 USING (id)
WHERE t2.id IS NULL;

/* INTERSECT  - mysql 안됨 
# 교집합 반환 
(SELECT column_list 
FROM table_1)
INTERSECT
(SELECT column_list
FROM table_2);
*/

# mysql ver 1
SELECT DISTINCT id FROM t1 
INNER JOIN t2 USING (id);
# mysql ver 2
SELECT DISTINCT id FROM t1
WHERE id IN (SELECT id FROM t2);


/* INSERT 
INSERT INTO TABLE(c1,c2,...)
VALUES (v1, v2,...);

# 여러행 insert
INSERT INTO TABLE(c1,c2,...)
VALUES 
	(v11,v12,...),
    (v21,v22,...),
    ...
    (vnn.vn2,...);
*/


CREATE TABLE IF NOT EXISTS tasks (
    task_id INT AUTO_INCREMENT, title VARCHAR(255) NOT NULL,
    start_date DATE, due_date DATE,
    priority TINYINT NOT NULL DEFAULT 3,   # DEFAULT 값이 3 
    description TEXT,
    PRIMARY KEY (task_id)
);

# INSERT 방법1 
INSERT INTO tasks(title, priority)
VALUE ('learn mysql INSERT statement',1);  # >>> 1 row(s) affected

SELECT * FROM tasks;   #>>> 0번째 col -> task_id 는 AUTO_INCREMENT COL:row가 하나씩 생길때 마다 순차적으로 정수 생성

# INSERT 방법 2
INSERT INTO tasks(title, priority)
VALUES ('understanding DEFAULT keyword in INSERT statement', DEFAULT);

SELECT * FROM tasks; # 를 보면 priority 가 3이 된걸 알수 있음

# INSERT 날짜  'YYYY-MM-DD' 
INSERT INTO tasks(title, start_date, due_date)
VALUES('Insert date into table','2018-01-09','2018-09-15');

INSERT INTO tasks(title, start_date, due_date)
VALUES('Insert date into table using current date',CURRENT_DATE(), CURRENT_DATE());

# 여러개 INSERT 
INSERT INTO tasks(title, priority)
VALUES ('my first task',1),
	('It\'second task',2),
    ('this is the third',3);
    
/* 여러헁 INSERT
INSERT INTO table_name( col_list)
VALUE 
    (value_list_1),
    (value_list_2),
    ...
    (value_list_n);
    
너무 많은 행을 넣으려고 하면 MAX_ALLOWED_PAXKET 오류 발생 
*/ 

SHOW VARIABLES LIKE 'max_allowed_packet'; # >>> 4194304  : mysql 서버의 출력 
# 사이즈 조정하려면 : SET GLOBAL max_allowed_packet=size;

DROP TABLE projects;
CREATE TABLE projects(
    project_id INT AUTO_INCREMENT, 
    name VARCHAR(100) NOT NULL,
    start_date DATE,
    end_date DATE,
    PRIMARY KEY(project_id)
);


INSERT INTO 
    projects(name, start_date, end_date)
VALUES
    ('AI for Marketing','2019-08-01','2019-12-31'),
    ('ML for Sales','2019-05-15','2019-11-20');

SELECT * FROM projects;


/* INSERT INTO SELECT 
INSERT INTO table_name(c1,c2,...)
VALUES(v1,v2,..);

# INSERT 다음에 VALUE 대신에 SELECT 사용가능 -> 다른 테이블의 데이터를 복사하거나 옮기는 등 작업을 할때 유용 
INSERT INTO table_name(column_list)
SELECT 
   select_list 
FROM 
   another_table
WHERE
   condition;

*/
CREATE TABLE suppliers (
    supplierNumber INT AUTO_INCREMENT,
    supplierName VARCHAR(50) NOT NULL,
    phone VARCHAR(50),
    addressLine1 VARCHAR(50),
    addressLine2 VARCHAR(50),
    city VARCHAR(50),
    state VARCHAR(50),
    postalCode VARCHAR(50),
    country VARCHAR(50),
    customerNumber INT,
    PRIMARY KEY (supplierNumber)
);

SELECT customernumber, customername, phone, addressline1, addressline2,
	city, state, postalcode, country
FROM customers
WHERE country = 'usa' AND state ='ca';

INSERT INTO suppliers ( suppliername, phone, addressline1, addressline2, 
						city, state, postalcode, country, customernumber)
SELECT customername, phone, addressline1, addressline2, 
	city, state, postalcode, country, customernumber
FROM customers
WHERE country = 'usa' AND state ='ca'; # >>> 11 row(s) affected Records: 11  Duplicates: 0  Warnings: 0

SELECT * FROM suppliers;


# INSERT INTO 와 VALUE 사용
CREATE TABLE stats ( totalproduct INT, totalcustomer INT, totalorder INT);
INSERT INTO stats(totalproduct, totalcustomer, totalorder)
VALUES ( 
	(SELECT COUNT(*) FROM products),
	(SELECT COUNT(*) FROM customers),
    (SELECT COUNT(*) FROM orders)
    );
SELECT * FROM stats;

/* INSERT ON DUPLICATE KEY UPDATE 중복키 업데이트! 
새로운 row입력할때 중복 유니크 인덱스나 피키 오류 발생할 수 있음 그때 사용

INSERT INTO table (column_list)
VALUES (value_list)
ON DUPLICATE KEY UPDATE
   c1 = v1, 
   c2 = v2,
   ...;


INSERT INTO table_name(c1)
VALUES(c1)
ON DUPLICATE KEY UPDATE c1 = VALUES(c1) + 1;

new row가 삽입되면 영향을 받는 1개의 행 -> 기존의 행이 업데이트 되면 2개가 영향받음
기존행이 현재값을 사용하여 업데이트 하면 영향 아무도 안받음! 
*/

CREATE TABLE devices (id INT AUTO_INCREMENT PRIMARY KEY, 
					name VARCHAR(100) 
                    ) ;
INSERT INTO devices (name)
VALUES ('router f1'),('switch 1'),('switch 2');

SELECT id, name FROM devices;

# 중복이 없으니까 기존처럼 insert 됨 
INSERT INTO devices (name)
VALUES ('printer') ON DUPLICATE KEY UPDATE name ='printer'; # 1 row(s) affected

# 기존 자리에 넣으려고 하면
INSERT INTO devices (id, name)
VALUES (4, 'printer') ON DUPLICATE KEY UPDATE name='central printer'; # 2 row(s) affected   # printer-> central printer로 업데이트 

/* INSERT IGNORE 

INSERT IGNORE INTO table(column_list)
VALUES( value_list),
      ( value_list),
      ...

INSERT : 여러줄을 추가하는데 오류가 발생하면 쿼리 종료하고 오류 리턴 -> 테이블에 행 삽입 안됨 
INSERT IGNORE : 오류를 발생시키는 데이터는 무시되고, 유효한 데이터만 테이블에 삽입 ! 
*/

CREATE TABLE subscribers ( id INT PRIMARY KEY AUTO_INCREMENT,
						email VARCHAR(50) NOT NULL UNIQUE) ;  # UNIQUE 는 중복이메일이 없게끔 

INSERT INTO subscribers(email)
VALUES('john.doe@gmail.com');

INSERT INTO subscribers(email)
VALUES('john.doe@gmail.com'), 
      ('jane.smith@ibm.com');  # Error Code: 1062. Duplicate entry 'john.doe@gmail.com' for key 'email'

SELECT * FROM subscribers;  # john만 있음

INSERT IGNORE INTO subscribers(email)
VALUES('john.doe@gmail.com'), 
      ('jane.smith@ibm.com');
SHOW WARNINGS; # 경고의 세부사항 볼수 있음 ! john이 왜 안된지! 

# INSERT IGNORE & STRICT 
CREATE TABLE tokens(s VARCHAR(6) );
INSERT INTO tokens VALUES ('abcdefg');  # Error Code: 1406. Data too long for column 's' at row 1
INSERT IGNORE INTO tokens VALUES ('abcdefg');  # 1 row(s) affected, 1 warning(s): 1265 Data truncated for column 's' at row 1

SELECT * FROM tokens; # >>> 마지막에 g짤림

/* UPDATE 
기존 데이터 수정 (1개 이상에서 가능 ) 
 
UPDATE [LOW_PRIORITY] [IGNORE] table_name 
SET 
    column_name1 = expr1,
    column_name2 = expr2,
    ...
[WHERE  condition];  # 생략하면 모든 행을 업데이트! 매우 주의!! 

UPDATE 절에서 
# LOW_PRIORITY : 데이터를 읽는 연결이 없을때까지 업데이트를 지연, storage engine 레벨을 사용 , memory, myISam, MERGE 등  -> 확인하기 
# IGNORE : 오류가 발생해도 업데이트를 계속 (오류는 제외하고 오류없는것만) 
*/

# mary 의 이메일 업데이트
SELECT firstname, lastname, email
FROM employees
WHERE employeenumber =1056;
 
UPDATE employees 
SET email = 'mary.patterson@classicmodelcars.com'
WHERE employeeNumber = 1056;

# 1행의 여러열 업데이트 
UPDATE employees 
SET lastname ='hill', 
	email = 'mary.hill@classicmodelcars.com'
WHERE employeenumber =1056;

# 문자열 바꾸기 모든 이메일의 도메인을 변경하고, office code도 6으로 업데이트
UPDATE employees
SET email = REPLACE (email, '@classicmodelcars.com','@mysqltutorial.org')
WHERE jobtitle ='sales rep' AND officecode = 6;
# Error Code: 1175. You are using safe update mode and you tried to update a table without a WHERE that uses a KEY column. 
# Cannot use range access on index 'officeCode' due to type or collation conversion on field 'officeCode' To disable safe mode, toggle the option in Preferences -> SQL Editor and reconnect.


# 영업담당자 sales rep employeenumber 이 NULL인 고객 업데이트
SELECT customername, salesrepemployeenumber
FROM customers
WHERE salesrepemployeenumber IS NULL;

SELECT employeenumber
FROM employees
WHERE jobtitle = 'sales rep'  # 인 사람중에서 
ORDER BY RAND()  # 임의로  선택
LIMIT 1;  # 1명  --> 아래에서 인라인뷰(하위 쿼리로 들어감) 


UPDATE customers 
SET salesrepemployeenumber = ( SELECT employeenumber 
								FROM employees 
                                WHERE jobtitle = 'sales rep'  # 인 사람중에서 
								ORDER BY RAND()  # 임의로  선택? 
								LIMIT 1) 
WHERE salesrepemployeenumber IS NULL;
# 22 row(s) affected Rows matched: 22  Changed: 22  Warnings: 0
# 이제 위에 IS NULL 실행하면 아무도 없음! 

/* UPDATE JOIN 

UPDATE T1, T2,
[INNER JOIN | LEFT JOIN] T1 ON T1.C1 = T2. C1     # 업데이트 쓰고 바로 여기서 테이블 지정해줘야 업데이트! 
SET T1.C2 = T2.C2, 
    T2.C3 = expr
WHERE condition  업데이트할 행을 제한하는 조건 

# 또다른 방법  (Cross table update ) 
UPDATE T1, T2
SET T1.c2 = T2.c2,
      T2.c3 = expr
WHERE T1.c1 = T2.c1 AND condition

# 이렇게도 작성 가능
UPDATE T1,T2
INNER JOIN T2 ON T1.C1 = T2.C1
SET T1.C2 = T2.C2,
      T2.C3 = expr
WHERE condition
*/


CREATE DATABASE IF NOT EXISTS empdb;
 
USE empdb;
 
-- create tables
CREATE TABLE merits (
    performance INT(11) NOT NULL,
    percentage FLOAT NOT NULL,
    PRIMARY KEY (performance)
);
 
CREATE TABLE employees (
    emp_id INT(11) NOT NULL AUTO_INCREMENT,
    emp_name VARCHAR(255) NOT NULL,
    performance INT(11) DEFAULT NULL,
    salary FLOAT DEFAULT NULL,
    PRIMARY KEY (emp_id),
    CONSTRAINT fk_performance FOREIGN KEY (performance)
        REFERENCES merits (performance)
);
-- insert data for merits table
INSERT INTO merits(performance,percentage)
VALUES(1,0),
      (2,0.01),
      (3,0.03),
      (4,0.05),
      (5,0.08);
-- insert data for employees table
INSERT INTO employees(emp_name,performance,salary)      
VALUES('Mary Doe', 1, 50000),
      ('Cindy Smith', 3, 65000),
      ('Sue Greenspan', 4, 75000),
      ('Grace Dell', 5, 125000),
      ('Nancy Johnson', 3, 85000),
      ('John Doe', 2, 45000),
      ('Lily Bush', 3, 55000);

# 직원의 성과에 따라서 급여를 조정하려고 할 때
# 평가와 직원을 우선 조인해야함! 
UPDATE employees 
INNER JOIN merits ON employees.performance = merits.performance
SET salary = salary + salary * percentage;

SELECT * FROM employees;


# 2명의 직원을 더 고용 (신입, 성과performance가 없음 아직) 
INSERT INTO employees (emp_name, performance, salary)
VALUES ('Jack William',NULL,43000),
      ('Ricky Bond',NULL,52000);      

UPDATE employees 
LEFT JOIN merits ON employees.performance = merits.performance 
SET salary = salary + salary * 0.015
WHERE merits.percentage IS NULL;
# 얘도 저위에처럼 에러 ㅠ
      

/* DELETE 데이터 삭제 (in 단일 테이블) 
DELETE FROM table_name
WHERE condition;

# 여러 테이블에서 데이터를 삭제하려면 DELETE JOIN 사용 
# 테이블의 모든 행을 삭제하려면 TRUNCATE TABLE 사용 
# 외래키 제약조건이 있는 테이블의 경우
driving 테이블에서 행을 삭제할때 ON DELETE CASCADE 옵션 사용하면 driven 테이블의 행이 자동 삭제

# 일단 데이터를 삭제하면 사라지니까 데이터베이스를 백업해야함!  지울때는 항상 조심! 

# 삭제할 행 수를 제한하려면 LIMIT 사용 (LIMIT 은 꼭 ORDER BY 와 함께 정렬하고 나서 사용!) 
DELETE FROM table_name
ORDER BY c1, c2, ...
LIMIT row_count;
*/

USE classicmodels;
DELETE FROM employees 
WHERE officecode = 4; # 얘도 안됨 ...왜지? 

DELETE FROM customers
ORDER BY customername
LIMIT 10;

DELETE FROM customers
WHERE country = 'france'
ORDER BY creditLimit
LIMIT 5;

/* DELETE JOIN : 여러 테이블에서 삭제하려고 할때 

# with INNER JOIN 
DELETE T1, T2
FROM T1
INNER JOIN T2 ON T1.key = T2.key
WHERE condition;

t2 테이블을 생략하면 t1만 삭제됨 

# with LEFT JOIN 
DELETE T1 
FROM T1
        LEFT JOIN
    T2 ON T1.key = T2.key 
WHERE
    T2.key IS NULL;

*/

DROP TABLE IF EXISTS t1, t2;
 
CREATE TABLE t1 (
    id INT PRIMARY KEY AUTO_INCREMENT
);
 
CREATE TABLE t2 (
    id VARCHAR(20) PRIMARY KEY,
    ref INT NOT NULL
);
 
INSERT INTO t1 VALUES (1),(2),(3); 
INSERT INTO t2(id,ref) VALUES('A',1),('B',2),('C',3);

SELECT * FROM t1; # 1,2,3 
SELECT * FROM t2;  # a:1, b:1, c:1 

DELETE t1,t2 FROM t1 
INNER JOIN t2 ON t2.ref = t1.id
WHERE t1.id =1;   # A 사라짐 

# 고객은 여러개 주문 가능, 주문은 1개의 고객 가능
# 주문을 하지않은 고객을 제거 -> 24개 사라짐 
DELETE customers 
FROM customers
LEFT JOIN orders ON customers.customernumber = orders.customernumber 
WHERE ordernumber IS NULL;  

# 위에서 주문을 하지않은 고객을 제거했으니, 당연히 빈 집합 나옴! 
SELECT c.customernumber, c.customername,ordernumber
FROM customers c
LEFT JOIN orders o USING(customernumber)
WHERE ordernumber IS NULL;

/* DELETE CASCADE 
건물 TABLE : 건물번호, 이름, 주소
방 TABLE : 방번호, 방이름, 건물번호

DELETE FROM buildings 
WHERE building_no =2 ;
2번 건물을 지우고, 그에 해당하는 방도 다 지우고싶을때! 부모에 연결된 자식도 제거 

*/

CREATE TABLE buildings (
    building_no INT PRIMARY KEY AUTO_INCREMENT,
    building_name VARCHAR(255) NOT NULL,
    address VARCHAR(255) NOT NULL
);

CREATE TABLE rooms (
    room_no INT PRIMARY KEY AUTO_INCREMENT,
    room_name VARCHAR(255) NOT NULL,
    building_no INT NOT NULL,
    FOREIGN KEY (building_no)
        REFERENCES buildings (building_no)
        ON DELETE CASCADE  # 외래키 정의할때 여기에 ON DELETE CASCADE 써있는거 주의! 
        
);

INSERT INTO buildings(building_name,address)
VALUES('ACME Headquaters','3950 North 1st Street CA 95134'),
      ('ACME Sales','5000 North 1st Street CA 95134');
      
SELECT * FROM buildings;

INSERT INTO rooms(room_name,building_no)
VALUES('Amazon',1),
      ('War Room',1),
      ('Office of CEO',1),
      ('Marketing',2),
      ('Showroom',2);

SELECT * FROM rooms;