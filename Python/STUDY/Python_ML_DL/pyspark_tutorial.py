# < pyspark tutorial 0 from https://yujuwon.tistory.com/entry/spark-tutorial


from pyspark import SparkContext
sc=SparkContext()
type(sc) # pyspark.context.SparkContext
dir(sc)  # 사용가능한 sc의 함수, attribute list확인
help(sc)  # 도움말
sc.version  # '2.4.4'

# xrange는 generator를 생성(RDD엔 이걸 더 추천) / range는 범위를 포함되는 모든 숫자 리스트 생성
# RDD 생성할땐 sc.parallelize(data, 숫자) -> 데이터를 숫자만큼 쪼개서 메모리에 저장해!

help(sc.parallelize)

data=range(1,10001)
rdd=sc.parallelize(data,8)

type(rdd) # 해당 RDD 타입 확인   pyspark.rdd.PipelinedRDD
rdd.getNumPartitions() # 해당 RDD의 파티션 숫자 확인  8
rdd.toDebugString()
'''
Out[12]: b'(8) PythonRDD[1] at RDD at PythonRDD.scala:53 []\n |  ParallelCollectionRDD[0] at parallelize at PythonRDD.scala:195 []'
'''
rdd.id() #id 확인   1
rdd.setName('My First RDD')  # My First RDD PythonRDD[1] at RDD at PythonRDD.scala:53
rdd.toDebugString()
'''
Out[15]: b'(8) My First RDD PythonRDD[1] at RDD at PythonRDD.scala:53 []\n |  ParallelCollectionRDD[0] at parallelize at PythonRDD.scala:195 []'

'''
help(rdd)

# map : rdd -> sub_rdd로 재탄생!
# collect (collect는 적은양의 데이터에만)

def sub(value):
    return (value-1)

sub_rdd=rdd.map(sub)  # sub함수를  적용!

sub_rdd.collect()  # 실제로 action이 일어나고 sub함수가 적용됨!

# count() rdd 내의 요소의 개수 세기 얘도 action
rdd.count()      # Out[20]: 10000
sub_rdd.count()  # Out[21]: 10000

# filter : 함수 결과가 참인 경우에만 요소들을 통과! 새로운 rdd 생성 action 아님
def ten(value):
    if (value < 10):
        return True
    else :
        return False

filtered_rdd = sub_rdd.filter(ten)
filtered_rdd.collect()  # Out[24]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# lambda () : 이름을 할당받을 필요가 없는 한줄짜리 익명함수를 만들때  -> filter 에서 사용
lambda_rdd=sub_rdd.filter(lambda x: x<10)
lambda_rdd.collect()  #Out[26]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

even_rdd = lambda_rdd.filter(lambda x: x%2 ==0)
even_rdd  # PythonRDD[7] at RDD at PythonRDD.scala:53
even_rdd.collect() # Out[29]: [0, 2, 4, 6, 8]































# < pyspark tutorial 1 from  https://medium.com/@amresh_php/spark-tutorial-starting-with-pyspark-part-1-7b6fbb20f9f0


from pyspark import SparkContext
sc=SparkContext('local','First App')

rdd=range(100)
data=sc.parallelize(rdd)
data.count()             #   Out[9]: 100

data.take(4)         # Out[10]: [0, 1, 2, 3]



























# < pyspark tutorial 2 from https://www.guru99.com/pyspark-tutorial.html#1

# < TODO income을 label로 싹다 바꾸기!

# SparkContext - 클러스터와 연결을 허용하는 내부 엔진! 작업실행시 필요
from pyspark import SparkContext
sc=SparkContext()

# create a collection of data called RDD
nums=sc.parallelize([1,2,3,4,5])
nums.take(1) # access the first row  # take로 확인가능
nums.take(2)

# using a lambda func
squared=nums.map(lambda x: x*x).collect()
squared


# SQLContext -위 보다 편리한 방법! dataFrame 사용하기 -> 다른 데이터 소스와 연결가능
from pyspark.sql import Row
from pyspark.sql import SQLContext
sqlcontext=SQLContext(sc)

list_p=[('John',19),('Smith',29),('Adam',35),('Henry',50)]  # step1) create the list of tuple

rdd=sc.parallelize(list_p)  # step2) build a RDD

ppl = rdd.map(lambda x: Row(name=x[0], age=int(x[1])))  # step3) conver the tuples

df_ppl=sqlcontext.createDataFrame(ppl) # step4) create a DF context

df_ppl.printSchema()



# ML with spark
'''
Step 1) Basic operation with PySpark
Step 2) Data preprocessing
Step 3) Build a data processing pipeline
Step 4) Build the classifier
Step 5) Train and evaluate the model
Step 6) Tune the hyperparameter
'''


from pyspark.sql import SQLContext
from pyspark import SparkContext

sc=SparkContext() # 위에서 지정해서 안되나봄 초기화 필요!
url= 'https://raw.githubusercontent.com/guru99-edu/R-Programming/master/adult_data.csv'
from pyspark import SparkFiles
sc.addFile(url)
sqlcontext=SQLContext(sc)


df=sqlcontext.read.csv(SparkFiles.get('adult_data.csv'),
                       header=True,
                       inferSchema=True)  # True: 데이터 타입별로 string, float로 읽어옴
df.printSchema()
df.show(5, truncate=False)


df_string=sqlcontext.read.csv(SparkFiles.get('adult_data.csv'),
                       header=True,
                       inferSchema=False)  # False: 모두다 string 으로 읽어옴

df_string.printSchema()
df_string.show(5, truncate=False)


# 전부 문자열인것을 원하는 data type으로 바꾸는 방법
from pyspark.sql.types import *
def convertCol(df, names, newtype):
    for name in names:
        df=df.withColumn(name, df[name].cast(newtype))  # withColumn이용해서 지정한 열만 변환
    return df


CONTI_FEATURES =['age','fnlwgt','capital-gain','educational-num','capital-loss','hours-per-week']
df_string =convertCol(df_string, CONTI_FEATURES, FloatType())  # string convert to float
df_string.printSchema()


'''
# StringIndexter 사용하는 방법! 
from pyspark.ml.feature import StringIndexer
stringIndexer=StringIndexer(inputCol='income', outputCol='newincome')
model=stringIndexer.fit(df_string)

df=model.transform(df)
df.printSchema()
'''

# select columns
df.select('age','fnlwgt').show(5)
# count by group  그룹바이 할때 대문자 B 소문자 상관 없
df.groupBy('education').count().sort('count',ascending=True).show()
df.groupby('education').count().sort('count',ascending=True).show()



# describe the data : count, mean, standarddeviation, min, max
df.describe().show() #  전체 통계
df.describe('capital-gain').show()  # 요약통계

# crosstab computation - 한개의 컬럼에 2개의 기준으로 나눠서 볼때
# 위에 income 인덱싱 했음!
# 비교할 변수2개, sort는 변수_변수 로 쓰면 됨
df.crosstab('age','income').sort('age_income').show()

# drop column
# drop(): drop a col   / dropna() : drop na's
df.drop('education-num')  # .columns 왜 뒤에 붙어있지..?

# filter()
df.filter(df.age >40).count()

# 그룹별 기술통계
# group data by group and statistical operation(mean, max,min ..)
df.groupBy('marital-status').agg({'capital-gain':'mean'}).show()





# step 2) data preprocessing

from pyspark.sql.functions import *

# 우선 열 선택해서 봐
age_square =df.select(col('age')**2) # sql.funtion하니까 col해도 오류 안생김
age_square.show()

# df에 추가하는 법 withColumns
df=df.withColumn('age_square', col ('age')**2)
df.printSchema()
df.describe('age_square').show()

# col list 를 통하여 col 순서 바꾸기
new_col=['age','age_square','workclass','fnlwgt','education','educational-num','marital-status','occupation',
         'relationship','race','gender','capital-gain','capital-loss','hours-per-week','native-country','income']

df=df.select(new_col)
df.first()
df.show()

# income -> label로 바꿈
df=df.withColumnRenamed('income','label')
df.printSchema()


df.filter(df['native-country'] =='Holand-Netherlands').count()  # 1명  다른것에 비해 넘 작아서 삭제할예정
df.groupBy('native-country').agg({'native-country':'count'}).sort(asc('count(native-country)')).show() # asc :ascending?

df_remove=df.filter(df['native-country'] != 'Holand-Netherlands')



# one hot encoding 예제
'''
Index the string to numeric
Create the one hot encoder
Transform the data
'''
# 문자열 범주형 데이터 -> 숫자 인덱스로 변환(string indexer) -> 원핫 인코더로 벡터화 시킴
from pyspark.ml.feature import StringIndexer, OneHotEncoder,VectorAssembler

stringindexer=StringIndexer(inputCol='workclass', outputCol='workclass_encoded')
model = stringindexer.fit(df)
indexed=model.transform(df)
indexed.show()

# eate the news columns based on the group. For instance,
# if there are 10 groups in the feature,
# the new matrix will have 10 columns, one for each group.

encoder=OneHotEncoder(dropLast=False, inputCol='workclass_encoded',outputCol='workclass_vec')
encoded=encoder.transform(indexed)
encoded.show(10)


# Build the Pipeline
'''
1) Encode the categorical data
2) Index the label feature
3) Add continuous variable
4) Assemble the steps.
'''

# 범주형을 먼저 라벨링(stringIndexer로) 하고 원 핫 / 수치형은 라벨링 필요 없
# vectorassmbler 는 말그대로 array로 변환 for ML


# 1) encoding the categorical data with string_indexer -> onehot

from pyspark.ml.feature import StringIndexer, OneHotEncoder,VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator

CATE_FEATURES=['workclass','education','marital-status','occupation','relationship','race','gender',
                  'native-country']
stages=[] # stages in Pipeline

for categoricalCol in CATE_FEATURES:
    stringIndexer=StringIndexer(inputCol=categoricalCol,
                              outputCol=categoricalCol+'Index')
    encoder=OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()],
                                   outputCols=[categoricalCol+'classVec'])
    stages += [stringIndexer, encoder]



# 2) index the label(label) feature
# convert label(label) into label indices using the StringIndexer
label_stringIDX= StringIndexer(inputCol='label',outputCol='newlabel')
stages += [label_stringIDX]  # label을 레이블 인덱스로 변환


# 3)  add continuous variable
'''
The inputCols of the VectorAssembler is a list of columns. 
You can create a new list containing all the new columns. 
The code below popluate the list with encoded categorical features and the continuous features.
'''
CONTI_FEATURES =['age','fnlwgt','capital-gain','educational-num','capital-loss','hours-per-week']
assemblerInputs=[c + 'classVec' for c in CATE_FEATURES] + CONTI_FEATURES
assemblerInputs

# 4) assemble the steps
assembler= VectorAssembler(inputCols=assemblerInputs, outputCol='features')
stages += [assembler]

# create a pipeline
# TODO 다른 예제에서는 train/test 먼저 나누고 변환하기도 함(이게 맞는거같음 더 찾아보기)
df_remove.show()

pipeline=Pipeline(stages=stages)
pipelineModel=pipeline.fit(df_remove)
model=pipelineModel.transform(df_remove)

model.take(1)
'''
Out[116]: [Row(age=25, age_square=625.0, 
workclass='Private', fnlwgt=226802, education='11th', educational-num=7, 
marital-status='Never-married', occupation='Machine-op-inspct', 
relationship='Own-child', race='Black', gender='Male', 
capital-gain=0, capital-loss=0, hours-per-week=40, native-country='United-States', 
label='<=50K', 
workclassIndex=0.0, 
workclassclassVec=SparseVector(8, {0: 1.0}), 
educationIndex=5.0, 
educationclassVec=SparseVector(15, {5: 1.0}), 
marital-statusIndex=1.0, 
marital-statusclassVec=SparseVector(6, {1: 1.0}), 
occupationIndex=6.0, 
occupationclassVec=SparseVector(14, {6: 1.0}), 
relationshipIndex=2.0, 
relationshipclassVec=SparseVector(5, {2: 1.0}), 
raceIndex=1.0, 
raceclassVec=SparseVector(4, {1: 1.0}), 
genderIndex=0.0, 
genderclassVec=SparseVector(1, {0: 1.0}), 
native-countryIndex=0.0, 
native-countryclassVec=SparseVector(40, {0: 1.0}), 
newlabel=0.0, 
features=SparseVector(99, {0: 1.0, 13: 1.0, 24: 1.0, 35: 1.0, 45: 1.0, 49: 1.0, 52: 1.0, 53: 1.0, 93: 25.0, 94: 226802.0, 96: 7.0, 98: 40.0}))]

'''

# step4) build the classifier : logistic
# to make the computation faster, convert model to a DF
# select newlabel and features from model using map

from pyspark.ml.linalg import DenseVector
input_data=model.rdd.map(lambda x: (x['newlabel'], DenseVector(x['features'])))

df_train = sqlcontext.createDataFrame(input_data, ['label','features'])
df_train.show(2)

train_data, test_data=df_train.randomSplit([.8,.2],seed=1234)

train_data.groupby('label').agg({'label':'count'}).show()
test_data.groupby('label').agg({'label':'count'}).show()

# build the logreg
from pyspark.ml.classification import LogisticRegression
# initialize logreg
lr=LogisticRegression(labelCol='label', featuresCol='features', maxIter=10, regParam=0.3)
linearModel=lr.fit(train_data)

print('coefficients:'+str(linearModel.coefficients))
print('intercept:' +str(linearModel.intercept))
'''
coefficients:[-0.06466122261245467,-0.15477705031513067,-0.05256419958240657,-0.1646151514623772,-0.13260631577277007,0.1440092662688071,0.18819650236987692,-0.2353917978203866,-0.19338304459729486,-0.062372844641401755,0.22197010817874652,0.39215010962219765,-0.00971725457317707,-0.3054741678671564,-0.01616078293549077,-0.335911608896674,-0.4339513254313805,0.5259759783441236,-0.37260884764314844,-0.2003170629817971,0.5939988732725539,-0.342253178630009,-0.39578598875147036,0.3302004575488263,-0.34555181671484486,-0.2176427962480631,-0.21092696634656566,-0.14178261082016447,-0.11691515241239007,0.19263520246733742,-0.06288041320729375,0.2929361580563072,-0.1053731267310531,0.039405070378344224,-0.290371961646508,-0.2108515197208075,-0.1659410714056909,-0.13056531343398278,-0.287840247111254,-0.3277598776047413,0.11972593557373297,0.1154654388987926,-0.2700104536773985,0.2733255410052606,-0.19817569357300152,-0.2923832704155252,-0.24381643996383054,0.4167809050119496,-0.05612947361095992,-0.18485464732076917,-0.06139393863288085,-0.25772417076552706,0.1688555631303888,-0.1044730103571624,-0.3799299516416315,-0.19451013295372438,-0.04672166273757962,-0.08288615048934182,-0.23631465194321336,-0.055988804927846206,-0.2741174344279237,-0.15680065773731863,-0.18405724884491245,0.10465497961499366,-0.24710916003464434,-0.38819054530944064,-0.14068140189316425,0.11681776315385664,-0.4354234229373039,-0.10377292379938331,-0.27896089022373993,-0.14982748649632385,-0.38722909982229214,-0.620751234508206,-0.18558883152065275,-0.07021003127647768,-0.1493138255484858,-0.04254852705563133,-0.034581977161956295,-0.3749060433864179,-0.4516185832484313,-0.3483801002012016,0.18473918358317712,0.07503541154214503,-0.44256777364078154,-0.2973328074034018,0.14929427077412652,-0.41112988600321076,0.22689366786806642,-0.5059947121860132,-0.4119542247183066,-0.35077883706182167,-0.18302963947569692,0.006816248510976649,1.3460719372024384e-07,2.2429111452546832e-05,0.028169227517533677,0.00022430176016130447,0.008927708045104206]
intercept:-2.060221527178715

'''
# step 5) train and evaluate the model
predictions =linearModel.transform(test_data)  # to generate prediction, transform!

predictions.printSchema()
'''
root
 |-- label: double (nullable = true)
 |-- features: vector (nullable = true)
 |-- rawPrediction: vector (nullable = true)
 |-- probability: vector (nullable = true)
 |-- prediction: double (nullable = false)

'''



selected= predictions.select('label','prediction','probability')
selected.show(20)


# Evaluate model
cm=predictions.select('label', 'prediction')
cm.groupby('label').agg({'label':'count'}).show()
cm.groupby('prediction').agg({'prediction':'count'}).show()

cm.filter(cm.label ==cm.prediction).count() / cm.count()       #  Out[51]: 0.8216095682140685

# accuracy
def accuracy_m(model):
    predictions=model.transform(test_data)
    cm=predictions.select('label', 'prediction')
    acc=cm.filter(cm.label ==cm.prediction).count() / cm.count()
    print('model accuracy : %.3f%%' %(acc*100))
accuracy_m(model=linearModel)                                 #   model accuracy : 82.161%

# use ROC for binary classification ) = True Positive Rate(recall)  # TODO : 확인하기
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator=BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
print(evaluator.evaluate(predictions))   #   0.8952698333157076
print(evaluator.getMetricName())    # areaUnderROC

# step 6) tune the hyperparameter
'''
To reduce the time of the computation, 
you only tune the regularization parameter with only two values.
'''
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
param_grid=(ParamGridBuilder().addGrid(lr.regParam, [0.01,0.5]).build())

# time check and kfold=5
from time import *

start_time=time()
cv=CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
cvModel=cv.fit(train_data)
end_time=time()
elapsed_time=end_time-start_time
print('time to train model : %.3f seconds'%elapsed_time) #  time to train model : 616.891 seconds
accuracy_m(model=cvModel)               # model accuracy : 84.857%



bestModel=cvModel.bestModel
bestModel.extractParamMap()
'''
Out[62]: 
{Param(parent='LogisticRegression_b571d7694e34', name='aggregationDepth', doc='suggested depth for treeAggregate (>= 2)'): 2,
 Param(parent='LogisticRegression_b571d7694e34', name='elasticNetParam', doc='the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty'): 0.0,
 Param(parent='LogisticRegression_b571d7694e34', name='family', doc='The name of family which is a description of the label distribution to be used in the model. Supported options: auto, binomial, multinomial.'): 'auto',
 Param(parent='LogisticRegression_b571d7694e34', name='featuresCol', doc='features column name'): 'features',
 Param(parent='LogisticRegression_b571d7694e34', name='fitIntercept', doc='whether to fit an intercept term'): True,
 Param(parent='LogisticRegression_b571d7694e34', name='labelCol', doc='label column name'): 'label',
 Param(parent='LogisticRegression_b571d7694e34', name='maxIter', doc='maximum number of iterations (>= 0)'): 10,
 Param(parent='LogisticRegression_b571d7694e34', name='predictionCol', doc='prediction column name'): 'prediction',
 Param(parent='LogisticRegression_b571d7694e34', name='probabilityCol', doc='Column name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities'): 'probability',
 Param(parent='LogisticRegression_b571d7694e34', name='rawPredictionCol', doc='raw prediction (a.k.a. confidence) column name'): 'rawPrediction',
 Param(parent='LogisticRegression_b571d7694e34', name='regParam', doc='regularization parameter (>= 0)'): 0.01,
 Param(parent='LogisticRegression_b571d7694e34', name='standardization', doc='whether to standardize the training features before fitting the model'): True,
 Param(parent='LogisticRegression_b571d7694e34', name='threshold', doc='threshold in binary classification prediction, in range [0, 1]'): 0.5,
 Param(parent='LogisticRegression_b571d7694e34', name='tol', doc='the convergence tolerance for iterative algorithms (>= 0)'): 1e-06}

'''


