# 데이터 출처 : https://www.kaggle.com/c/titanic
# coding = 'uft-8'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')

import os
os.getcwd()

df_train = pd.read_csv('../ML/data/train.csv')
df_test=pd.read_csv('../ML/data/test.csv')
answer =pd.read_csv('../ML/data/gender_submission.csv')
answer=answer.iloc[:,1:]
answer[:30]

df_train.shape
df_train.info()

data= pd.concat([df_train,df_test],axis=0)
data.shape
data.columns =data.columns.str.lower()
data.head()

# all nan
data.isnull().sum()

# survived
np.bincount(data['survived'].isnull())

# name -> title - encoding
np.bincount(data['name'].isnull())
data['title'] = data.name.str.extract(' ([A-Za-z]+)\.', expand=False)
data['title']

pd.value_counts(data['title'])

plt.figure(figsize=(20,10))
sns.countplot(data['title'],data=data, hue=data['survived'])

title_mapping ={'Mr':1, 'Mrs':2, 'Miss':3, 'Master':4}
data['title'] =data['title'].map(title_mapping)
data['title'].unique()
data['title']=data['title'].fillna(0)
data['title']=data['title'].astype(int)

data['title']
pd.value_counts(data['title'])

#  age  - standardscaler
data['age'].value_counts()
data['age'].describe()
data['age'].fillna(method='pad',inplace=True)
np.bincount(data['age'].isnull())


#  age_range  - encoding
bins= [0,5,10,15,20,30,40,50,60,80,100]
labels=['0','1','2','3','4','5','6','7','8','9']
data['age_range']=pd.cut(data['age'],bins,labels=labels)
data['age_range']
np.bincount(data['age_range'].isnull())
data.age_range.unique()
sns.countplot(data['age_range'],data=data,hue=data['survived'])
plt.close('all')
# del data['age']
data['age_range']


# cabin-> cabin list -> encoding
data['cabin'].unique()
np.bincount(data['cabin'].isnull())
data['cabin']

data.loc[data.cabin.str[0] =='A', 'cabinlist']=1
data.loc[data.cabin.str[0] =='B', 'cabinlist']=2
data.loc[data.cabin.str[0] =='C', 'cabinlist']=3
data.loc[data.cabin.str[0] =='D', 'cabinlist']=4
data.loc[data.cabin.str[0] =='E', 'cabinlist']=5
data.loc[data.cabin.str[0] =='F', 'cabinlist']=6
data.loc[data.cabin.str[0] =='G', 'cabinlist']=7
data.loc[data.cabin.str[0] =='T', 'cabinlist']=8

data['cabin']
data['cabinlist']=data['cabinlist'].fillna(0)

data['cabinlist'].unique()
data['cabinlist']=data['cabinlist'].astype(int)

pd.value_counts(data['cabinlist'])
sns.countplot(data['cabinlist'],data=data, hue=data['survived'])


# family  - encoding
data['family'] =data['parch']+data['sibsp']+1
data['family']
data['family'].unique()
data['family'].value_counts()
sns.countplot(data['family'],data=data,hue=data['survived'])

del data['sibsp']
del data['parch']

#  alone - encoding
data['alone']=0
data.loc[data['family']==1, 'alone']=1
data['alone'].value_counts()
#sns.countplot(data['alone'],data=data,hue=data['survived'])
# plt.close('all')

#  embarked - encoding
np.bincount(data['embarked'].isnull())
# sns.countplot(data['embarked'],data=data, hue=data['survived'])

data['embarked'] =data['embarked'].fillna('S')
pd.value_counts(data['embarked'])
data['embarked']

data.loc[data['embarked']=='S','embarked']=0
data.loc[data['embarked']=='C','embarked']=1
data.loc[data['embarked']=='Q','embarked']=2



# fare -standard scaler

data['fare']
np.bincount(data['fare'].isnull())
data['fare'] =data['fare'].fillna(0)
data['fare']
data['fare']=data['fare'].astype(int)
data['fare'].value_counts()
data['fare'].describe()



# fare -> fare_range - encoding
data.loc[data['fare'] <= 7.8958, 'fare_range'] = 1
data.loc[(data['fare'] > 7.8958) & (data['fare'] <= 14.454200), 'fare_range'] = 2
data.loc[(data['fare'] > 14.454200) & (data['fare'] <= 31.275000),'fare_range'] = 3
data.loc[data['fare'] >31.275000, 'fare_range'] = 4
data['fare_range']=data['fare_range'].fillna(0)
data['fare_range']=data['fare_range'].astype(int)
pd.value_counts(data['fare_range'])

sns.countplot(data['fare_range'],data=data, hue=data['survived'])
plt.close()

# sex - encoding
data.loc[data['sex']=='female','sex']=0
data.loc[data['sex']=='male','sex']=1
pd.value_counts(data['sex'])


# ticket -> ti  - encoding
np.bincount(data['ticket'].isnull())
data['ticket'].describe()
data['ticket'].unique()
data['ticket'].replace('^\d+','0')

data['ti']  = data.ticket.str.extract('([A-Za-z]+)', expand=False)
data['ti']=data['ti'].fillna(0)
data['ti']
data.loc[data['ti']!=0,'ti']=1

sns.countplot(data['ti'],data=data, hue=data['survived'])
data['ti']
np.bincount(data['ti'].isnull())

# pclass - encoding
data['pclass'].value_counts()
data.loc[data['pclass']==3,'pclass']=0

plt.close('all')
sns.countplot(data['pclass'],data=data, hue=data['survived'])


data.columns

del data['name']
del data['cabin']
del data['passengerid']
del data['ticket']

data.columns


data.head()
data['survived'].value_counts()

#############################


data1=data.copy()
data1.head()
# data1['title_2']=data1['title']*data1['title']

data1['age']=data1['age'].astype(int)
data1['age_range']=data1['age_range'].astype(int)
data1.head()
data1.info()

plt.close('all')

# data3 : 모델기반 특성 선택으로 정렬
# >>> title, sex, pclass, fare, 까지 선택
data3 =pd.get_dummies(data1[['pclass','sex','title', 'fare','survived']],
                      columns=['pclass','sex','title'],
                      drop_first=True)
data3.head()
data3.info()

train = data3[:891]
train.head()
train.info()
train.shape

test =data3[891:]

test.head()
test=test.drop('survived', axis=1)
test.head()



'''
# randomforest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(train.drop('survived',axis=1), train['survived'],
                                                      stratify=train['survived'], random_state=1)
param_grid={'n_estimators':[10,20,30,40,50]}
grid=GridSearchCV(RandomForestClassifier(random_state=3, n_jobs=-1), param_grid, cv=5, return_train_score=True )
grid.fit(x_train,y_train)
grid.best_params_

grid.score(x_train, y_train)
grid.score(x_valid, y_valid)


# 0.75119
forest=RandomForestClassifier(random_state=3, n_jobs=-1, criterion='gini', n_estimators=50, max_features='auto')
forest.fit(train.drop('survived',axis=1), train['survived'])
forest.score(train.drop('survived',axis=1), train['survived'])
y_pred=grid.predict(test)
y_pred



'''
# 전처리 과정 preprocessing
TRAIN=train.drop('survived',axis=1)
Y_TRAIN = train['survived']

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train_scaled = sc.fit_transform(TRAIN)
X_test_scaled = sc.transform(test)

X_train_scaled.shape
X_test_scaled.shape

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid= train_test_split(X_train_scaled, Y_TRAIN,stratify=Y_TRAIN, random_state=3)



# logreg  0.77511

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)

param_grid={'C':[0.01,0.1,1,10]}
grid=GridSearchCV(LogisticRegression(solver='liblinear', random_state=1), param_grid,cv=kfold,
                  return_train_score=True )
grid.fit(x_train, y_train)
grid.best_params_  # 0.1

grid.score(x_train, y_train)
grid.score(x_valid, y_valid)

grid.fit(X_train_scaled, Y_TRAIN)
grid.best_params_  # 10
grid.score(X_train_scaled, Y_TRAIN)
y_pred=grid.predict(X_test_scaled)
y_pred



# SVC  0.79425

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)

param_grid={'C':[0.001,0.01,0.1,1],
            'gamma':[0.001,0.01,0.1,1,10,100]}
grid=GridSearchCV(SVC(random_state=1), param_grid,cv=kfold, return_train_score=True )
grid.fit(x_train, y_train)
grid.best_params_  # 1,1
grid.score(x_train, y_train)
grid.score(x_valid, y_valid)

grid.fit(X_train_scaled, Y_TRAIN)
grid.score(X_train_scaled, Y_TRAIN)

y_pred=grid.predict(X_test_scaled)
print(y_pred)



# MLP  0.78468
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
param_grid={'hidden_layer_sizes':[[10,10],[16,16],[32,32]]}
grid= GridSearchCV(MLPClassifier(random_state=3, max_iter=1000, validation_fraction=0.2), param_grid, cv=kfold)
grid.fit(x_train, y_train)
grid.best_params_  # 32

grid.score(x_train, y_train)
grid.score(x_valid, y_valid)

grid.fit(X_train_scaled, Y_TRAIN)
grid.best_params_  #16
grid.score(X_train_scaled, Y_TRAIN)

y_pred=grid.predict(X_test_scaled)
print(y_pred)



# keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
X_train_scaled.shape
X_test_scaled.shape
Y_TRAIN.shape

X_train=X_train_scaled.reshape(891,8).astype('float32')
X_test=X_test_scaled.reshape(418,8).astype('float32')


y_train=to_categorical(Y_TRAIN)
y_train[0]

# keras1 0.78947
model=Sequential()
model.add(Dense(16,input_shape=(8,), activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam',metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=200)
model.evaluate(X_train, y_train)[1]

y_pred=model.predict(X_test)
y_pred_class=np.argmax(y_pred, axis=1)
y_pred_class



# keras2  0.76555
model=Sequential()
model.add(Dense(32,input_shape=(8,), activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam',metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=200)
model.evaluate(X_train, y_train)[1]

y_pred=model.predict(X_test)
y_pred_class=np.argmax(y_pred, axis=1)
y_pred_class





# keras3  0.78468
model=Sequential()
model.add(Dense(16,input_shape=(8,), activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam',metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=200)
model.evaluate(X_train, y_train)[1]

y_pred=model.predict(X_test)
y_pred_class=np.argmax(y_pred, axis=1)
y_pred_class


# keras 4  0.78468
model=Sequential()
model.add(Dense(8,input_shape=(8,), activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(8, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam',metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=300)
model.evaluate(X_train, y_train)[1]

y_pred=model.predict(X_test)
y_pred_class=np.argmax(y_pred, axis=1)
y_pred_class












submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": y_pred_class
})
submission['Survived'] = submission['Survived'].astype(int)
submission

submission.to_csv('./submission.csv', index=False)
