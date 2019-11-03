# 데이터 출처 :
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

df=pd.read_csv('./kaggle/pima/diabetes.csv')
df.head()

df.isnull().sum()
df.columns=df.columns.str.lower()
df.info()

# 산점도 scatter_matrix
df.columns[0]
pd.plotting.scatter_matrix(df.drop('outcome',axis=1), c=df[
    'outcome'], figsize=(15,15),  marker='o',  s=60, alpha=.8,
                           cmap='GnBu')

plt.close('all')
# glucose가 상대적으로 잘 분류되어 있으므로 특성 추가
df['glu1']=df['glucose']**2
df.loc[:,['glucose','glu1']]


# preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('outcome', axis=1), df['outcome'], stratify=df[
    'outcome'], random_state=1)
X_train.shape
X_test.shape
X_train.head()


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled =sc.transform(X_test)
X_train_scaled.shape
X_test_scaled.shape
X_train_scaled[0]

# rf  0.80729 0.739 {'max_depth': 3, 'n_estimators': 100}
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
param_grid={'n_estimators':[100,200,300,400],'max_depth':[1,2,3]}
grid=GridSearchCV(RandomForestClassifier(random_state=3, n_jobs=-1), param_grid, cv=5, return_train_score=True )
grid.fit(X_train, y_train)
grid.best_params_


grid.score(X_train, y_train)
grid.score(X_test, y_test)


# logreg 0.7812 0.7656

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)

param_grid={'C':[0.001, 0.01,0.1,1,10,100]}
grid=GridSearchCV(LogisticRegression(solver='liblinear', random_state=1), param_grid,cv=kfold,
                  return_train_score=True )
grid.fit(X_train_scaled, y_train)
grid.best_params_  # 0.1

grid.score(X_train_scaled, y_train)
grid.score(X_test_scaled, y_test)



# keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
X_train_scaled.shape
X_test_scaled.shape
X_train=X_train_scaled.reshape(576,9).astype('float32')
X_test=X_test_scaled.reshape(192,9).astype('float32')


y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


# model1 0.85 0.723
model=Sequential()
model.add(Dense(64, input_shape=(9,), activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train,validation_split=0.3, epochs=100)
model.evaluate(X_train, y_train)[1]
model.evaluate(X_test, y_test, verbose=0)[1]


# model20.907 0.710
model=Sequential()
model.add(Dense(128, input_shape=(9,), activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=100)

model.evaluate(X_train, y_train)[1]
model.evaluate(X_test, y_test, verbose=0)[1]


# model 3 0.864  0.744
model=Sequential()
model.add(Dense(32, input_shape=(9,), activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=200)

model.evaluate(X_train, y_train)[1]
model.evaluate(X_test, y_test, verbose=0)[1]


# model 4 0.782  0.757
model=Sequential()
model.add(Dense(16, input_shape=(9,), activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=200)

model.evaluate(X_train, y_train)[1]
model.evaluate(X_test, y_test, verbose=0)[1]


# model 5 0.796  0.757
model = Sequential()
model.add(Dense(16, input_shape=(9,), activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=200)

model.evaluate(X_train, y_train)[1]
model.evaluate(X_test, y_test, verbose=0)[1]

