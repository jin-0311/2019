# 데이터 출처 : https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
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

df=pd.read_csv('./kaggle/cancer/data.csv')
df.head()

df.info()
del df['Unnamed: 32']
df.describe()

df.columns
df.isnull().sum()

df.loc[df['diagnosis']=='M', 'diagnosis']=1   # positive = cancer
df.loc[df['diagnosis']=='B', 'diagnosis']=0    # negative = non cancer

pd.value_counts(df['diagnosis'])
df.rename(columns={df.columns[1]:'target'}, inplace=True)
df.target



# pca 그래프 고치기
fig, axes=plt.subplots(15,2, figsize=(10,20))
m=df[df.target ==1]
b=df[df.target ==0]
del m['id']
del m['target']
del b['id']
del b['target']
m.head()

ax=axes.ravel()
data=df.copy()
del data['id']
del data['target']


for i in range(30):
    _, bins=np.histogram(data.iloc[:,i],bins=50)
    ax[i].hist(m.iloc[:,i], bins, color='Blue',alpha=.5)
    ax[i].hist(b.iloc[:,i], bins, color='Green',alpha=.5)
    ax[i].set_title(data.columns[i])
    ax[i].set_yticks(())
ax[0].set_xlabel('특성 크기')
ax[0].set_ylabel('빈도')
ax[0].legend(['m:positive','b:negative'], loc='best')

# radius_worst, concavity_mean, concave_points_worst 등이 구분이 잘됨
plt.close()

del df['id']

# preprocessing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], stratify=df['target'],
                                                    random_state=1)
X_train.shape
X_test.shape

from sklearn.model_selection import train_test_split
X_t_train, X_t_valid, y_t_train, y_t_valid= train_test_split(X_train, y_train, stratify=y_train, random_state=1)

# train, valid,
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_t_train_s=sc.fit_transform(X_t_train)
X_t_valid_s=sc.transform(X_t_valid)

# all train /test
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled =sc.transform(X_test)

X_train_scaled.shape
X_test_scaled.shape

X_t_train_s.shape
X_t_valid_s.shape


# rf 1.0  0.944
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
param_grid={'n_estimators':[100, 200,300,400,500]}
grid=GridSearchCV(RandomForestClassifier(random_state=3, n_jobs=-1), param_grid, cv=5, return_train_score=True )
grid.fit(X_train,y_train)
grid.best_params_

grid.score(X_train, y_train)
grid.score(X_test, y_test)


# logreg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)

param_grid={'C':[ 0.01,0.1,1,10]}
grid=GridSearchCV(LogisticRegression(solver='liblinear', random_state=1), param_grid,cv=kfold,
                  return_train_score=True )
# train, valid, test comparison
# X_t_train_s, X_t_valid_s, y_t_train, y_t_valid
grid.fit(X_t_train_s, y_t_train)
grid.best_params_    # 0.1

grid.score(X_t_train_s, y_t_train)  # 0.987
grid.score( X_t_valid_s, y_t_valid) # 1.0

grid.score(X_train_scaled, y_train) # 0.990

# train+valid -> fit
grid.fit(X_train_scaled, y_train)
grid.best_params_   #0.1

grid.score(X_train_scaled, y_train) #  0.990
grid.score(X_test_scaled, y_test)   #  0.96




# PCA + keras

# all train /test
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled =sc.transform(X_test)


X_train_scaled.shape
X_test_scaled.shape


from sklearn.decomposition import PCA
pca=PCA(random_state=1, n_components=3)
X_train_pca=pca.fit_transform(X_train_scaled)
X_test_pca=pca.transform(X_test_scaled)

X_train_pca.shape  #(426, 3)
X_test_pca.shape   # (143, 3)

X_train_pca[0]

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

X_train=X_train_pca.reshape(426,3).astype('float32')
X_test=X_test_pca.reshape(143,3).astype('float32')
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

X_train[0]
y_train[0]

# keras 1  0.962   0.958
model=Sequential()
model.add(Dense(16, input_shape=(3,), activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(16,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=100)
model.evaluate(X_train, y_train)[1]
model.evaluate(X_test, y_test)[1]


# keras2 0.960 0.958
model=Sequential()
model.add(Dense(16, input_shape=(3,), activation='tanh'))
model.add(Dropout(rate=0.5))
model.add(Dense(16,activation='tanh'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=100)
model.evaluate(X_train, y_train)[1]
model.evaluate(X_test, y_test)[1]


# keras 3  0.967 0.951
model=Sequential()
model.add(Dense(16, input_shape=(3,), activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(16,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(16,activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=100)
model.evaluate(X_train, y_train)[1]
model.evaluate(X_test, y_test)[1]



# keras 4   0.967  0.961
model=Sequential()
model.add(Dense(16, input_shape=(3,), activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(16,activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(16,activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=200)
model.evaluate(X_train, y_train)[1]
model.evaluate(X_test, y_test)[1]




# keras 6 0.970 0.944
model=Sequential()
model.add(Dense(16, input_shape=(3,), activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(16,activation='relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(16,activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_split=0.3, epochs=200)
model.evaluate(X_train, y_train)[1]
model.evaluate(X_test, y_test)[1]


