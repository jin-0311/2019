# 데이터 출처 : https://data.seoul.go.kr/dataList/datasetView.do?infId=OA-15182&srvType=F&serviceKind=1&currentPageNo=1

import pandas as pd
import numpy as np

train=pd.read_csv('./Final_dataset/train/p_1911.csv')
train.head()
train.columns
train.index=train['Unnamed: 0']
train.index=pd.to_datetime(train.index)
del train['Unnamed: 0']

train1=train.iloc[:,[0,1,4,9,11,-1]]  # count, holiday, out_num, dust, mean_t, sun
train1.head()
train1.shape

test=pd.read_csv('./Final_dataset/test/p_1911.csv')
test.head()
test.columns
test.index=test['Unnamed: 0']
test.index=pd.to_datetime(test.index)
del test['Unnamed: 0']
test1=test.iloc[:,[0,1,4,9,11,-1]]
test1.shape
test1.head()

dataset=pd.concat([train1, test1], axis=0)
dataset.head()
dataset.tail()

values=dataset.values
groups=[0,1,2,3,4,5]
i=1

import matplotlib.pyplot as plt
# plt.figure()
# for group in groups:
    plt.subplot(len(groups),1,i)
    plt.plot(values[:,group])
    plt.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
# plt.show()
plt.close()

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars=1 if type(data) is list else data.shape[1]
    df=pd.DataFrame(data)
    cols, names=list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i== 0:
            names +=[('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t)' % (j+1, i)) for j in range(n_vars)]

    agg=pd.concat(cols, axis=1)
    agg.columns=names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

values=values.astype('float32')

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
scaled=scaler.fit_transform(values)
reframed=series_to_supervised(scaled, 1,1)
reframed.head()

values=reframed.values
values.shape


train=values[:365,:]
train.shape # 365 12
train_X, train_Y=train[:,:-1], train[:,-1]
train_X.shape # 365 11
train_Y.shape  # 365

test=values[365:,:]
test.shape  # 121 12

test_X, test_Y=test[:,:-1], test[:,-1]
test_X.shape # 121 11
test_Y.shape  # 121


train_X=train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
test_X=test_X.reshape((test_X.shape[0],1,test_X.shape[1]))

train_X.shape  # 365 1 11
test_X.shape  # 121 1 11





import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense,Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

np.random.seed(1)

model=Sequential()
model.add(LSTM(32, input_shape=(train_X.shape[1],train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.summary()

hist=model.fit(train_X, train_Y, epochs=100, verbose=2, validation_split=0.2, shuffle=False)

plt.plot(hist.history['loss'],label='train')
plt.plot(hist.history['val_loss'], label='valid')
plt.legend()
plt.show()
plt.close()
# train_X score
trainhat=model.predict(train_X)
train_X=train_X.reshape(train_X.shape[0], train_X.shape[2])

inv_trainhat=np.concatenate((trainhat, train_X[:,1:]), axis=1)
inv_trainhat=inv_trainhat[:,0]
inv_trainhat.shape

train_Y=train_Y.reshape(len(train_Y),1)
inv_train=np.concatenate((train_Y, train_X[:,1:]), axis=1)
inv_train=inv_train[:,0]
inv_train.shape

rmse=math.sqrt(mean_squared_error(inv_train, inv_trainhat))
print('test RMSE: %.3f' % rmse)



# make pred
yhat=model.predict(test_X)
test_X=test_X.reshape(test_X.shape[0], test_X.shape[2])

inv_yhat=np.concatenate((yhat, test_X[:,1:]), axis=1)
inv_yhat=scaler.inverse_transform(inv_yhat)
inv_yhat=inv_yhat[:,0]
inv_yhat

test_Y=test_Y.reshape(len(test_Y),1)
inv_y=np.concatenate((test_Y, test_X[:,1:]), axis=1)
inv_y=inv_y[:,0]
inv_y
rmse=math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('test RMSE: %.3f' % rmse)

train_Y.shape  # real train
test_Y.shape   # real test

yhat  # pred test
trainhat  # pred train


# plot

trainPP=np.empty_like(dataset)
trainPP[:,:]=np.nan
trainPP[1:len(trainhat)+1,:]= trainhat


testPP=np.empty_like(dataset)
testPP[:,:]=np.nan
testPP[len(trainhat):len(dataset)-1 ,:]= yhat


real=np.concatenate((train_Y, test_Y),axis=0)
real.shape

plt.close('all')
plt.plot(real, label='real')
plt.plot(trainhat, label='train prediction')
plt.plot(yhat, label='test prediction')

plt.legend(loc='best')
plt.show()


plt.plot(callBack.losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
