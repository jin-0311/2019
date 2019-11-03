# 데이터 출처 : https://data.seoul.go.kr/dataList/datasetView.do?infId=OA-15182&srvType=F&serviceKind=1&currentPageNo=1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train=pd.read_csv('./Final_dataset/train/p_1911.csv')
train.head()
train.columns
train.index=train['Unnamed: 0']
train.index=pd.to_datetime(train.index)
del train['Unnamed: 0']
train1=train.iloc[:,0:1]
train1.shape

test=pd.read_csv('./Final_dataset/test/p_1911.csv')
test.head()
test.columns
test.index=test['Unnamed: 0']
test.index=pd.to_datetime(test.index)
del test['Unnamed: 0']
test1=test.iloc[:,0:1]
test1.shape
365+122

365/487

import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense,Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
'''
from keras.callbacks import Callback

class losses_callback(Callback):
    def init(self):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

callBack = losses_callback()
callBack.init()
'''

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

np.random.seed(1)

# 원래는 scale을 한번에 함
scaler=MinMaxScaler(feature_range=(0,1))
train1=scaler.fit_transform(train1)
test1=scaler.transform(test1)

dataset=np.concatenate((train1, test1), axis=0)
dataset  # 487 * 1
dataset.shape
dataset=dataset.astype('float32')


# < lstm 32
train_size=int(len(dataset)* 0.75)
train_size
test_size=len(dataset)-train_size
test_size

train, test=dataset[0:train_size,:], dataset[train_size:len(dataset),:]

train.shape
test.shape


look_back=7
trainX, trainY=create_dataset(train, 7)
testX, testY=create_dataset(test, 7)

trainX.shape  # 357 7 1
testX.shape   # 114 7 1

trainX=np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX=np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

trainX.shape    # 357 1 7
trainX
testX.shape    # 114 1 7

model=Sequential()
model.add(LSTM(32, input_shape=(None,look_back)))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='RMSprop')
model.summary()

hist=model.fit(trainX, trainY,validation_split=0.2, epochs=100, batch_size=7, shuffle=False, callbacks=[callBack])
train_score= model.evaluate(trainX, trainY)
train_score

'''
mse/adam : 0.019078 
mae/adam : 0.0895

mse/RMSprop: 0.0193
mae/RMSprop: 0.0914


'''


test_score=model.evaluate(testX, testY)
test_score

# scaled Prediction
trainPredict=model.predict(trainX)
trainPredict
testPredict=model.predict(testX)
testPredict

# reverse prediction
trainPredict=scaler.inverse_transform(trainPredict)
trainY=scaler.inverse_transform(trainY)

trainPredict
trainY


testPredict=scaler.inverse_transform(testPredict)
testY=scaler.inverse_transform(testY)

testPredict
testY



trainS=math.sqrt(mean_squared_error(trainY, trainPredict))
trainS
testS=math.sqrt(mean_squared_error(testY, testPredict))
print(trainS,testS)


trainPP=np.empty_like(dataset)
trainPP[:,:]=np.nan
trainPP[look_back:len(trainPredict)+look_back,:]= trainPredict

testPP=np.empty_like(dataset)
testPP[:,:]=np.nan
testPP[len(trainPredict)+(look_back*2)+1:len(dataset)-1 ,:]= testPredict



plt.close()
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPP)
plt.plot(testPP)

plt.legend(['real', 'train','test'],loc='best')
plt.show()


plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Train loss & Valid loss')
plt.show()

plt.close()

predTrain=pd.DataFrame(trainPredict)
predTrain.shape
realTrain=pd.DataFrame(trainY)
realTrain.shape

train_result=pd.concat([predTrain, realTrain], axis=1)
train_result.sample
train_result=train_result.astype('float32')
train_result.rename(columns={train_result.columns[0]:'Pred',
                            train_result.columns[1]:'Real'}, inplace=True)

train_result

plt.figure(figsize=(20,10))
plt.plot(train_result)

train_result['date']=pd.date_range('2017-09-01','2018-08-23')
train_result.index=train_result['date']
del train_result['date']

plt.close('all')

#<  lstm 64

train_size=int(len(dataset)* 0.75)
train_size
test_size=len(dataset)-train_size
test_size

train, test=dataset[0:train_size,:], dataset[train_size:len(dataset),:]

train.shape
test.shape


look_back=7
trainX, trainY=create_dataset(train, 7)
testX, testY=create_dataset(test, 7)

trainX.shape  # 357 7 1
testX.shape   # 114 7 1

trainX=np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX=np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

trainX.shape    # 357 1 7
trainX
testX.shape    # 114 1 7

model=Sequential()
model.add(LSTM(128, input_shape=(None,look_back)))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

hist=model.fit(trainX, trainY,validation_split=0.2, verbose=2, epochs=100, callbacks=[callBack])
model.evaluate(trainX, trainY)


'''
mse adam :  0.018
mse RMS:

mae adam
mae RMS

'''
loss=hist.history['loss']
val_loss=hist.history['val_loss']
epochs=range(1, len(loss)+1)
plt.figure()
plt.plot(epochs,loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='valid loss')
plt.title('training and validation loss')
plt.legend()
plt.show()

plt.close()



trainPredict=model.predict(trainX)
trainPredict
testPredict=model.predict(testX)
testPredict

trainPredict=scaler.inverse_transform(trainPredict)
trainY=scaler.inverse_transform(trainY)

trainPredict
trainY


testPredict=scaler.inverse_transform(testPredict)
testY=scaler.inverse_transform(testY)

testPredict
testY



trainS=math.sqrt(mean_squared_error(trainY, trainPredict))
trainS
testS=math.sqrt(mean_squared_error(testY, testPredict))
testS


trainPP=np.empty_like(dataset)
trainPP[:,:]=np.nan
trainPP[look_back:len(trainPredict)+look_back,:]= trainPredict

testPP=np.empty_like(dataset)
testPP[:,:]=np.nan
testPP[len(trainPredict)+(look_back*2)+1:len(dataset)-1 ,:]= testPredict



plt.close()
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPP)
plt.plot(testPP)

plt.legend(['real', 'train','test'],loc='best')
plt.show()


plt.plot(hist.history['loss'])
plt.title('loss')
plt.show()

#
loss=hist.history['loss']
val_loss=hist.history['val_loss']
epochs=range(1, len(loss)+1)
plt.figure()
plt.plot(epochs,loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='valid loss')
plt.title('training and validation loss')
plt.legend()
plt.show()


# < lstm stateful

train_size=int(len(dataset)* 0.75)
train_size
test_size=len(dataset)-train_size
test_size

train, test=dataset[0:train_size,:], dataset[train_size:len(dataset),:]

train.shape
test.shape


look_back=7
trainX, trainY=create_dataset(train, 7)
testX, testY=create_dataset(test, 7)

trainX.shape  # 357 7 1
testX.shape   # 114 7 1

trainX=np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX=np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

trainX.shape    # 357  7 1
testX.shape    # 114   7 1

early_stop=EarlyStopping()


model=Sequential()
model.add(LSTM(32, batch_input_shape=(1,look_back,1), stateful=True))
model.add(Dropout(0.2))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# 에포크 너무 많음
hist=model.fit(trainX, trainY, epochs=20,  batch_size=1,shuffle=False, validation_split=0.2) #, callbacks=[early_stop])

from keras.models import load_model
# model.save('stateful_model.h5')
# load할때
# model = load_model('stateful_model.h5')


trainScore=model.evaluate(trainX, trainY, batch_size=1, verbose=0)
trainScore # 0.0318  0.0425 이거말고 밑에

testScore=model.evaluate(testX, testY, verbose=0, batch_size=1)
testScore  # 0.19  0.1433

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])   # epoch 줄여야함
plt.close()


trainPredict=model.predict(trainX, batch_size=1)
trainPredict

testPredict=model.predict(testX, batch_size=1)
testPredict

trainPredict=scaler.inverse_transform(trainPredict)
trainY=scaler.inverse_transform(trainY)

trainPredict
trainY


testPredict=scaler.inverse_transform(testPredict)
testY=scaler.inverse_transform(testY)

testPredict
testY



trainS=math.sqrt(mean_squared_error(trainY, trainPredict))
trainS
testS=math.sqrt(mean_squared_error(testY, testPredict))
testS

# 7. 모델 사용하기
look_ahead = 250
xhat = x_test[0]
predictions = np.zeros((look_ahead, 1))
for i in range(look_ahead):
    prediction = model.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:], prediction])

plt.figure(figsize=(12, 5))
plt.plot(np.arange(look_ahead), predictions, 'r', label="prediction")
plt.plot(np.arange(look_ahead), y_test[:look_ahead], label="test function")
plt.legend()
plt.show()



loss=hist.history['loss']
val_loss=hist.history['val_loss']
epochs=range(1, len(loss)+1)
plt.figure()
plt.plot(epochs,loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='valid loss')
plt.title('training and validation loss')
plt.legend()
plt.show()


# < lstm stack stateful


train_size=int(len(dataset)* 0.75)
train_size
test_size=len(dataset)-train_size
test_size

train, test=dataset[0:train_size,:], dataset[train_size:len(dataset),:]

train.shape
test.shape


look_back=7
trainX, trainY=create_dataset(train, 7)
testX, testY=create_dataset(test, 7)

trainX.shape  # 357 7 1
testX.shape   # 114 7 1

trainX=np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX=np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

trainX.shape    # 357  7 1
testX.shape    # 114   7 1

model=Sequential()
# model.add(LSTM(32, batch_input_shape=(1,look_back,1), stateful=True, return_sequences=True))
model.add(LSTM(32, batch_input_shape=(1, look_back,1), return_sequences=True, stateful=True))
model.add(Dropout(0.2))
model.add(LSTM(32, batch_input_shape=(1, look_back,1), stateful=True))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# 에포크 너무 많음  ㅠ 짱오래걸림
hist=model.fit(trainX, trainY, validation_split=0.2, epochs=20, batch_size=1,shuffle=False, callbacks=[callBack])

from keras.models import load_model
# model.save('stateful_model.h5')
# load할때
# model = load_model('stateful_model.h5')


trainScore=model.evaluate(trainX, trainY, batch_size=1, verbose=0)
trainScore  # 0.029
testScore=model.evaluate(testX, testY, verbose=0, batch_size=1)
testScore # 0.1556

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.close()


trainPredict=model.predict(trainX, batch_size=1)

testPredict=model.predict(testX, batch_size=1)

trainPredict=scaler.inverse_transform(trainPredict)
trainY=scaler.inverse_transform(trainY)

testPredict=scaler.inverse_transform(testPredict)
testY=scaler.inverse_transform(testY)


trainS=math.sqrt(mean_squared_error(trainY, trainPredict))
trainS
testS=math.sqrt(mean_squared_error(testY, testPredict))
testS



trainPP=np.empty_like(dataset)
trainPP[:,:]=np.nan
trainPP[look_back:len(trainPredict)+look_back,:]= trainPredict

testPP=np.empty_like(dataset)
testPP[:,:]=np.nan
testPP[len(trainPredict)+(look_back*2)+1:len(dataset)-1 ,:]= testPredict



plt.close()
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPP)
plt.plot(testPP)

plt.legend(['real', 'train','test'],loc='best')
plt.show()

plt.plot(hist.history['loss'])


#
loss=hist.history['loss']
val_loss=hist.history['val_loss']
epochs=range(1, len(loss)+1)
plt.figure()
plt.plot(epochs,loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='valid loss')
plt.title('training and validation loss')
plt.legend()
plt.show()






