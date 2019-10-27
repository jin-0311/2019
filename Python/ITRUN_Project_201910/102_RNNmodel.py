import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform
from matplotlib import font_manager, rc
import seaborn as sns


plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
else:
    print('Unknown system... sorry~~~~')

a=pd.read_csv('./102/data/102df_real_final.csv')
a.head()

a.index=a['Unnamed: 0']
a['date']=a.index
a.index=a['date']
a.index = pd.to_datetime(a.index, format='%Y-%m-%d')
del a['Unnamed: 0']

a.info()
a['count']=a['count'].astype(int)
plt.figure(figsize=(20,10))
plt.plot(a.index,'count',data=a)
plt.xlabel('날짜')
plt.ylabel('일별 대여량')
plt.title('102번 망원역 1번 출구 앞')  # 2015-09-09 15:19

plt.close()




del a['date']


import math
from keras.models import Sequential
from keras.layers import Dense,Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# setting the data

df=a.copy()
df.head()
df.tail()
np.random.seed(1)
scaler=MinMaxScaler(feature_range=(0,1))
dataset=scaler.fit_transform(df)

dataset.shape # 1193, 1
dataset

dataset=dataset.astype('float32')



# < lstm 32, look_back 7/ 30
train_size=int(len(dataset)* 0.75) # 864
test_size=len(dataset)-train_size # 299

train, test=dataset[0:train_size,:], dataset[train_size:len(dataset),:]


look_back=7
trainX, trainY=create_dataset(train, 7)
testX, testY=create_dataset(test, 7)

trainX=np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX=np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))






model=Sequential()
model.add(LSTM(32, input_shape=(None,look_back)))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['f1_score'])
model.summary()

hist=model.fit(trainX, trainY,validation_split=0.2,epochs=50)
model.evaluate(trainX, trainY)   #  0.024381034732134026 : loss


loss=hist.history['loss']
val_loss=hist.history['val_loss']
epochs=range(1, len(loss)+1)
plt.figure()
plt.plot(epochs,loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('LSTM 32 (7일) : Training and Validation LOSS')
plt.grid()
plt.legend()
plt.show()

plt.close()

model.save('lstm32_7.h5')

trainPredict=model.predict(trainX)
testPredict=model.predict(testX)
trainPredict.shape
trainPredict
inv_trainPredict.shape
inv_trainPredict=scaler.inverse_transform(trainPredict)
inv_trainY=scaler.inverse_transform(trainY)
inv_testPredict=scaler.inverse_transform(testPredict)
inv_testY=scaler.inverse_transform(testY)


trainPP=np.empty_like(dataset)
trainPP[:,:]=np.nan
trainPP[look_back:len(trainPredict)+look_back,:]= inv_trainPredict
testPP=np.empty_like(dataset)
testPP[:,:]=np.nan
testPP[len(trainPredict)+(look_back*2)+1:len(dataset)-1 ,:]= inv_testPredict


plt.plot(scaler.inverse_transform(dataset), 'yellowgreen')
plt.plot(trainPP,'red')
plt.plot(testPP, 'blue')
plt.grid()
plt.xlabel('날짜')
plt.ylabel('대여량')
plt.title('LSTM 32 (7일)')
plt.legend(['Real Data', 'Training Data','Test Data'],loc='best')
plt.show()

plt.close()



trainS=math.sqrt(mean_squared_error(inv_trainY, inv_trainPredict))
testS=math.sqrt(mean_squared_error(inv_testY, inv_testPredict))
print(trainS, testS)
# look 7     1.8737313110966796 2.2490726255205105
# look30     1.6775675325401687 2.0960762424143047



# < lstm stateful , look_back = 30

train_size = int(len(dataset) * 0.75)  # 864
test_size = len(dataset) - train_size  # 299

train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

look_back = 30
trainX, trainY = create_dataset(train, 30)
testX, testY = create_dataset(test, 30)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))




trainX=np.reshape(trainX, (trainX.shape[0], trainX.shape[2], 1))
testX=np.reshape(testX, (testX.shape[0], testX.shape[2], 1))


model=Sequential()
model.add(LSTM(32, batch_input_shape=(30,look_back,1), stateful=True))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


hist=model.fit(trainX, trainY, epochs=10,  batch_size=30,shuffle=False, validation_split=0.2)


loss=hist.history['loss']
val_loss=hist.history['val_loss']
epochs=range(1, len(loss)+1)
plt.figure()
plt.plot(epochs,loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('LSTM stateful (30일) : Training and Validation LOSS')
plt.grid()
plt.legend()
plt.show()

plt.close()

trainPredict=model.predict(trainX, batch_size=1)
testPredict=model.predict(testX, batch_size=1)
inv_trainPredict=scaler.inverse_transform(trainPredict)
inv_trainY=scaler.inverse_transform(trainY)
inv_testPredict=scaler.inverse_transform(testPredict)
inv_testY=scaler.inverse_transform(testY)


trainPP=np.empty_like(dataset)
trainPP[:,:]=np.nan
trainPP[look_back:len(trainPredict)+look_back,:]= inv_trainPredict
testPP=np.empty_like(dataset)
testPP[:,:]=np.nan
testPP[len(trainPredict)+(look_back*2)+1:len(dataset)-1 ,:]= inv_testPredict


plt.plot(scaler.inverse_transform(dataset), 'yellowgreen')
plt.plot(trainPP,'red')
plt.plot(testPP, 'blue')
plt.grid()
plt.xlabel('날짜')
plt.ylabel('대여량')
plt.title('LSTM stateful (30일)')
plt.legend(['Real Data', 'Training Data','Test Data'],loc='best')
plt.show()
plt.close()


trainS=math.sqrt(mean_squared_error(inv_trainY, inv_trainPredict))
testS=math.sqrt(mean_squared_error(inv_testY, inv_testPredict))
print(trainS, testS)

# look30     1.6775675325401687 2.0960762424143047

# model.save('lstm stateful_301.h5')



# < lstm stack stateful look_back=30

train_size = int(len(dataset) * 0.75)  # 864
test_size = len(dataset) - train_size  # 299

train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

look_back = 30
trainX, trainY = create_dataset(train, 30)
testX, testY = create_dataset(test, 30)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


trainX=np.reshape(trainX, (trainX.shape[0], trainX.shape[2], 1))
testX=np.reshape(testX, (testX.shape[0], testX.shape[2], 1))



model=Sequential()
model.add(LSTM(32, batch_input_shape=(1, look_back,1), return_sequences=True, stateful=True))
model.add(Dropout(0.3))
model.add(LSTM(32, batch_input_shape=(1, look_back,1), return_sequences=True, stateful=True))
model.add(Dropout(0.3))
model.add(LSTM(32, batch_input_shape=(1, look_back,1), stateful=True))
model.add(Dropout(0.3))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


hist=model.fit(trainX, trainY, validation_split=0.2, epochs=10, batch_size=1,shuffle=False )


trainPredict = model.predict(trainX, batch_size=1)
testPredict = model.predict(testX, batch_size=1)

inv_trainPredict=scaler.inverse_transform(trainPredict)
inv_trainY=scaler.inverse_transform(trainY)
inv_testPredict=scaler.inverse_transform(testPredict)
inv_testY=scaler.inverse_transform(testY)


loss=hist.history['loss']
val_loss=hist.history['val_loss']
epochs=range(1, len(loss)+1)
plt.figure()
plt.plot(epochs,loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('LSTM stack and stateful (30일) : Training and Validation LOSS')
plt.grid()
plt.legend()
plt.show()

plt.close()



trainPP=np.empty_like(dataset)
trainPP[:,:]=np.nan
trainPP[look_back:len(trainPredict)+look_back,:]= inv_trainPredict
testPP=np.empty_like(dataset)
testPP[:,:]=np.nan
testPP[len(trainPredict)+(look_back*2)+1:len(dataset)-1 ,:]= inv_testPredict


plt.plot(scaler.inverse_transform(dataset), 'yellowgreen')
plt.plot(trainPP,'red')
plt.plot(testPP, 'blue')
plt.grid()
plt.xlabel('날짜')
plt.ylabel('대여량')
plt.title('LSTM stack and stateful (30일)')
plt.legend(['Real Data', 'Training Data','Test Data'],loc='best')
plt.show()
plt.close()


trainS=math.sqrt(mean_squared_error(inv_trainY, inv_trainPredict))
testS=math.sqrt(mean_squared_error(inv_testY, inv_testPredict))
print(trainS, testS)
# look 7        2.78897239902827 2.703
# look 30       2.500087545769394 2.4463949032118544

# model.save('stack_state30.h5')



# < gru 32 - 진행중 : error 값 높음
train_size = int(len(dataset) * 0.75)  # 864
test_size = len(dataset) - train_size  # 299

train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

look_back = 7
trainX, trainY = create_dataset(train, 7)
testX, testY = create_dataset(test, 7)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# model
model=Sequential()
model.add(GRU(32, dropout=0.3, recurrent_dropout=0.2, input_shape=(None, 7)))
model.add(Dense(1))

model.compile(optimizer='adam',  loss='mse')
model.summary()
hist=model.fit(trainX, trainY, epochs=20,  batch_size=1,shuffle=False, validation_split=0.2)


trainPredict = model.predict(trainX, batch_size=1)
testPredict = model.predict(testX, batch_size=1)
inv_trainPredict=scaler.inverse_transform(trainPredict)
inv_trainY=scaler.inverse_transform(trainY)
inv_testPredict=scaler.inverse_transform(testPredict)
inv_testY=scaler.inverse_transform(testY)



loss=hist.history['loss']
val_loss=hist.history['val_loss']
epochs=range(1, len(loss)+1)
plt.figure()
plt.plot(epochs,loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('GRU (7일) : Training and Validation LOSS')
plt.grid()
plt.legend()
plt.show()

plt.close()



trainPP=np.empty_like(dataset)
trainPP[:,:]=np.nan
trainPP[look_back:len(trainPredict)+look_back,:]= inv_trainPredict
testPP=np.empty_like(dataset)
testPP[:,:]=np.nan
testPP[len(trainPredict)+(look_back*2)+1:len(dataset)-1 ,:]= inv_testPredict


plt.plot(scaler.inverse_transform(dataset), 'yellowgreen')
plt.plot(trainPP,'red')
plt.plot(testPP, 'blue')
plt.grid()
plt.xlabel('날짜')
plt.ylabel('대여량')
plt.title('GRU (7일)')
plt.legend(['Real Data', 'Training Data','Test Data'],loc='best')
plt.show()
plt.close()


trainS=math.sqrt(mean_squared_error(inv_trainY, inv_trainPredict))
testS=math.sqrt(mean_squared_error(inv_testY, inv_testPredict))
print(trainS, testS)
# look 7
# look 30



