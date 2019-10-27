# < con2d cat vs dog

import os , shutil
os.getcwd()
original_data_dir='./data/dog_cat/train'
base_dir='./data/cats_and_dogs_small'
# os.mkdir(base_dir)

train_dir=os.path.join(base_dir,'train')
# os.mkdir(train_dir)
validation_dir=os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir=os.path.join(base_dir,'test')
# os.mkdir(test_dir)

train_cats_dir=os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)
train_dogs_dir=os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)

validation_cats_dir=os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cats_dir)
validation_dogs_dir=os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dogs_dir)

test_cats_dir=os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)
test_dogs_dir=os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)
'''
# 고양이 1000개 train_cats_dir에 복사
fnames=['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src=os.path.join(original_data_dir, fname)
    dst=os.path.join(train_cats_dir, fname)
    shutil.copyfile(src,dst)

# 고양이 500개 validation_cats_dir에 복사
fnames=['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src=os.path.join(original_data_dir, fname)
    dst=os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src,dst)

# 고양이 500개 test_cats_dir에 복사
fnames=['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src=os.path.join(original_data_dir, fname)
    dst=os.path.join(test_cats_dir, fname)
    shutil.copyfile(src,dst)


# 강아지 1000개 train 복사
fnames=['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src=os.path.join(original_data_dir, fname)
    dst=os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src,dst)

# 강아지 500개 validation에 복사
fnames=['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src=os.path.join(original_data_dir, fname)
    dst=os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src,dst)

# 강아지 500개 test 에 복사
fnames=['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src=os.path.join(original_data_dir, fname)
    dst=os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src,dst)
'''

# count 확인하기
print('train_cat:', len(os.listdir(train_cats_dir)),'\n train_dog:', len(os.listdir(train_dogs_dir)))
print('validation_cat:', len(os.listdir(validation_cats_dir)),'\n validation_dog:', len(os.listdir(validation_dogs_dir)))
print('test_cat:', len(os.listdir(test_cats_dir)),'\n test_dog:', len(os.listdir(test_dogs_dir)))

# python generator 파이썬 제너레이터 == 반복자(iterator)처럼 작동하는 객체로, for~in 연산자에 사용가능
# yield를 사용하면 제너레이터 함수, 소괄호와 리스트내포 구문을 사용하면 제너레이터 표현식 이라고 함
def generator():
    i=0
    while True:
        i +=1
        yield i

for item in generator():
    print(item)
    if item > 4:
        break

# preprocessing the data
from keras.preprocessing.image import ImageDataGenerator # 전처리된 배치텐서로 자동으로 바꿔주는 파이썬 제너레이터를 만들어줌
train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255) # 1/255로 스케일 조정
train_generator=train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=20,
                                                  class_mode='binary') # 강아지1000, 고양이1000 -> 총 2000 배치 사이즈 20
validation_generator=test_datagen.flow_from_directory(validation_dir, target_size=(150,150), batch_size=20,
                                                      class_mode='binary') # 강아지 500 고양이 500 -> 총 1000, 배치사이즈 20

for data_batch, labels_batch in train_generator:
    print('배치 데이터 크기:', data_batch.shape, '\n 배치 레이블 크기:',labels_batch.shape)
    break



# model, binary classification -> sigmoid activation
# conv2d(output_depth, (window(==filter)_h, window_w) , input_shape는 임의로 150*150으로 지정

from keras import layers
from keras import models

model=models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))   # 74->36 -> 17 -> 7 까지 줄어듬
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

from keras import optimizers
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# ealry stopping, model check, tensorboard
import keras
callback_list=[keras.callbacks.EarlyStopping(monitor='val_acc', patience=1,),
               keras.callbacks.ModelCheckpoint(filepath='./cat_dog_model.h5',monitor='val_loss', save_best_only=True,
                                               )]#,
#               keras.callbacks.TensorBoard(log_dir='my_log_dir', histogram_freq=1,embeddings_freq=1,)]

history=model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator,
                            validation_steps=50, callbacks=callback_list)



# loss acc graph
import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1, len(acc)+1)

plt.plot(epochs, acc,'bo', label='training acc')
plt.plot(epochs, val_acc,'b', label='validation acc')
plt.title('training and validation acc')
plt.legend()

plt.close()

plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.legend()

plt.close()

# 텐서보드 실행할때 prompt ) tensorboard --logdir=my_log_dir , localhost::6006
# 그래프 그리기
from keras.utils import plot_model
plot_model(model, to_file='cat_dog_model.png', show_shapes=True )


# <  데이터 증식하기  - 랜덤변환 적용(데이터 수가 적을때 )
# ImageDataGenerator 사용
# fill_mode= 인접한 픽셀 사용,  reflect or wrap등이 있음 keras홈페이지 참조

datagen=ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
                           zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
# 그려보기
from keras.preprocessing import image
fnames=sorted([os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)])
img_path=fnames[1] # random selected picture
img=image.load_img(img_path, target_size=(150,150))
x=image.img_to_array(img) # 150,150,3 numpy array로 변환
x=x.reshape((1,)+ x.shape) # (1,150,150,3)으로 변환 -> 원본이미지

i=0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot=plt.imshow(image.array_to_img(batch[0]))
    i +=1
    if i%4 ==0:
        break
plt.show()
plt.close('all')


# < dropout and 데이터 증식 사용한 모델

model=models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))   # 74->36 -> 17 -> 7 까지 줄어듬
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

callback1=[keras.callbacks.EarlyStopping(monitor='val_acc', patience=2,),
           keras.callbacks.ModelCheckpoint(filepath='./cat_dog_model2.h5',monitor='val_loss',save_best_only=True,),
           keras.callbacks.TensorBoard(log_dir='my_log_dir',)]

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

train_datagen=ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)
test_datagen= ImageDataGenerator(rescale=1./255) # 절대로 검증/ 테스트 셋은 증식되어서는 안돼!

train_generator=train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary')
validation_generator=test_datagen.flow_from_directory(validation_dir, target_size=(150,150), batch_size=32,
                                                      class_mode='binary')


history=model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator,
                            validation_steps=500, callbacks=callback1)


# 텐서보드 실행할때 prompt )cd PycharmProjects/keras_DL  tensorboard --logdir=my_log_dir , localhost::6006
plot_model(model, to_file='cat_dog_model2.png', show_shapes=True )

import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1, len(acc)+1)

plt.plot(epochs, acc,'bo', label='training acc')
plt.plot(epochs, val_acc,'b', label='validation acc')
plt.title('training and validation acc')
plt.legend()

plt.close()

plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('training and validation loss')
plt.legend()

plt.close()





# < 사전훈련된 pretrained network  VGG
# 을 사용하려면 특성추출feature extraction, 미세조정 fine tuning 방법이 있음


# < 특성 추출
from keras.applications import VGG16
conv_base=VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
conv_base.summary()

# 데이터 증식을 사용하지 않는 빠른 특성 추출
import os
import numpy as np

base_dir='./data/cats_and_dogs_small'
train_dir=os.path.join(base_dir, 'train')
validation_dir=os.path.join(base_dir,'validation')
test_dir=os.path.join(base_dir, 'test')

datagen=ImageDataGenerator(rescale=1./255)
batch_size=20

def extract_features(directory, sample_count):
    features=np.zeros(shape=(sample_count, 4,4,512))
    labels=np.zeros(shape=(sample_count))
    generator=datagen.flow_from_directory(directory, target_size=(150,150), batch_size=batch_size, class_mode='binary')
    i=0
    for inputs_batch, labels_batch in generator:
        features_batch=conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size]=features_batch
        labels[i*batch_size: (i+1)*batch_size]=labels_batch
        i +=1
        if i *batch_size >= sample_count:
            break
    return features, labels

train_features, train_labels=extract_features(train_dir,2000)
validation_features, validation_labels=extract_features(validation_dir, 1000)
test_features, test_labels=extract_features(test_dir, 1000)

print(train_features.shape, train_labels.shape)  # (2000, 4, 4, 512) (2000,)


train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
print(train_features.shape, train_labels.shape)  # (2000, 4, 4, 512) (2000,)  -> (2000,8192) (2000,)


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))  # 8192
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])
import keras
callback2=[keras.callbacks.EarlyStopping(monitor='val_acc', patience=1,),
               keras.callbacks.ModelCheckpoint(filepath='./cat_dog_model.h5',monitor='val_loss', save_best_only=True,),
               keras.callbacks.TensorBoard(log_dir='my_log_dir', histogram_freq=1,)]

history = model.fit(train_features, train_labels, epochs=30, batch_size=20, validation_data=(validation_features, validation_labels),
                    callbacks=callback2)


# graph
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.close()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# < 미세 조정