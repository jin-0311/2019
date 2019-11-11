'''
contents
0. 교통 표지판 인식 - Tensorflow       # 출처 : 실전활용! 텐서플로 딥러닝 프로젝트
1. 강아지 고양이 사진 구분 - Keras     # 출처 : 케라스 창시자에게 배우는 딥러닝

'''

# < 0. 교통 표지판 인식
# Glob

import glob
import pandas as pd
path='file path'
file_list=glob.glob(path + '//*.csv')
for file in file_list:
    data=pd.read_csv(file, encoding='utf8')
    print(data)

# 여러종류의 파일
images = glob.glob('*.JPG' or '*.jpg' or '*.png')

# CNN - 교통 표지판 인식
N_classes=43  # 43차원 배열
Resized_image=(32,32) # 32*32 사이즈로 조정

import matplotlib.pyplot as plt
import glob
from skimage.color import rgb2lab   # https://scikit-image.org/
from skimage.transform import resize
from collections import namedtuple  #
import numpy as np
np.random.seed(101)

# 모든 이미지 읽어서 크기 재조정, 회색으로 변환, 원핫인코딩
# dataset이라는 이름의 명명된 튜플사용할 것(named tuple)
Dataset=namedtuple('Dataset', ['X','y'])
def to_tf_format(imgs):
    return np.stack([img[:,:,np.newaxis] for img in imgs], axis=0).astype(np.float32)

def read_dataset_ppm(rootpath, n_labels, resize_to):
    images=[]
    labels=[]
    for c in range(n_labels):
        full_path=rootpath + '/' + format(c, '05d') + '/'
        for img_name in glob.glob(full_path + '*.ppm'):   # 이미지가 .ppm

            img=plt.imread(img_name).astype(np.float32)
            img=rgb2lab(img/255.0)[:,:,0]
            if resize_to :
                img=resize(img, resize_to, mode='reflect')
            label=np.zeros((n_labels,),dtype=np.float32)
            label[c] =1.0

            images.append(img.astype(np.float32))
            labels.append(label)
    return Dataset(X=to_tf_format(images).astype(np.float32),
                   y=np.matrix(labels).astype(np.float32))

dataset=read_dataset_ppm('GTSRB/Final_Training/Images', N_classes, Resized_image)
print(dataset.X.shape, dataset.y.shape)  # (39209, 32, 32, 1) (39209, 43)  -> 4차원, r:4) 39209개의 데이터, 32*32 크기,
# 1:회색조 / y:표본의 특징벡터(레이블이 43차원 =N_classes)

# 1번 데이터
plt.imshow(dataset.X[0,:,:,:].reshape(Resized_image))
print(dataset.y[0,:])  # 42개의 0, 맨처음만 1
# 마지막 데이터
plt.imshow(dataset.X[-1,:,:,:].reshape(Resized_image))
print(dataset.y[-1,:])  # 42개의 0, 맨 마지막만 1

from sklearn.model_selection import train_test_split
idx_train, idx_test=train_test_split(range(dataset.X.shape[0]), test_size=0.25, random_state=101) # 인덱스 뽑고
X_train=dataset.X[idx_train,:,:,:]  # 나눠주기
X_test=dataset.X[idx_test,:,:,:]
y_train=dataset.y[idx_train,:]
y_test=dataset.y[idx_test,:]

print(X_train.shape, y_train.shape)  # (29406, 32, 32, 1) (29406, 43)
print(X_test.shape, y_test.shape)   # (9803, 32, 32, 1) (9803, 43)

# 미니배치 생성 : 데이터 순서를 암기하는 것이아니라 입력-출력연결을 학습 , 데이터를 잘 섞어줌
def minibatcher(X,y,batch_size, shuffle):
    assert X.shape[0] == y.shape[0]      # assert는 뒤의 조건이 True가 아니면 AssertError를 발생한다.
    n_samples=X.shape[0]

    if shuffle :
        idx=np.random.permutation(n_samples)
    else:
        idx=list(range(n_samples))

    for k in range(int(np.ceil(n_samples/batch_size))):
        from_idx=k * batch_size
        to_idx=(k+1) * batch_size
        yield X[idx[from_idx:to_idx],:,:,:], y[idx[from_idx:to_idx],:]

# minibatcher test 배치사이즈가 10000일때 3개의 묶음이 나옴! 싱기해라
for mb in minibatcher(X_train,y_train,10000, True):
    print('X:', mb[0].shape)
    print('y:',mb[1].shape)

# model
import tensorflow as tf
def fc_no_activation_layer(in_tensors, n_units):  # (fc:fully connected)
    w=tf.get_variable('fc_W', [in_tensors.get_shape()[1],n_units], tf.float32, tf.contrib.layers.xavier_initializer())
    b=tf.get_variable('fc_B', [n_units,], tf.float32, tf.constant_initializer(0.0))
    return tf.matmul(in_tensors, w)+b

def fc_layer(in_tensors, n_units): # 전결합 계층
    return tf.nn.leaky_relu(fc_no_activation_layer(in_tensors, n_units))

def conv_layer(in_tensors, kernel_size, n_units):  # 입력데이터, 커널크기, 필터(또는 유닛)개수를 인수로 취하는 conv2d층
    w=tf.get_variable('conv_W',[kernel_size, kernel_size, in_tensors.get_shape()[3], n_units], tf.float32,
                      tf.contrib.layers.xavier_initializer())
    b=tf.get_variable('conv_B', [n_units,],tf.float32, tf.constant_initializer(0.0))
    return tf.nn.leaky_relu(tf.nn.conv2d(in_tensors,w,[1,1,1,1], 'SAME')+b)

def maxpool_layer(in_tensors, sampling): # 윈도우크기, 스트라이드 모두 정사각형
    return tf.nn.max_pool(in_tensors,[1,sampling, sampling, 1],[1,sampling, sampling, 1],'SAME')

def dropout(in_tensors, keep_proba, is_training):  # traning set에서만 사용할 드롭아웃 정의 (그래서 조건연산자 필요 lambda)
    return tf.cond(is_training, lambda :tf.nn.dropout(in_tensors, keep_proba), lambda :in_tensors)

# modeling
'''
1.2d conv 5*5, filter:32
2.2d conv 5*5, filter:64
3.flatten
4.fc, units:1024
5.dropout 0.6
6.fc, no_activation_function
7.softmax_activation -> output layer 

'''
def model(in_tensors, is_training):
    # l1 : 5*5 conv2d, 32filter, 2x:maxpooling, dropout0.8
    with tf.variable_scope('l1'):
        l1=maxpool_layer(conv_layer(in_tensors,5,32),2)
        l1_out=dropout(l1, 0.8, is_training)
    # l2: 5*5 conv2d, 64 filter, 2x:maxpooling, dropout0.8
    with tf.variable_scope('l2'):
        l2=maxpool_layer(conv_layer(l1_out, 5, 64),2)
        l2_out=dropout(l2, 0.8, is_training)

    with tf.variable_scope('flatten'):
        l2_out_flat=tf.layers.flatten(l2_out)
    # fc, units=1024, dropout0.6
    with tf.variable_scope('l3'):
        l3= fc_layer(l2_out_flat, 1024)
        l3_out=dropout(l3,0.6, is_training)
    # output
    with tf.variable_scope('out'):
        out_tensors=fc_no_activation_layer(l3_out, N_classes)
    return out_tensors

from sklearn.metrics import classification_report, confusion_matrix
def train_model(X_train, y_train, X_test, y_test, learning_rate, max_epochs, batch_size):
    in_X_tensors_batch = tf.placeholder(tf.float32, shape=(None, Resized_image[0],  Resized_image[1], 1))
    in_y_tensors_batch = tf.placeholder(tf.float32, shape=(None, N_classes))
    is_training=tf.placeholder(tf.bool)

    logits=model(in_X_tensors_batch, is_training)
    out_y_pred=tf.nn.softmax(logits)
    loss_score=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=in_y_tensors_batch)
    loss=tf.reduce_mean(loss_score)
    optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for epoch in range(max_epochs):
            print('epoch=', epoch)
            tf_score=[]
            for mb in minibatcher(X_train, y_train, batch_size, shuffle=True):
                tf_output= session.run([optimizer, loss], feed_dict={in_X_tensors_batch :mb[0],
                                                                     in_y_tensors_batch : mb[1],
                                                                     is_training : True})
                tf_score.append(tf_output[1])
            print('train_loss_score=', np.mean(tf_score))


        # test set session 안에 들어있어야함
        print('test set performance')
        y_test_pred, test_loss=session.run([out_y_pred, loss], feed_dict={in_X_tensors_batch : X_test,
                                                                          in_y_tensors_batch : y_test,
                                                                          is_training : False})

        # classification_report and confusion_matrix and log2 version of Confusion Matrix
        print('test_loss_score=', test_loss)
        y_test_pred_classified=np.argmax(y_test_pred, axis=1).astype(np.int32)
        y_test_true_classified=np.argmax(y_test, axis=1).astype(np.int32)
        print(classification_report(y_test_true_classified, y_test_pred_classified))

        cm=confusion_matrix(y_test_true_classified, y_test_pred_classified)

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.tight_layout()
        plt.title('cm')
        plt.show()

        # 잘못 분류한 내용을 강조하기 위한 log2 버전 CM의
        plt.imshow(np.log2(cm+1), interpolation='nearest', cmap=plt.get_cmap('tab20'))
        plt.colorbar()
        plt.tight_layout()
        plt.title('log2 ver. of CM')
        plt.show()


tf.reset_default_graph()
# 미니배치당 표본개수가 256개, 에포크는 10번
train_model(X_train, y_train, X_test, y_test, learning_rate=0.001, max_epochs=10, batch_size=256)





# < 1. 강아지 고양이 구분 하기
# 출처 : 케라스 창시자에게 배우는 딥러닝
# https://www.gilbut.co.kr/book/view?bookcode=BN002235

#  con2d cat vs dog

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


# 데이터 증식하기  - 랜덤변환 적용(데이터 수가 적을때 )
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


#  dropout and 데이터 증식 사용한 모델

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





#  사전훈련된 pretrained network  VGG
# 을 사용하려면 특성추출feature extraction, 미세조정 fine tuning 방법이 있음


#  특성 추출
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


#  미세 조정