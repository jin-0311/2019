# 출처 : 실전활용! 텐서플로 딥러닝 프로젝트
# https://wikibook.co.kr/tensorflow-projects/
# < CNN - 교통 표지판 인식

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



#