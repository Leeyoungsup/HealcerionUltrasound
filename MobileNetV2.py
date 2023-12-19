import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
import tensorflow.keras as K
from keras.applications import MobileNetV2
from tensorflow.keras import datasets, layers, models, losses, Model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pre
from glob import glob
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

epochs = 50
size = 224


train_df = pd.read_csv(
    '../../data/classificationDDH/train_aug_classification_dataset.csv')
val_df = train_df.loc[train_df['case'].isin([2, 3])]
train_df = train_df.loc[~train_df['case'].isin([2, 3])]
train_img_list = train_df['file name'].to_list()
train_label_list = train_df['standard class'].to_list()
train_case_list = train_df['case'].to_list()
img_path = '../../data/classificationDDH/aug_dataset/train/'
val_img_list = val_df['file name'].to_list()
val_label_list = val_df['standard class'].to_list()
val_case_list = val_df['case'].to_list()

x_train = np.zeros((len(train_img_list), size, size, 3), dtype=np.uint8)
for i in tqdm(range(len(train_img_list))):
    x_train[i] = np.array(Image.open(
        img_path+str(train_case_list[i])+'/'+train_img_list[i]).resize((size, size)))
y_train = np.array(train_label_list)

x_val = np.zeros((len(val_img_list), size, size, 3), dtype=np.uint8)
for i in tqdm(range(len(val_img_list))):
    x_val[i] = np.array(Image.open(
        img_path+str(val_case_list[i])+'/'+val_img_list[i]).resize((size, size)))
y_val = np.array(val_label_list)


def batch_generator(image, label, batchsize):
    N = len(image)
    indices = np.arange(N)  # 0부터 N-1까지의 인덱스 배열 생성
    np.random.shuffle(indices)  # 인덱스 배열을 무작위로 섞음

    i = 0
    while True:
        if i + batchsize <= N:
            batch_indices = indices[i:i+batchsize]
            i = i + batchsize
        else:  # 남은 데이터가 batchsize보다 작을 때, 배열을 wrap around하여 다시 섞음
            batch_indices = np.concatenate(
                (indices[i:], indices[:batchsize - (N - i)]))
            i = batchsize - (N - i)

            np.random.shuffle(indices)  # 다음 에포크를 위해 인덱스 배열을 무작위로 섞음

        yield image[batch_indices], label[batch_indices]


batch_size = 32
checkpoint_filepath = "../../model/classification/MobileNetV2_checkpoints.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True
)
# Compute class weights
class_weight_ratio = compute_class_weight(class_weight="balanced",
                                          classes=np.unique(y_train),
                                          y=y_train)
class_weight = {0: class_weight_ratio[0], 1: class_weight_ratio[1]}

# Create the model
input_t = K.Input(shape=(size, size, 3))
input_tensor = layers.experimental.preprocessing.Resizing(size, size, interpolation="bilinear",
                                                          input_shape=(size, size, 3))(input_t)
ResNet = MobileNetV2(include_top=True, weights='imagenet',
                     input_tensor=input_tensor)
model = K.Sequential()
model.add(ResNet)
model.add(tf.keras.layers.Dropout(0.2, input_shape=(64,)))
model.add(K.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.1)))
model.add(K.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=K.optimizers.AdamW(lr=2e-5),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=["accuracy"])

histo = model.fit(
    batch_generator(x_train, y_train, 32),
    validation_data=(x_val, y_val),
    epochs=500,
    steps_per_epoch=len(x_train)//32,
    callbacks=[model_checkpoint_callback],
    shuffle=True,
    class_weight=class_weight
)

model.save('../../model/classification/MobileNetV2.h5')
