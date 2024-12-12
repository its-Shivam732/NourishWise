import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import numpy as np
import tensorflow as tf

# Run following comands to load dataset
# !mkdir -p ~/.kaggle
# !mv kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !pip install kaggle
# !kaggle datasets download -d dansbecker/food-101

import zipfile

with zipfile.ZipFile('food-101.zip', 'r') as zip_ref:  # Replace with 'food-101.zip' if using the alternative command
    zip_ref.extractall('food101')

from shutil import copy
from collections import defaultdict
import os

#function to help preparing tarining set and test set....ONLY RUN ONCE
def prepare_data(filepath, src,dest):
  classes_images = defaultdict(list)
  with open(filepath, 'r') as txt:
      paths = [read.strip() for read in txt.readlines()]
      for p in paths:
        food = p.split('/')
        classes_images[food[0]].append(food[1] + '.jpg')

  for food in classes_images.keys():
    print("\nCopying images into ",food)
    if not os.path.exists(os.path.join(dest,food)):
      os.makedirs(os.path.join(dest,food))
    for i in classes_images[food]:
      copy(os.path.join(src,food,i), os.path.join(dest,food,i))
  print("Copying Done!")

prepare_data('food101/food-101/food-101/meta/train.txt', 'food101/food-101/food-101/images', 'food101/food-101/food-101/train')
prepare_data('food101/food-101/food-101/meta/test.txt', 'food101/food-101/food-101/images', 'food101/food-101/food-101/test')


n_classes = 101
img_width, img_height = 224, 224
train_data_dir = 'food101/food-101/food-101/train'
validation_data_dir = 'food101/food-101/food-101/test'
nb_train_samples = 75750
nb_validation_samples = 25250
batch_size = 20


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


with tf.device('/device:GPU:0'):
    mbv2 = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = mbv2.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    predictions = Dense(101, kernel_regularizer=regularizers.l2(0.005), activation='softmax')(x)

    model = Model(inputs=mbv2.input, outputs=predictions)
    model.compile(
        optimizer=SGD(learning_rate=0.001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    checkpointer = ModelCheckpoint(filepath='best_model_3class_sept.keras', verbose=1, save_best_only=True)
    csv_logger = CSVLogger('history.log')

    history = model.fit(
        train_generator,
        steps_per_epoch=300,
        validation_data=validation_generator,
        validation_steps=50,
        epochs=51,
        verbose=1,
        callbacks=[csv_logger, checkpointer]
    )