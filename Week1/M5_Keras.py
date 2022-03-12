from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, AveragePooling2D, ReLU, BatchNormalization, Conv2D, Input, MaxPooling2D, Dropout, DepthwiseConv2D
from tensorflow.keras.models import load_model

from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers 
import matplotlib.pyplot as plt
import os, datetime
import pandas as pd
import shutil

train_data_dir='datasets/MIT_small_train_3/test'
val_data_dir='datasets/MIT_small_train_1/test'
test_data_dir='datasets/MIT_small_train_2/test'

val_len = 0
for elem in os.listdir(val_data_dir):
 val_len += len(os.listdir(os.path.join(val_data_dir, elem)))

img_width=256
img_height=256
batch_size=64
number_of_epoch=50
validation_samples=val_len

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_data_format()
    assert dim_ordering in {'channels_first', 'channels_last'}

    if dim_ordering == 'channels_first':
        # 'RGB'->'BGR'
        x = x[ ::-1, :, :]
        # Zero-center by mean pixel
        x[ 0, :, :] -= 103.939
        x[ 1, :, :] -= 116.779
        x[ 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x


datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    #preprocessing_function=preprocess_input,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=None)


train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
		
		
def conv_block(name, kernel, stride, num_filters, activation="relu"):

  mdl = Sequential(name=name)
  
  mdl.add(Conv2D(kernel_size=kernel,
            strides=stride,
            filters=num_filters,
            padding="same",
            activation=activation))
  mdl.add(BatchNormalization())

  return mdl


def depthwise_seperable_conv_block(name, stride, num_filters, activation="relu"):

  mdl = Sequential(name=name)

  mdl.add(DepthwiseConv2D(kernel_size=3, strides=stride, activation=activation, padding="same"))
  mdl.add(BatchNormalization())
  mdl.add(Conv2D(kernel_size=1,
            strides=stride,
            filters=num_filters,
            padding="same",
            activation=activation))
  mdl.add(BatchNormalization())

  return mdl
  

inputs = Input(shape=(img_height, img_width, 3))
num_filters = 32
"""
t = conv_block("conv_block1", 3, 1, num_filters)(inputs)
t = conv_block("conv_block2", 3, 1, num_filters*2)(t)
t = MaxPooling2D()(t)
t = conv_block("conv_block3", 3, 1, num_filters*4)(t)
t = MaxPooling2D()(t)
t = conv_block("conv_block4", 3, 1, num_filters*4)(t)
t = conv_block("conv_block5", 3, 1, num_filters*8)(t)

t = AveragePooling2D(4)(t)
t = Flatten()(t)
t =  Dense(512, activation="relu")(t)
outputs = Dense(8, activation='softmax')(t)
"""

t = conv_block("conv_block1", 3, 2, num_filters)(inputs)
t = depthwise_seperable_conv_block("depthwise_seperable_conv_block1", 1, num_filters)(t)
t = conv_block("conv_block2", 1, 1, num_filters*2)(t)
t = depthwise_seperable_conv_block("depthwise_seperable_conv_block2", 2, num_filters*2)(t)
t = conv_block("conv_block3", 1, 1, num_filters*4)(t)
t = depthwise_seperable_conv_block("depthwise_seperable_conv_block3", 1, num_filters*4)(t)
t = conv_block("conv_block4", 1, 1, num_filters*8)(t)
t = depthwise_seperable_conv_block("depthwise_seperable_conv_block4", 2, num_filters*8)(t)

t = AveragePooling2D(8)(t)
t = Flatten()(t)
t =  Dense(num_filters*8, activation="relu")(t)

outputs = Dense(8, activation='softmax')(t)


model = Model(inputs, outputs)

lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0008,
    decay_steps=1000,
    decay_rate=0.9) 

optimizer = optimizers.Adam(learning_rate=0.0008)


model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])
			  
model_name = input("Give me a name:")
plot_model(model, to_file=model_name + ".png", show_shapes=True)

csv_logger = CSVLogger(model_name + '.csv')
early_stop = EarlyStopping(patience=20, restore_best_weights=True)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit_generator(train_generator,
      steps_per_epoch=(int(validation_samples//batch_size)+1),
      epochs=100,
      validation_data=validation_generator,
      validation_steps= (int(validation_samples//batch_size)+1), callbacks=[tensorboard_callback, csv_logger])

result = model.evaluate_generator(test_generator)

model.save(model_name + ".h5")
