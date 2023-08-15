#Transfer learning implementation by Alexander Breeze
#using the pretrained model in the model_weights.h5 file, which come from a model trained on the imagenet100 dataset
#to classify 130 000 images from 100 classes, I freeze the CNN layers and remove the DNN layers
#I then add a new dnn to learn classification on a separate dataset of 5 types of flowers

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LeakyReLU
import keras_tuner as kt
import json
from PIL import Image
import os
import gc
from skimage.transform import resize
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class dataGenerator(keras.utils.Sequence):
    appendix={'daisy':0,'dandelion':1,'rose':2,'sunflower':3,'tulip':4}
    def __init__(self, dataIndices, train):
        self.dataIndices=dataIndices
        self.train=train
        self.batchSize=128
        self.threshold=256 #image shape
        self.data = np.zeros((len(self.dataIndices), self.threshold, self.threshold, 3), dtype=np.uint8)  #specifying datatype and not normalizing to [0,1] to save RAM
        self.labels = np.zeros(len(self.dataIndices), dtype=np.uint8)

        i=-1
        pos=0
        for dirname, _, filenames in os.walk(".\\flowersDatabase\\"):
            for filename in filenames:
                i+=1
                if not i in self.dataIndices: #only take data from self.dataIndices list, so we separate train and val
                    continue
                filePath = os.path.join(dirname, filename)

                img = np.array(Image.open(filePath))
                if img.ndim==2:
                    img=np.stack((img,) * 3, axis=-1)
                if img.shape[-1]==4:
                    img=img[:,:,:3]

                scaling_factor = self.threshold / max(img.shape[0], img.shape[1])
                img = resize(img, (int(img.shape[0] * scaling_factor), int(img.shape[1] * scaling_factor), 3), order=1)
                img = (img * 255).astype(np.uint8)
                canvas = np.zeros((self.threshold, self.threshold, 3), dtype=np.uint8)
                pad_top = (self.threshold - img.shape[0]) // 2
                pad_bottom = pad_top + img.shape[0]
                pad_left = (self.threshold - img.shape[1]) // 2
                pad_right = pad_left + img.shape[1]
                canvas[pad_top:pad_bottom, pad_left:pad_right, :] = img
                self.data[pos] = canvas
                label=filePath.split('\\')[-2]
                self.labels[pos]=dataGenerator.appendix[label]
                pos+=1

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.size = self.data.shape[0]
        self.indices = np.random.permutation(self.size)
    
    def on_epoch_end(self):
        self.indices = np.random.permutation(self.size)

    def __len__(self):
        return int(self.size/self.batchSize) +1 #add in partial batch at end

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batchSize : (index + 1) * self.batchSize]
        batch_data = self.data[batch_indices] / 255.0  # Convert back to floats in [0, 1] range
        batch_labels = np.array([tf.one_hot(self.labels[i], 5) for i in batch_indices])
        return batch_data, batch_labels

def makeModel():
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(96, kernel_size=(11, 11), strides=(5, 5), activation='LeakyReLU', input_shape=(500, 500, 3)))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(keras.layers.Conv2D(256, kernel_size=(5, 5), activation='LeakyReLU'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(keras.layers.Conv2D(384, kernel_size=(3, 3), activation='LeakyReLU'))
    model.add(keras.layers.Conv2D(384, kernel_size=(3, 3), activation='LeakyReLU'))
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation='LeakyReLU'))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(2048, activation='LeakyReLU'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2048, activation='LeakyReLU'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(100, activation='softmax'))

    model.load_weights("model_weights.h5") #load in saved weights

    model.pop()  # Remove the output layer
    model.pop()  # Remove the last dropout layer
    model.pop()  # Remove the last dense layer
    model.pop()  # Remove the second dropout layer
    model.pop()  # Remove the second dense layer

    for layer in model.layers: #freeze remaining layers to keep them from training
        layer.trainable = False

    #add new classification layers
    model.add(keras.layers.Dense(1000, activation='LeakyReLU'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1000, activation='LeakyReLU'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(5, activation='softmax'))
    
    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def main():
    numDataPoints=2746
    trainSize=int(numDataPoints*0.8)
    dataIndices=np.random.permutation(numDataPoints)
    print("Making train")
    train=dataGenerator(dataIndices[:trainSize], True)
    print("Making val")
    val=dataGenerator(dataIndices[trainSize:], False)
    print("Making model")
    model=makeModel()
    print("Running model!")
    checkpoint = ModelCheckpoint(filepath='flower_model_weights.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)
    model.fit(train, validation_data=val, epochs=500, verbose=1, callbacks=[checkpoint])

main()