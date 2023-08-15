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
import warnings

class dataGenerator(keras.utils.Sequence):
    with open('./imagenet100/Labels.json', 'r') as json_file:
        appendix = json.load(json_file)
    i=0
    for key,value in appendix.items():
        appendix[key]=(i,value)
        i+=1
    numClasses=len(appendix)
    
    def __init__(self, dirPaths,val):
        self.val=val
        self.dirPaths=dirPaths
        self.batchSize=100*50
        self.dataPaths=[]
        self.labels=[]
        for dirPath in self.dirPaths:
            for dirname, _, filenames in os.walk(dirPath):
                for filename in filenames:
                    filePath = os.path.join(dirname, filename)
                    self.dataPaths.append(filePath)
                    label=filePath.split('\\')[-2]
                    self.labels.append(dataGenerator.appendix[label][0])
        self.dataPaths = np.array(self.dataPaths)
        self.labels = np.array(self.labels)
        self.size = self.dataPaths.shape[0]
        self.indices = np.random.permutation(self.size)
        self.datagen = ImageDataGenerator(
            rotation_range=20,  # Random rotation within [-20, 20] degrees
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,  # Random horizontal shift within [-0.1, 0.1] of image width
            height_shift_range=0.1,  # Random vertical shift within [-0.1, 0.1] of image height
        )
    
    def on_epoch_end(self):
        self.indices = np.random.permutation(self.size)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        data=[]
        labels=[]
        for i in self.indices[index*self.batchSize:(index+1)*self.batchSize]:
            img = np.array(Image.open(self.dataPaths[i]))
            if img.ndim==2:
                img=np.stack((img,) * 3, axis=-1)
            if img.shape[-1]==4:
                img=img[:,:,:3]

            if not self.val: #do data augmentation if not validation set
                img = self.datagen.random_transform(img)

            threshold=256
            scaling_factor = threshold / max(img.shape[0], img.shape[1])
            img = resize(img, (int(img.shape[0] * scaling_factor), int(img.shape[1] * scaling_factor), 3), order=1)

            canvas = np.zeros((threshold, threshold, 3))
            pad_top = (threshold - img.shape[0]) // 2
            pad_bottom = pad_top + img.shape[0]
            pad_left = (threshold - img.shape[1]) // 2
            pad_right = pad_left + img.shape[1]
            canvas[pad_top:pad_bottom, pad_left:pad_right, :] = img
            img=canvas / np.max(canvas)

            data.append(img)
            labels.append(self.labels[i])
        data=np.array(data)
        labels=np.array(labels)
        return data, labels

def makeModel():
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(96, kernel_size=(11, 11), strides=(5, 5), activation='LeakyReLU', input_shape=(None, None, 3)))
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

    model.load_weights("model_weights.h5")
    
    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def calculate_top5_error(predictions, ground_truth_labels):
    top5_errors = 0
    total_images = len(predictions)

    for pred_probs, true_label in zip(predictions, ground_truth_labels):
        top5_indices = pred_probs.argsort()[-5:][::-1]
        if true_label not in top5_indices:
            top5_errors += 1

    top5_error_rate = (top5_errors / total_images) * 100
    return top5_error_rate

def main():
    val=dataGenerator(['./imagenet100/val.X'], True)
    data, labels = val.__getitem__(0)
    print("GOT DATA")
    model=makeModel()
    pred=model.predict(data)
    top5_error_rate = calculate_top5_error(pred, labels)
    print(f"Top-5 Error Rate: {top5_error_rate}%")
    

main()