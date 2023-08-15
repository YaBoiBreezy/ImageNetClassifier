#this is a variant that loads the database into memory, to handle the data we only use 5% of the database
#It should be able to quickly optimize hyperparameters

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import json
from PIL import Image
import os

class dataGenerator(keras.utils.Sequence): #disk-cached data generator, batchSize=1 since tensorflow can't handle images of different sizes in same batch
    with open('./imagenet100/Labels.json', 'r') as json_file:
        appendix = json.load(json_file)
    i=0
    for key,value in appendix.items():
        appendix[key]=(i,value)
        i+=1
    numClasses=len(appendix)
    
    def __init__(self, dirPaths):
        dataPaths=[]
        self.labels=[]
        for dirPath in dirPaths:
            for dirname, _, filenames in os.walk(dirPath):
                for filename in filenames:
                    filePath = os.path.join(dirname, filename)
                    dataPaths.append(filePath)
                    label=filePath.split('\\')[-2]
                    self.labels.append(dataGenerator.appendix[label][0])
        dataPaths=dataPaths[::100] #only every xth data value, so it fits in memory        2, 4, 5, 8, 10, 20, 25, 40
        self.labels=self.labels[::100]
        self.data=[]
        for filePath in dataPaths:
            img = Image.open(filePath)
            img_array = np.array(img)
            if img_array.ndim==2:
                img_array=np.stack((img_array,) * 3, axis=-1)
            if img_array.shape[-1]>3:
                img_array=img_array[:,:,:3]
            self.data.append(img_array)

        self.labels = np.array(self.labels)
        self.size = len(self.data)
        self.indices = np.random.permutation(self.size)
    
    def on_epoch_end(self):
        self.indices = np.random.permutation(self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        i=self.indices[index]
        data=self.data[i]
        data=np.array([data])
        labels=self.labels[i]
        labels=np.array([tf.one_hot(labels, depth=dataGenerator.numClasses)])
        return data, labels

def makeModel(hp):
    model = keras.Sequential()
    
    numConv=hp.Int('numConv', min_value=2, max_value=7, step=1)
    if numConv>0:
        model.add(keras.layers.Conv2D(hp.Choice('convF1', values=[16, 32, 64, 128, 256], ordered=True), hp.Choice('convK1', values=[3, 5, 7]), activation='elu', input_shape=(None, None, 3)))
        if hp.Boolean('pool1'):
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    if numConv>1:
        model.add(keras.layers.Conv2D(hp.Choice('convF2', values=[16, 32, 64, 128, 256], ordered=True), hp.Choice('convK2', values=[3, 5, 7]), activation='elu'))
        if hp.Boolean('pool2'):
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    if numConv>2:
        model.add(keras.layers.Conv2D(hp.Choice('convF3', values=[16, 32, 64, 128, 256], ordered=True), hp.Choice('convK3', values=[3, 5, 7]), activation='elu'))
        if hp.Boolean('pool3'):
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    if numConv>3:
        model.add(keras.layers.Conv2D(hp.Choice('convF4', values=[16, 32, 64, 128, 256], ordered=True), hp.Choice('convK4', values=[3, 5, 7]), activation='elu'))
        if hp.Boolean('pool4'):
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    if numConv>4:
        model.add(keras.layers.Conv2D(hp.Choice('convF5', values=[16, 32, 64, 128, 256], ordered=True), hp.Choice('convK5', values=[3, 5, 7]), activation='elu'))
        if hp.Boolean('pool5'):
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    if numConv>5:
        model.add(keras.layers.Conv2D(hp.Choice('convF6', values=[16, 32, 64, 128, 256], ordered=True), hp.Choice('convK6', values=[3, 5, 7]), activation='elu'))
        if hp.Boolean('pool6'):
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    if numConv>6:
        model.add(keras.layers.Conv2D(hp.Choice('convF7', values=[16, 32, 64, 128, 256], ordered=True), hp.Choice('convK7', values=[3, 5, 7]), activation='elu'))
        if hp.Boolean('pool7'):
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    global_pooling_type = hp.Choice('global_pooling_type', values=['avg', 'max'])
    if global_pooling_type == 'avg':
        model.add(keras.layers.GlobalAveragePooling2D())
    elif global_pooling_type == 'max':
        model.add(keras.layers.GlobalMaxPooling2D())

    numDense=hp.Int('numDense', min_value=1, max_value=3, step=1)
    if numDense>0:
        model.add(keras.layers.Dense(units=hp.Int('denseN1', min_value=32, max_value=512, step=32), activation=hp.Choice('denseA1', values=['relu', 'sigmoid', 'tanh'])))
        model.add(keras.layers.Dropout(hp.Float('denseD1', min_value=0.0, max_value=0.5, step=0.1)))
    if numDense>1:
        model.add(keras.layers.Dense(units=hp.Int('denseN2', min_value=32, max_value=512, step=32), activation=hp.Choice('denseA2', values=['relu', 'sigmoid', 'tanh'])))
        model.add(keras.layers.Dropout(hp.Float('denseD2', min_value=0.0, max_value=0.5, step=0.1)))
    if numDense>2:
        model.add(keras.layers.Dense(units=hp.Int('denseN3', min_value=32, max_value=512, step=32), activation=hp.Choice('denseA3', values=['relu', 'sigmoid', 'tanh'])))
        model.add(keras.layers.Dropout(hp.Float('denseD3', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(keras.layers.Dense(100, activation='softmax'))
    
    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def objectiveFunc(hp):
    train=dataGenerator(['./imagenet100/train.X1','./imagenet100/train.X2','./imagenet100/train.X3','./imagenet100/train.X4'])
    val=dataGenerator(['./imagenet100/val.X'])
    model=makeModel(hp)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
    model.fit(train, validation_data=val, epochs=500, callbacks=[early_stopping], verbose=1)
    _, valAcc = model.evaluate(val, verbose=0)
    print(f"Validation Accuracy: {valAcc}")
    return valAcc

def main():
    tuner = kt.BayesianOptimization(objectiveFunc, objective='val_accuracy', max_trials=20)
    tuner.search()
    best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]
    print(best_hps.values)
main()