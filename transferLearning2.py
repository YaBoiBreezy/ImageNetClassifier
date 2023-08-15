#Transfer learning implementation by Alexander Breeze
#Using the pretrained efficientNet model, which comes from a model trained on the imagenet100 dataset
#I freeze the CNN layers and remove the DNN layers
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
from tensorflow.keras.applications import MobileNetV2
from scipy.stats import mode

class dataGenerator(keras.utils.Sequence):
    appendix={'daisy':0,'dandelion':1,'rose':2,'sunflower':3,'tulip':4}
    def __init__(self, dataIndices, train):
        self.dataIndices=dataIndices
        self.train=train
        self.batchSize=40    #RAM USAGE OF PRETRAINED MODEL IS PROPORTIONAL TO BATCH SIZE!!!
        self.threshold=224 #image shape
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
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    #for layer in base_model.layers: #freeze remaining layers to keep them from training
    #    layer.trainable = False

    model.add(base_model)
    model.add(keras.layers.GlobalAveragePooling2D())
    regularization_factor = 0.001  # You can adjust this value
    model.add(keras.layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(regularization_factor)))
    model.add(keras.layers.Dropout(0.5))  # Adding dropout for regularization
    model.add(keras.layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(regularization_factor)))
    model.add(keras.layers.Dropout(0.5))  # Adding dropout for regularization
    model.add(keras.layers.Dense(5, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
    #print(model.summary())
    return model

def evalAcc(models, val):
    numTotal = 0
    correct_common = 0
    correct_weighted = 0
    for batch in range(len(val)):
        data, labels = val.__getitem__(batch)
        batch_size = data.shape[0]
        predictions = np.zeros((batch_size, len(models), 5))
        for i, model in enumerate(models):
            predictions[:, i, :] = model.predict(data, verbose=0)
        majority_predictions, _ = mode(np.argmax(predictions, axis=2), axis=1)
        majority_predictions = majority_predictions.flatten()
        weighted_predictions = np.argmax(np.sum(predictions, axis=1), axis=1)
        true_labels = np.argmax(labels, axis=1)
        correct_common += np.sum(majority_predictions == true_labels)
        correct_weighted += np.sum(weighted_predictions == true_labels)
        numTotal += batch_size
    accuracy_common = (correct_common / numTotal) * 100
    accuracy_weighted = (correct_weighted / numTotal) * 100
    return accuracy_common, accuracy_weighted

def main():
    numDataPoints=2216
    trainSize=int(numDataPoints*0.8)
    dataIndices=np.random.permutation(numDataPoints)
    print("Making train")
    train=dataGenerator(dataIndices[:trainSize], True)
    print("Making val")
    val=dataGenerator(dataIndices[trainSize:], False)

    #'''
    print("Making model") #standard learning
    model=makeModel()
    print("Running model!")
    checkpoint = ModelCheckpoint(filepath='mobileNet_flower_model_weights.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)
    model.fit(train, validation_data=val, epochs=500, verbose=1, callbacks=[checkpoint])
    #'''

    models=[] #ensemble learning
    for numModels in range(1,100):  #NOTE: DON'T LOAD IN WEIghts for this, it will randomize the train and val sets so it will give overtrained accuracies
        print("Making model "+str(numModels))
        model=makeModel()
        model.fit(train, epochs=1, verbose=1)
        model.save(f"./multiModels/flower_model_{numModels}.h5")
        models.append(model)
        common, weighted = evalAcc(models, val)
        print(f"COMMON VAL ACCURACY: {common}%         WEIGHTED VAL ACCURACY: {weighted}%")

#currBest: 91% accuracy
main()