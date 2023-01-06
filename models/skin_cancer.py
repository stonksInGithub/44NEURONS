#!/usr/bin/env python

# coding: utf-8

# In[1]:

import pathlib

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

from PIL import Image

from tensorflow import keras

from keras import layers

from keras.models import Sequential

np.random.seed(42)

from keras.utils.np_utils import to_categorical

import sklearn

from sklearn.model_selection import train_test_split

from scipy import stats

from sklearn.preprocessing import LabelEncoder

from sklearn.utils import resample

from glob import glob

print("imported")

print(tf.__version__)

print(keras.__version__)

# In[3]:

scdf=pd.read_csv("E:\dataset\HAM10000_metadata.csv")

# In[4]:

l=LabelEncoder()

l.fit(scdf['dx'])

LabelEncoder()

print(list(l.classes_))

# In[5]:

scdf['label']=l.transform(scdf['dx'])

# In[6]:

print(scdf['label'].value_counts())

# In[7]:

df0=scdf[scdf['label']==0]

df1=scdf[scdf['label']==1]

df2=scdf[scdf['label']==2]

df3=scdf[scdf['label']==3]

df4=scdf[scdf['label']==4]

df5=scdf[scdf['label']==5]

df6=scdf[scdf['label']==6]

# In[8]:

n=5500

df0b=resample(df0,replace=True,n_samples=n,random_state=42)

df1b=resample(df1,replace=True,n_samples=n,random_state=42)

df2b=resample(df2,replace=True,n_samples=n,random_state=42)

df3b=resample(df3,replace=True,n_samples=n,random_state=42)

df4b=resample(df4,replace=True,n_samples=n,random_state=42)

df5b=resample(df5,replace=True,n_samples=n,random_state=42)

df6b=resample(df6,replace=True,n_samples=n,random_state=42)

# In[9]:

scdfb=pd.concat([df0b,df1b,df2b,df3b,df4b,df5b,df6b])

# In[10]:

print(scdfb['label'].value_counts())

# In[11]:

imgpath={os.path.splitext(os.path.basename(x))[0]: x

        for x in glob(os.path.join('E:\\dataset\\','*','*.jpg'))}

# In[12]:

scdfb['path']=scdf['image_id'].map(imgpath.get)

# In[13]:

scdfb['image']=scdfb['path'].map(lambda x: np.asarray(Image.open(x).resize((32,32))))

# In[14]:

x=np.asarray(scdfb['image'].tolist())

x=x/255

y=scdfb['label']

yc=to_categorical(y,num_classes=7)

trainx,testx,trainy,testy=train_test_split(x,yc,test_size=0.25,random_state=42)

nc=7

# In[15]:

model= tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(256,(3,3),activation="relu",input_shape=(32,32,3)))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu"))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu"))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(32))

model.add(tf.keras.layers.Dense(7,activation="softmax"))

model.summary()

# In[16]:

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['acc'])

# In[17]:

history=model.fit(trainx,trainy,epochs=70,batch_size=100,shuffle=True,validation_data=(testx,testy),verbose=2)

# In[18]:

loss=history.history['loss']

val_loss=history.history['val_loss']

epochs=range(1,len(loss)+1)

plt.plot(epochs,loss,'y',label='Training loss')

plt.plot(epochs,val_loss,'r',label='Validation loss')

plt.title("Losses")

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()

# In[19]:

acc=history.history['acc']

val_acc=history.history['val_acc']

plt.plot(epochs,acc,'y',label="Training accuracy")

plt.plot(epochs,val_acc,'r',label='Validation accuracy')

plt.title('Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()

# In[25]:

prediction=model.predict(testx)

print(prediction[1])

# In[26]:

# In[29]:

model.save('skin_cancer_detector.h5')

# In[ ]:

