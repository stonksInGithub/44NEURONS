# %%
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model

print(tf.__version__)

# %%
labels = ['PNEUMONIA', 'NORMAL']
batch_size = 16

train_datagen = ImageDataGenerator(
    rotation_range = 0.2,
    shear_range=0.2
)
test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    r'\train',
    target_size=(320, 320),  
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    r'\val',
    target_size=(320, 320),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    r'\test',
    target_size=(320, 320),
    batch_size=batch_size,
    class_mode='binary'
)

# %%
labels = train_generator.labels
N = labels.shape[0]

positive_frequencies = np.sum(labels, axis=0) / N
negative_frequencies = 1 - positive_frequencies

pos_weights = negative_frequencies
neg_weights = positive_frequencies
pos_contribution = positive_frequencies * pos_weights 
neg_contribution = negative_frequencies * neg_weights

# %%
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(320,320,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation="relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# %%
model.summary()

# %%
history = model.fit(train_generator, 
    validation_data=validation_generator,
    steps_per_epoch = len(train_generator.labels)//batch_size,
    validation_steps= len(validation_generator.labels)//batch_size,
    epochs=10,
    verbose=1,
    class_weight = {0:neg_weights, 1:pos_weights}
)
# %%
model.save("model.h5")
