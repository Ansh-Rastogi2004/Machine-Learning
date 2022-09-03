from tensorflow.keras.applications.inception_v3 import InceptionV3
import urllib.request
from tensorflow.keras import layers
import os
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing import image
import glob

# Sets all unwanted warnings as OFF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=RMSprop(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['acc'])

training_dir = 'horse-or-human/training/'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(training_dir,
                                                    batch_size=20,
                                                    target_size=(150, 150),
                                                    class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=7,
                    verbose=1)

path = glob.glob("C:/Users/Kiwi/PycharmProjects/Machine Learning/horse-or-human/test/*.jpg")

for file in path:
    # Load the image
    img = image.load_img(file, target_size=(150, 150))
    # Change img into 2D array
    x = image.img_to_array(img)
    #Now because we r working with 3D arrays set the 3rd dimensiion as 0
    x = np.expand_dims(x, axis=0)
    # Stack the array in a vertical style
    image_tensor = np.vstack([x])
    # Use predict command to run
    classes = model.predict(image_tensor)
    if classes[0] > 0.5:
        print(" img is a human")
    else:
        print("img is a horse")
