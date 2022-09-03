import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import glob


# Downloaded zip files and stored them in directory
local_zip = 'rps.zip'
learn_dir = 'multiclass/'

local_zip_1 = 'rps-test-set.zip'
test_dir = 'multiclass/'

# Defined directories
rock_dir = os.path.join('multiclass/rps/rock')
paper_dir = os.path.join('multiclass/rps/paper')
scissors_dir = os.path.join('multiclass/rps/scissors')

# Printed amount of files
print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

# Printed first 10 items
rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

# Instructions for training directory
TRAINING_DIR = "multiclass/rps/"
training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical'
)

# Instructions for Validation directory
VALIDATION_DIR = "multiclass/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)


validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical'
)

# CNN model
model = tf.keras.models.Sequential([
    # Input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN (Dense Neural Network)
    tf.keras.layers.Flatten(),
    # Stopping OverFitting
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Output Layer
    tf.keras.layers.Dense(3, activation='softmax')
])

# Printed summary of Model (Trainable Parameters)
model.summary()

# Compiled the model
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Ran the model
history = model.fit(train_generator, epochs=15, validation_data = validation_generator, verbose = 1)

# Tested files from the Internet with the machine
path = glob.glob("C:/Users/Kiwi/PycharmProjects/Machine Learning/multiclass/testing/*.png")

for file in path:
    # Loaded the image
    img = tf.keras.utils.load_img(file, target_size=(150, 150))

    # Changed img into 2D array
    x = tf.keras.utils.img_to_array(img)

    #Now because we are working with 3D arrays set the 3rd dimensiion as 0
    x = np.expand_dims(x, axis=0)

    # Stacked the array in a vertical style
    image_tensor = np.vstack([x])

    # Ran the model to predict the type of image
    classes = model.predict(image_tensor, batch_size=10)
    print(image)
    print(classes)

# [1. 0. 0.] shows paper , [0. 1. 0.] shows rock , [0. 0. 1.] shows scissor