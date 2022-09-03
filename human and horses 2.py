from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
#from keras import preprocessing
#from keras.preprocessing import image
import glob

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epochs, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') > 0.995:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

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
                                                    target_size=(300, 300),
                                                    class_mode='binary')

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3),
                                                           activation='relu',
                                                           input_shape=(300, 300, 3)),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(32, (3,3),
                                                           activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3),
                                                           activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3),
                                                           activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(64, (3,3),
                                                           activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512,
                                                          activation='relu'),
                                    tf.keras.layers.Dense(1,
                                                          activation='sigmoid')
                                    ])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=15,
                    callbacks=[callbacks])

path = glob.glob("C:/Users/Kiwi/PycharmProjects/Machine Learning/horse-or-human/test/*.jpg")

for file in path:
    img = tf.keras.utils.load_img(file, target_size=(300, 300))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    if classes[0] > 0.5:
        print(" img is a human")
    else:
        print("img is a horse")
