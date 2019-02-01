# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# feature scaling is compulsory in deep learning and computer vision

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential  # used to initialize our Neural Network as sequence of layers(other option is as a graph)
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense  # is used to create a fully connected layer in ANN

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))
# no of feature detectors = 32

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(activation='relu', units=128))
classifier.add(Dense(activation='sigmoid', units=1))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

# this prepares the image augmentation
# this creates the training with some modifications to it like shear,zoom, os that the CNN does not overfit
# if not done it will give good accuracy for training set but for test set it will be poor
# create batches, apply random transformations on random selection
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)  # rescaling the pixel values between 0 and 1

# create train set of images,test set
# train set composed of all augmented images extracted from ImageDataGenerator
# here we apply image augmentation on images of our training set, at the same time resizing
# all our images
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

# test set to evaluate model performance
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)

# classifier.fit_generator(training_set,
#                          samples_per_epoch=8000,
#                          nb_epoch=25,
#                          validation_data=test_set,
#                          nb_val_samples=2000)
# this fits the CNN and tests its performance against the test set
