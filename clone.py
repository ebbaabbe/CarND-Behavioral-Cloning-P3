import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Read in the data, starting with the CSV that provides image paths and steering angles
lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path_center = line[0] # Image paths for all three cameras
    source_path_left = line[1]
    source_path_right = line[2]

    center_path = 'data/' + source_path_center # Updated pathnames
    left_path = 'data/' + source_path_left
    right_path = 'data/' + source_path_right

    center_image = cv2.imread(center_path) # Read in images
    left_image = cv2.imread(left_path)
    right_image = cv2.imread(right_path)

    if center_image is not None: # Create dataset using images/steering angles from each camera
        images.append(center_image)
        measurements.append(float(line[3]) * 1.05)
    if left_image is not None:
        images.append(left_image)
        measurements.append(float(line[3]) * 0.2)
    if right_image is not None:
        images.append(right_image)
        measurements.append(float(line[3]) * -0.2)

# Training set
X_train = np.array(images)
y_train = np.array(measurements)

# Keras model based on Nvidia modes https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model = Sequential()

# pre process as described in lecture
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) # Normalize input data
model.add(Cropping2D(cropping=((70,20),(0,0)))) # Crop 70 pixels off the top and 20 pixels off the bottom of images

model.add(Dense(10))
model.add(Dense(50))
model.add(Dense(100))
model.add(Dense(1164))

#model.add(Convolution2D(6,5,5, border_mode='valid',activation='relu'))
#model.add(MaxPooling2D(pool_size=(3,3), border_mode='valid'))
#model.add(Convolution2D(15,5,5, border_mode='valid',activation='relu'))
#model.add(MaxPooling2D(pool_size=(3,3), border_mode='valid'))
#model.add(Dense(100))
#model.add(Dropout(0.65))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam') # Adam optimizer and mean absolute error for loss
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)

model.save('model.h5')
