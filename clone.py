import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random

lines = []

img_file = '/home/chencan/Project/dataset/CarND-Behavioral-Cloning-P3/img/'
csv_file = '/home/chencan/Project/dataset/CarND-Behavioral-Cloning-P3/driving_log.csv'

with open(csv_file) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []


for i in range(1, len(lines)):
    source_path_center = lines[i][0]
    filename_center = source_path_center.split('/')[-1]
    current_path_center = img_file + filename_center
    image_center = cv2.imread(current_path_center)
    img_center = image_center.copy()
    # print(img.shape)
    cropped_center = img_center[44:150, 0:320]
    # print(cropped.shape)
    resized_center = cv2.resize(cropped_center, (200, 66), interpolation=cv2.INTER_CUBIC)
    rgb_center = cv2.cvtColor(resized_center, cv2.COLOR_BGR2RGB)
    # rgb_center = cv2.cvtColor(rgb_center, cv2.COLOR_RGB2YUV)
    # print(resized.shape)

    source_path_left = lines[i][1]
    filename_left = source_path_left.split('/')[-1]
    current_path_left = img_file + filename_left
    image_left = cv2.imread(current_path_left)
    img_left = image_left.copy()
    cropped_left = img_left[44:150, 0:320]
    resized_left = cv2.resize(cropped_left, (200, 66), interpolation=cv2.INTER_CUBIC)
    rgb_left = cv2.cvtColor(resized_left, cv2.COLOR_BGR2RGB)
    # rgb_left = cv2.cvtColor(rgb_left, cv2.COLOR_RGB2YUV)

    source_path_right = lines[i][2]
    filename_right = source_path_right.split('/')[-1]
    current_path_right = img_file + filename_right
    image_right = cv2.imread(current_path_right)
    img_right = image_right.copy()
    cropped_right = img_right[44:150, 0:320]
    resized_right = cv2.resize(cropped_right, (200, 66), interpolation=cv2.INTER_CUBIC)
    rgb_right = cv2.cvtColor(resized_right, cv2.COLOR_BGR2RGB)
    # rgb_right = cv2.cvtColor(rgb_right, cv2.COLOR_RGB2YUV)

    rgb_flipped = np.copy(np.fliplr(rgb_center))
    rgb_flipped_left = np.copy(np.fliplr(rgb_center))
    rgb_flipped_right = np.copy(np.fliplr(rgb_center))


    correction = 0.06


    measurement_center = float(lines[i][3])
    measurement_left = float(lines[i][3]) + correction
    measurement_right = float(lines[i][3]) - correction
    measurement_flipped = -float(lines[i][3])
    measurement_flipped_left = -measurement_left
    measurement_flipped_right = -measurement_right

    if measurement_center > 0.01 or measurement_center < -0.01:
        images.append(rgb_center)
        images.append(rgb_left)
        images.append(rgb_right)
        images.append(rgb_flipped)
        images.append(rgb_flipped_left)
        images.append(rgb_flipped_right)



        measurements.append(measurement_center)
        measurements.append(measurement_left)
        measurements.append(measurement_right)
        measurements.append(measurement_flipped)
        measurements.append(measurement_flipped_left)
        measurements.append(measurement_flipped_right)


X_train = np.array(images)
y_train = np.array(measurements)

index = [i for i in range(len(X_train))]
random.shuffle(index)

X_train = X_train[index]
y_train = y_train[index]

# print(images[3].shape)
# plt.imshow(images[3])


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Conv2D, normalization, Activation
from keras import regularizers, optimizers

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66, 200, 3)))
# model.add(Cropping2D(cropping=((50, 20), (0, 0))))
# print(model.output_shape)
# model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu", kernel_regularizer=regularizers.l2(0.001),
#                  activity_regularizer=regularizers.l1(0.001)))
# print(model.output_shape)
# model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu", kernel_regularizer=regularizers.l2(0.001),
#                  activity_regularizer=regularizers.l1(0.001)))
# print(model.output_shape)
# model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu", kernel_regularizer=regularizers.l2(0.001),
#                  activity_regularizer=regularizers.l1(0.001)))
# print(model.output_shape)
# model.add(Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001), \
#                  activity_regularizer=regularizers.l1(0.001)))
# print(model.output_shape)
# model.add(Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.001), \
#                  activity_regularizer=regularizers.l1(0.001)))
# print(model.output_shape)
# model.add(Flatten())


model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(Flatten())
model.add(Dropout(0.4))
model.add(normalization.BatchNormalization(gamma_initializer="one", epsilon=1e-06, beta_initializer="zero", weights=None, momentum=0.9, axis=-1))
# model.add(Activation('relu'))

model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.4))
model.add(normalization.BatchNormalization(gamma_initializer="one", epsilon=1e-06, beta_initializer="zero", weights=None, momentum=0.9, axis=-1))
# model.add(Activation('relu'))

model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.5))
model.add(normalization.BatchNormalization(gamma_initializer="one", epsilon=1e-06, beta_initializer="zero", weights=None, momentum=0.9, axis=-1))
# model.add(Activation('relu'))

model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.5))
model.add(normalization.BatchNormalization(gamma_initializer="one", epsilon=1e-06, beta_initializer="zero", weights=None, momentum=0.9, axis=-1))
# model.add(Activation('relu'))
model.add(Dense(1))

# adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='mse', optimizer='adam')

history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=15, verbose=1, batch_size=128)

model.save('./model/model.h5')


plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()