import numpy as np
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPool2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# 1) Initialising and gathering data -----------------------------------------------------------------------------------

# setting constant seed to reproduce results
np.random.seed(123)

# loading the data into numpy arrays
apple_data = np.load("..\\Data\\apple.npy")
clock_data = np.load("..\\Data\\alarm_clock.npy")

# apple_data[apple_data != 0] = 255
# clock_data[clock_data != 0] = 255

# sanity check
# print(apple_data.shape)

# showing an apple
# apple = np.reshape(apple_data[56755],(28,28))
# cv2.imshow('apple',apple)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 2) Assigning labels --------------------------------------------------------------------------------------------------

# useful variables
apple_m, apple_n = apple_data.shape
clock_m, clock_n = clock_data.shape

# apple ->0 ; clock->1

apple_y = [0] * apple_m
clock_y = [1] * clock_m

X = np.concatenate([apple_data, clock_data])
Y = np.concatenate([apple_y, clock_y])

data = np.column_stack((X,Y))
np.random.shuffle(data)
print(data.shape)

# sanity check
# print("{}\n{}".format(X.shape,Y.shape))

# 3) Train-test split --------------------------------------------------------------------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# sanity checks
# print("X_train : {}\nY_train : {}\nX_test : {}\nY_test : {}".format(X_train.shape, Y_train.shape, X_test.shape,
# Y_test.shape))

# print(Y_train[0:10])

# 2) Pre - processing the data -----------------------------------------------------------------------------------------

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

Y_test = np_utils.to_categorical(Y_test, 2)
Y_train = np_utils.to_categorical(Y_train, 2)

# # normalising the data
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

# sanity check
# print(Y_train[3))

# 4) Preparing our model -----------------------------------------------------------------------------------------------

model = Sequential()

# adding a conv layer 32 x (3,3)
model.add(
    Convolution2D(filters=15, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))

# sanity check
print(model.output_shape)

# adding a max pool layer (2,2)
model.add(MaxPool2D(pool_size=(2, 2)))
print(model.output_shape)

# adding a conv layer 10 x (3,3)
model.add(Convolution2D(filters=5, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
print(model.output_shape)
# final max pool layer
model.add(MaxPool2D())
print(model.output_shape)
# adding a dropout layer for regularising
model.add(Dropout(0.1))

# Now connecting this to a normal neural network

# important to 'flatten' previous output so it can pass through dense layers
model.add(Flatten())
print(model.output_shape)
# adding dense (normal neural) layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))
# have to use 2 states with softmax+ categorical_crossentropy
# else we can use one state(0,1) + sigmoid + binary_crossentropy
model.add(Dense(2, activation='softmax'))
print(model.summary())

# 5) Compiling our model -----------------------------------------------------------------------------------------------

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 6) Fitting the model -------------------------------------------------------------------------------------------------
model.fit(X_train, Y_train, batch_size=32, epochs=7, verbose=1)

# 7) Save the model and weights

model.save("..\\Data\\trained_model.h5")
model.save_weights('..\\Data\\trained_model_weights.h5')

# 7) Evaluating the model ----------------------------------------------------------------------------------------------

score = model.evaluate(X_test, Y_test, verbose=0)

print(score)
print(model.metrics_names)
