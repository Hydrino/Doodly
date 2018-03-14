import numpy as np
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPool2D
from keras.models import Sequential
from keras.utils import np_utils
from keras import regularizers

np.random.seed(123)

# -------------------------------- get the data ------------------------------------------

apple_data = np.load("..\\Data\\apple.npy")
clock_data = np.load("..\\Data\\alarm_clock.npy")

print("apple data : {}\nclock data : {}\ntotal : {}".format(apple_data.shape, clock_data.shape,
                                                            apple_data.shape[0] + clock_data.shape[0]))

# ---------------------------pre process the data ----------------------------------

print("Pre processing data...")
# apple : 0 , clock : 1
y_apple = [0] * apple_data.shape[0]
y_clock = [1] * clock_data.shape[0]

X = np.concatenate((apple_data, clock_data))
Y = np.concatenate((y_apple, y_clock))

data = np.column_stack((X, Y))
np.random.shuffle(data)
print("Data : {}".format(data.shape))

X = data[:, 0:784]
Y = data[:, 784]

# putting values as 0 or 1
X[X != 0] = 1

X = X.reshape(X.shape[0], 28, 28, 1)
Y = np_utils.to_categorical(Y, 2)

print("X : {}\nY : {}".format(X.shape, Y.shape))
print("Done!")
print("*" * 70)

# --------------------  creating model ---------------------------------

# hyper parameters

input_shape = (28, 28, 1)
activation = 'relu'
conv2d_1_filters = 15
conv2d_1_kernel_size = (3, 3)
conv2d_1_strides = (1, 1)
reg = 0.03
conv2d_2_filters = 30
conv2d_2_kernel_size = (3, 3)
conv2d_2_strides = (1, 1)
dropout_rate = 0.275
max_pool_size = (2, 2)
dense_1_units = 64
dense_2_units = 10
dense_output_units = 2
final_activation = 'softmax'

print("Creating CNN model....")
model = Sequential()

model.add(Convolution2D(filters=conv2d_1_filters, kernel_size=conv2d_1_kernel_size, strides=conv2d_1_strides,
                        activation=activation, input_shape=input_shape,
                        kernel_regularizer=regularizers.l2(reg),
                        bias_regularizer=regularizers.l2(reg),
                        activity_regularizer=regularizers.l2(reg)))
print(model.output_shape)

model.add(MaxPool2D(max_pool_size))  # default (2,2)
print(model.output_shape)

model.add(Convolution2D(filters=conv2d_2_filters, kernel_size=conv2d_2_kernel_size, strides=conv2d_2_strides,
                        activation=activation,
                        kernel_regularizer=regularizers.l2(reg),
                        bias_regularizer=regularizers.l2(reg),
                        activity_regularizer=regularizers.l2(reg)))
print(model.output_shape)

model.add(MaxPool2D(max_pool_size))
print(model.output_shape)

model.add(Dropout(dropout_rate))

# important to 'flatten' previous output so it can pass through dense layers
model.add(Flatten())
print(model.output_shape)

model.add(Dense(dense_1_units, activation=activation))
model.add(Dropout(dropout_rate))

model.add(Dense(dense_2_units, activation=activation))
model.add(Dropout(dropout_rate))

# have to use 2 states with softmax+ categorical_crossentropy
# else we can use one state(0,1) + sigmoid + binary_crossentropy
model.add(Dense(dense_output_units, activation=final_activation,
                kernel_regularizer=regularizers.l2(reg),
                bias_regularizer=regularizers.l2(reg),
                activity_regularizer=regularizers.l2(reg)))

print(model.output_shape)
print(model.summary())
print("Done!")
print("*" * 70)

# -----------------------time to train ! ----------------------------
print("Training our model.....")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, Y, batch_size=32, epochs=5)
print("Done!")
print("*" * 70)

# -------------------------------- Save the model and weights ---------------------
print("Saving our model...")
model.save("..\\Data\\trained_model.h5")
model.save_weights('..\\Data\\trained_model_weights.h5')
print("Done!")
print("*" * 70)
