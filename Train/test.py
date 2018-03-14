# here we saved the model as a pb file to access it in Android

from keras.models import load_model
import numpy as np
import cv2

apple = np.load("..\\Data\\apple.npy")
alarm = np.load("..\\Data\\alarm_clock.npy")
model = load_model("..\\Data\\trained_model.h5")

a = apple[93123]
a = a.reshape((28, 28))
a[a != 0] = 1
cv2.imshow('image', a)
cv2.waitKey(0)
cv2.destroyAllWindows()
# a[a != 0] = 1
a = a.reshape(1, 28, 28, 1)

print(model.predict(a))

# print(a)

# test[test != 0] = 1
a = a.reshape((784))
with open("params.txt", "w") as file:
    for i in range(28):
        for j in range(28):
            file.write(np.array_str(a[i * 28 + j]))
        file.write("\n")
