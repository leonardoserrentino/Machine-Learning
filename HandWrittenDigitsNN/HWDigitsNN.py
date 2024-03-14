import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #input 1x(28*28) pixel dell'immagine
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu)) #Dense significa che sono connessi tutti a tutti quelli precedenti, numero di unita' e funzione di attivazione da decidere opportunamente
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu)) 
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)) #numero di unita' e softmax per il numero di classi in cui puoi predirre il valore

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3) #3 epoche

accuracy, loss = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')


for i in range(1,6):
    img = cv.imread(f'{i}.png')[:,:,0]
    img = np.array([img])
    plt.imshow(img[0])
    plt.show()
