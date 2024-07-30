#import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import tensorflow as tf
from tensorflow.python.framework import ops
#from tensorflow.examples.tutorials.mnist import input_data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
from PIL import Image

# Import Fashion MNIST
fashion_mnist = tf.keras.datasets.mnist
#fashion_mnist = input_data.read_data_sets('input/data',one_hot=True)
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser','Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu' ))
model.add(tf.keras.layers.Dense(10, activation='softmax' ))
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
# test with 10,000 images
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('10,000 image Test accuracy:', test_acc)