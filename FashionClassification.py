# import libraries TensorFlow 
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import datetime 

print(tf.__version__)

#load fashion mnist dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['Kaos/atasan', 'Celana panjang', 'Sweater', 'Gaun', 'Mantel', 
               'Sandal', 'Kemeja', 'Sepatu kets', 'Tas', 'Sepatu bot']

#set pixel value of the training data
#Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
#To do so, divide the values by 255. 
# It's important that the training set and the testing set be preprocessed in the same way:
train_images = train_images / 255.0
test_images = test_images / 255.0

#verify first 25 image
""" plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show() """

#building neural network
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#compile model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])
#tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
print (model.get_weights())
#training model
model.fit(train_images, train_labels, epochs=10, callbacks=[tensorboard_callback])
#evaluate accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#prediction
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
     
predictions = probability_model.predict(test_images)
predictions[0]
#highest confidence value
np.argmax(predictions[0])
test_labels[0]

#Define functions to graph the full set of 10 class predictions.
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                    100*np.max(predictions_array),
                                    class_names[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def ext_img(img):
    testImg = Image.open(img)
    testImg.load()
    data = np.asarray( testImg, dtype="float" )
    #data = tf.image.rgb_to_grayscale(data)
    data = 255 - data
    data = data / 255.0
    #data = tf.transpose(data, perm=[2,0,1])
    image = (np.expand_dims(data,0))
    print(image.shape)
    predictions_single = probability_model.predict(image)
    print ("Prediction Output")
    print(predictions_single)
    print()
    NumberElement = predictions_single.argmax()
    Element = np.amax(predictions_single)
    print ("Network menyimpulkan bahwa file '" + img + "' adalah "+class_names[NumberElement])
    print ("Dengan konfiden level " + str(int(Element*100)) + "%")
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

#verify prediction
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red. 
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

#using train model from test dataset
img = test_images[15]
#add to tf.keras collection
img = (np.expand_dims(img,0))
print(img.shape)
#prediction
predictions_single = probability_model.predict(img)
print("Prediction output")
print(predictions_single)
NumberElement = predictions_single.argmax()
Element = np.amax(predictions_single)
print ("Network menyimpulkan bahwa gambar No. '15' adalah " + class_names[NumberElement])
print ("Dengan konfiden level " + str(int(Element*100)) + "%")
img1 = np.array(img, dtype='float')
pixels = img1.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
#predict external image file, place the file in the same directory with grayscale - 28x28 pixel condition
#use ext_img() function like below & change the parameter value with your file
ext_img("pic2.jpg")

     


     

