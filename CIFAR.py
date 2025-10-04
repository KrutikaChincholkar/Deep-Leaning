import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Step 2: Define class names
class_names = ['airplane','car','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# Step 3: Display 25 random sample images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.show()
