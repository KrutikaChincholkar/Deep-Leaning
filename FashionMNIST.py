# Step 0 : Install & Import
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime

# Step 1 : Load and Prepare Data
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize images to [0,1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Class labels
class_names = [
    "T-shirt/top","Trouser","Pullover","Dress","Coat",
    "Sandal","Shirt","Sneaker","Bag","Ankle boot"
]

# Step 2 : Define the Network
model = keras.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation="relu"),
    layers.Dense(10)
])

# Step 3 : Compile the Network
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Step 4 : Setup TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Step 5 : Train the model
model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback]
)

# Step 6 : Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

# Step 7 : Make predictions
probability_model = keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test[:5])
print("Predicted classes:", tf.argmax(predictions, axis=1).numpy())
print("True labels:", y_test[:5])
