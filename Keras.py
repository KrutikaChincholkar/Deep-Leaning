import tensorflow as tf
from tensorflow import keras

# 1️⃣ Load MNIST dataset (handwritten digits 0–9)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 2️⃣ Normalize the data (convert from [0,255] → [0,1])
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3️⃣ Build the model (Flatten → Dense ReLU → Dense Softmax)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),    # 784 input features
    keras.layers.Dense(128, activation="relu"),   # Hidden layer
    keras.layers.Dense(10, activation="softmax")  # Output layer (10 classes)
])

# 4️⃣ Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # since labels are integers
    metrics=["accuracy"]
)

# 5️⃣ Train the model
model.fit(x_train, y_train, epochs=10)

# 6️⃣ Evaluate on test set
model.evaluate(x_test, y_test)
