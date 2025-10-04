import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# Step 1 : Generate sample data
X = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
Y = np.array([1.5,3.1,4.5,6.2,7.9,9.0,11.2,12.8,14.5,16.1], dtype=float)

# Step 2 : Build Model
model = keras.Sequential([
    layers.Dense(1, input_shape=[1])  # one input, one output (linear regression)
])

"""
# Optional: Add a hidden layer for non-linear regression
model = keras.Sequential([
    layers.Dense(8, activation="relu", input_shape=[1]),  # hidden layer
    layers.Dense(1)  # output layer
])
"""

# Step 3 : Compile Model
model.compile(optimizer="adam", loss="mean_squared_error")

# Step 4 : Train Model
history = model.fit(X, Y, epochs=800, verbose=0)

# Step 5 : Test Prediction
prediction = model.predict(np.array([12.0]).reshape(-1,1))
print("Predicted value for input 12.0:", prediction[0][0])


plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()
