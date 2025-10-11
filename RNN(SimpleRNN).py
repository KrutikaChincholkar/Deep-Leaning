# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# Generate Sine Wave Data
x = np.linspace(0, 100, 1000)
y = np.sin(x)

# Prepare the Dataset
sequence_length = 20  # use past 20 values to predict next one
X, Y = [], []

# Create Input-Output pairs
for i in range(len(y) - sequence_length):
    X.append(y[i:i + sequence_length])
    Y.append(y[i + sequence_length])

# Convert to NumPy Arrays
X = np.array(X)
Y = np.array(Y)

# Reshape for RNN Input (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build the RNN Model
model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(sequence_length, 1)),
    Dense(1)
])

# Compile and Train the Model
model.compile(optimizer='adam', loss='mse')
history = model.fit(X, Y, epochs=20, batch_size=16, verbose=1)

# Predict
Y_pred = model.predict(X)

# Plot Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(y, label='Actual', color='blue')
plt.plot(np.arange(sequence_length, len(y)), Y_pred, label='Predicted', color='orange')
plt.legend()
plt.title("RNN - Sine Wave Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Sine Value")
plt.show()
