import tensorflow as tf
import matplotlib.pyplot as plt

# 🧠 Step 1: Load the MNIST Dataset (handwritten digits 0–9)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ⚙️ Step 2: Normalize pixel values (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 🧩 Step 3: Define the Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),     # Convert 2D (28x28) → 1D (784)
    tf.keras.layers.Dense(64, activation='relu'),      # Hidden layer with 64 neurons
    tf.keras.layers.Dense(10, activation='softmax')    # Output layer (10 classes)
])

# 🧮 Step 4: Compile the Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 🏋️ Step 5: Train the Model
history = model.fit(
    x_train, y_train,
    epochs=4,
    validation_data=(x_test, y_test),
    verbose=1
)

# 📊 Step 6: Plot Accuracy
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ✅ Step 7: Evaluate on Test Data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
