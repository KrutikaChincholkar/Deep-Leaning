import tensorflow as tf
from tensorflow.keras import layers, models

# Assuming X_train, X_test, y_train, y_test are already loaded from mnist.load_data()

# Step 1: Reshape and normalize
X_train = X_train.reshape((60000, 784)).astype("float32") / 255.0
X_test = X_test.reshape((10000, 784)).astype("float32") / 255.0

# Step 2: One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Step 3: Build the model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),  # hidden layer 1
    layers.Dense(64, activation='relu'),                       # hidden layer 2
    layers.Dense(10, activation='softmax')                     # output layer
])

# Step 4: Compile model
model.compile(
    optimizer='adam',                         # adaptive learning optimizer
    loss='categorical_crossentropy',          # for one-hot labels
    metrics=['accuracy']
)

# Step 5: Train model
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,  # 10% data used for validation
    verbose=1
)
