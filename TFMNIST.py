import tensorflow as tf
import numpy as np

# Step 1: Load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 2: Preprocess data
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# Convert labels to one-hot
num_classes = 10
y_train_onehot = tf.one_hot(y_train, num_classes)
y_test_onehot = tf.one_hot(y_test, num_classes)

# Step 3: Define model parameters
hidden_units = 128
W1 = tf.Variable(tf.random.normal([784, hidden_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([hidden_units]))
W2 = tf.Variable(tf.random.normal([hidden_units, num_classes], stddev=0.1))
b2 = tf.Variable(tf.zeros([num_classes]))

# Step 4: Define loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Step 5: Define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Step 6: Training loop
batch_size = 128
num_batches = x_train.shape[0] // batch_size

for epoch in range(5):
    epoch_loss = 0
    for i in range(num_batches):
        x_batch = x_train[i*batch_size:(i+1)*batch_size]
        y_batch = y_train_onehot[i*batch_size:(i+1)*batch_size]

        with tf.GradientTape() as tape:
            # Forward pass
            layer1 = tf.nn.relu(tf.matmul(x_batch, W1) + b1)
            logits = tf.matmul(layer1, W2) + b2
            loss = loss_fn(y_batch, logits)

        # Compute gradients
        grads = tape.gradient(loss, [W1, b1, W2, b2])

        # Update parameters
        optimizer.apply_gradients(zip(grads, [W1, b1, W2, b2]))
        epoch_loss += loss.numpy()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss/num_batches:.4f}")

# Step 7: Evaluate on test data
layer1_test = tf.nn.relu(tf.matmul(x_test, W1) + b1)
logits_test = tf.matmul(layer1_test, W2) + b2
preds = tf.argmax(logits_test, axis=1)

accuracy = np.mean(preds.numpy() == y_test)
print(f"Test Accuracy: {accuracy:.4f}")
