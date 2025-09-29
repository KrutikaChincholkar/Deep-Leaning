import tensorflow as tf
import datetime

# Initialize a TensorFlow variable
theta = tf.Variable(5.0, dtype=tf.float32)

# Define SGD optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# Set TensorBoard log directory
log_dir = "Logs/sgd/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

# Gradient descent loop
for step in range(20):
    with tf.GradientTape() as tape:
        loss = theta ** 2  # loss function: f(theta) = theta^2

    # Compute gradient
    grad = tape.gradient(loss, [theta])

    # Apply gradient using SGD optimizer
    optimizer.apply_gradients(zip(grad, [theta]))

    # Log values for TensorBoard
    with writer.as_default():
        tf.summary.scalar("Loss", loss, step=step)
        tf.summary.scalar("Theta", theta, step=step)
        tf.summary.scalar("Gradient", grad[0], step=step)

    # Print progress
    print(f"[GD] Step {step}: θ = {theta.numpy():.4f}, Loss = {loss.numpy():.4f}, Gradient = {grad[0].numpy():.4f}")
