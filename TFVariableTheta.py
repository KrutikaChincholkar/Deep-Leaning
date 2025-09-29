import tensorflow as tf
import datetime

# Initialize a TensorFlow variable theta
theta = tf.Variable(0.5, dtype=tf.float32)

# Learning rate
lr = 0.1

# Set directory path for TensorBoard logs
log_dir = "logs/gradient_descent/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create a TensorFlow summary writer
writer = tf.summary.create_file_writer(log_dir)

# Perform gradient descent for 20 steps
for step in range(20):
    with tf.GradientTape() as tape:
        loss = theta ** 2  # Define the loss function (f(theta) = theta^2)

    # Compute gradient of loss with respect to theta
    grad = tape.gradient(loss, [theta])[0]

    # Gradient descent update
    theta.assign(theta - lr * grad)

    # Log values to TensorBoard
    with writer.as_default():
        tf.summary.scalar("Loss", loss, step=step)
        tf.summary.scalar("Theta", theta, step=step)
        tf.summary.scalar("Gradient", grad, step=step)

    # Print progress
    print(f"[GD] Step {step}: θ = {theta.numpy():.4f}, Loss = {loss.numpy():.4f}, Gradient = {grad.numpy():.4f}")
