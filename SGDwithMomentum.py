import tensorflow as tf
import datetime

# Variable initialization
theta = tf.Variable(5.0)

# SGD with Momentum
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.7)

# TensorBoard log directory
log_dir = "logs/sgd/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

# Training loop
for step in range(20):
    with tf.GradientTape() as tape:
        loss = theta ** 2  # Loss function: f(theta) = θ²
    
    # Compute Gradient
    grad = tape.gradient(loss, [theta])
    
    # Apply gradients using optimizer with momentum
    optimizer.apply_gradients(zip(grad, [theta]))
    
    # Log to TensorBoard
    with writer.as_default():
        tf.summary.scalar("Loss", loss, step=step)
        tf.summary.scalar("Theta", theta, step=step)
        tf.summary.scalar("Gradient", grad[0], step=step)
    
    # Console output
    print(f"[SGD + Momentum] Step {step}: θ = {theta.numpy():.4f}, Loss = {loss.numpy():.4f}, Grad = {grad[0].numpy():.4f}")
