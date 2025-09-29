import tensorflow as tf

# Create a random generator with a fixed seed
r1 = tf.random.Generator.from_seed(1)  # seed ensures reproducibility

# Generate random numbers from a normal distribution
r1 = r1.normal(shape=(3, 2))

print("Random Tensor:\n", r1)
#Output-> Random Tensor: tf.Tensor([[-0.20470765  0.4789431 ][-0.5194382  -0.5557303 ][ 1.9657806   1.3934051 ]], shape=(3, 2), dtype=float32)

print("Shape:", r1.shape)
#Output-> Shape: (3, 2)

print("Dtype:", r1.dtype)
#Output-> Dtype: <dtype: 'float32'>
