import tensorflow as tf

# Create a 2-D tensor (matrix)
m1 = tf.constant([[10, 11], [12, 13]])

# Print the tensor
print("Tensor:\n", m1)
# Output: Tensor: tf.Tensor([[10 11],[12 13]], shape=(2, 2), dtype=int32)

# Print the number of dimensions
print("Number of dimensions:", m1.ndim)
#Output: Number of dimensions: 2
