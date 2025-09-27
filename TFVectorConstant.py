import tensorflow as tf

# Create a vector constant (1-D tensor)
v1 = tf.constant([10, 11])

# Print the tensor
print("Tensor:", v1)
# Output-> Tensor: tf.Tensor([10 11], shape=(2,), dtype=int32)


# Print the number of dimensions
print("Number of dimensions:", v1.ndim)
#Output -> Number of dimensions: 1
