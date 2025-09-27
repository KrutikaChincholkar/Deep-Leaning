import tensorflow as tf

# Create a constant tensor
s1 = tf.constant(7)

# Print the tensor
print("Tensor:", s1)
#Tensor: tf.Tensor(7, shape=(), dtype=int32)


# Print the number of dimensions
print("Number of dimensions:", s1.ndim)
#Number of dimensions: 0
