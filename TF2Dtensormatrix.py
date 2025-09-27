import tensorflow as tf

# Create a 2-D tensor (matrix with floats)
m2 = tf.constant([[10., 11.],
                  [12., 13.],
                  [8.,  9.]])

# Print the tensor
print("Tensor:\n", m2)
#Output: Tensor: tf.Tensor([[10. 11.][12. 13.][ 8.  9.]], shape=(3, 2), dtype=float32)


# Print the number of dimensions
print("Number of dimensions:", m2.ndim)
#Output: Number of dimensions: 2
