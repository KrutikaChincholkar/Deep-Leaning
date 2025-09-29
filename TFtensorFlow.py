import tensorflow as tf

# Create a 3-D list (shape = 2 x 2 x 3)
t3d = [
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
]

# Convert it into a TensorFlow tensor
tensor3d = tf.convert_to_tensor(t3d)

# Print results
print("Tensor:\n", tensor3d)
#Output-> Tensor:tf.Tensor([[[ 1  2  3][ 4  5  6]][[ 7  8  9] [10 11 12]]], shape=(2, 2, 3), dtype=int32)

print("Shape:", tensor3d.shape)
#Output->  Shape: (2, 2, 3)

print("Dtype:", tensor3d.dtype)
#Output->  Dtype: <dtype: 'int32'>

print("Number of dimensions:", tensor3d.ndim)
#Output->  Number of dimensions: 3
