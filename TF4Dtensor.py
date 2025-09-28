import tensorflow as tf

# Generate a 4-dimensional tensor of integers
t4 = tf.constant(
    [
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ],
        [
            [[9, 10], [11, 12]],
            [[13, 14], [15, 14]]
        ]
    ]
)

# Print the tensor
print("Tensor:\n", t4)
#Output-> Tensor: tf.Tensor([[[[ 1  2]   [ 3  4]]  [[ 5  6]   [ 7  8]]] [[[ 9 10]   [11 12]]  [[13 14]   [15 14]]]], shape=(2, 2, 2, 2), dtype=int32)

# Print the number of dimensions
print("Number of dimensions:", t4.ndim)
#Output->Number of dimensions: 4

# Also print shape for clarity
print("Shape:", t4.shape)
#Output->Shape: (2, 2, 2, 2)
