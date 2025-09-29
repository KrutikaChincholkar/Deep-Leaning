import tensorflow as tf

# Create a 2-D TensorFlow variable
t2d = tf.Variable([[12, 23, 36],
                   [34, 25, 18]])

# Print the variable
print("Tensor:\n", t2d)
# Output-> Tensor: <tf.Variable 'Variable:0' shape=(2, 3) dtype=int32, numpy=array([[12, 23, 36], [34, 25, 18]], dtype=int32)>

# Find the index of the maximum value along the last axis (axis=-1 → row-wise)
argmax_values = tf.argmax(t2d, axis=-1)

print("Row-wise argmax indices:", argmax_values)
# Output-> Row-wise argmax indices: tf.Tensor([2 0], shape=(2,), dtype=int64)
