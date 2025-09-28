import tensorflow as tf

# Create a constant tensor
t1 = tf.constant([1, 2, 3, 4])

# Convert it into a TensorFlow variable
v1 = tf.Variable(t1)

# Print details
print("Constant:", t1)
#Output-> Constant: tf.Tensor([1 2 3 4], shape=(4,), dtype=int32)
print("Variable:", v1)
#Output-> Variable: <tf.Variable 'Variable:0' shape=(4,) dtype=int32, numpy=array([1, 2, 3, 4], dtype=int32)>

print("Shape:", v1.shape)
#Output-> Shape: (4,)

print("Dtype:", v1.dtype)
#Output-> Dtype: <dtype: 'int32'>

print("Number of dimensions:", v1.ndim)
#Output-> Number of dimensions: 1
