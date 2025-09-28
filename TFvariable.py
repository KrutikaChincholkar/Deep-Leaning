import tensorflow as tf

# Create a TensorFlow variable with string values
t_f = tf.Variable(["a", "b", "c"], dtype=tf.string)

# Print the variable
print("TensorFlow Variable:", t_f)
#Output -> TensorFlow Variable: <tf.Variable 'Variable:0' shape=(3,) dtype=string, numpy=array([b'a', b'b', b'c'], dtype=object)>

# Print details
print("Shape:", t_f.shape)
# Output -> Shape: (3,)

print("Dtype:", t_f.dtype)
# Output -> Dtype: <dtype: 'string'>

print("Number of dimensions:", t_f.ndim)
# Output -> Number of dimensions: 1
