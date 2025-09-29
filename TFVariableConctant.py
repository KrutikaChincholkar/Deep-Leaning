import tensorflow as tf

# Create a variable and a constant
vt = tf.Variable([10, 7])
ct = tf.constant([10, 7])

# Print both
print("Variable:", vt)
#Output-> Variable: <tf.Variable 'Variable:0' shape=(2,) dtype=int32, numpy=array([10,  7], dtype=int32)>

print("Constant:", ct)
#Output-> Constant: tf.Tensor([10  7], shape=(2,), dtype=int32)

# Check details
print("\n--- Details ---")
#Output-> --- Details ---

print("Variable - shape:", vt.shape, ", dtype:", vt.dtype, ", ndim:", vt.ndim)
#Output-> Variable - shape: (2,) , dtype: <dtype: 'int32'> , ndim: 1

print("Constant - shape:", ct.shape, ", dtype:", ct.dtype, ", ndim:", ct.ndim)
#Output-> Constant - shape: (2,) , dtype: <dtype: 'int32'> , ndim: 1
