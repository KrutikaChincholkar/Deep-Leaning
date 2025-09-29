import tensorflow as tf

# Create a 2-D TensorFlow variable
t2d = tf.Variable([[12, 23, 36],
                   [34, 25, 18]])

print("Original Tensor:\n", t2d)
#Output -> Original Tensor: <tf.Variable 'Variable:0' shape=(2, 3) dtype=int32, numpy=array([[12, 23, 36],[34, 25, 18]], dtype=int32)>

# Reshape (just to show effect) – but note: (1, 4) is invalid for a (2,3) tensor
# because total elements must match (2*3 = 6, not 4)
# Correct reshape example:
reshaped = tf.reshape(t2d, (1, 6))
print("\nReshaped Tensor (1,6):\n", reshaped)
#Output -> Reshaped Tensor (1,6):tf.Tensor([[12 23 36 34 25 18]], shape=(1, 6), dtype=int32)
 
# Argmax along axis=1 (row-wise max indices)
argmax_values = tf.argmax(t2d, axis=1)
print("\nRow-wise argmax indices:", argmax_values)
#Output -> Row-wise argmax indices: tf.Tensor([2 0], shape=(2,), dtype=int64)

# Argmax along axis=0 (column-wise max indices)
argmax_cols = tf.argmax(t2d, axis=0)
print("Column-wise argmax indices:", argmax_cols)
#Output -> Column-wise argmax indices: tf.Tensor([1 1 0], shape=(3,), dtype=int64)
