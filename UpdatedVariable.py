import tensorflow as tf

# Create variable and constant
vt = tf.Variable([10, 7])
ct = tf.constant([10, 7])

print("Original Variable:", vt.numpy())
#Output-> Original Variable: [10  7]

print("Original Constant:", ct.numpy())
#Output-> Original Constant: [10  7]

# Update the first element of the variable
vt[0].assign(7)
print("\nUpdated Variable:", vt.numpy())
#Output-> Updated Variable: [7 7]

# Trying to update constant will raise an error
# ct[0].assign(7)  # This will throw: 'EagerTensor' object has no attribute 'assign'


Original Variable: [10  7]
Original Constant: [10  7]

Updated Variable: [7 7]
