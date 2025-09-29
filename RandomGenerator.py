import tensorflow as tf

# Using seed = 1
gen1 = tf.random.Generator.from_seed(1)
print("Seed 1:", gen1.normal(shape=[3]))
#Output-> Seed 1: tf.Tensor([-0.20470765  0.4789431  -0.5194382 ], shape=(3,), dtype=float32)

# Using seed = 2
gen2 = tf.random.Generator.from_seed(2)
print("Seed 2:", gen2.normal(shape=[3]))
#Output-> Seed 2: tf.Tensor([ 0.16951734 -0.16114324 -0.28298414], shape=(3,), dtype=float32)

# Using seed = 3
gen3 = tf.random.Generator.from_seed(3)
print("Seed 3:", gen3.normal(shape=[3]))
#Output-> Seed 3: tf.Tensor([ 1.2087164 -0.894989   0.5884349], shape=(3,), dtype=float32)
