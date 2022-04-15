import theano
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

a = theano.tensor.dscalar()
b = theano.tensor.dscalar()
s = a + b
funct = theano.function([a, b], s)
g = funct(1, 3)
print(g)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add = tf.add(a, b)
sess = tf.Session()
binding = {a: 1.5, b: 2.5}
c = sess.run(add, feed_dict=binding)
print(c)
