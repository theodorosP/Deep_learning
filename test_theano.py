import theano

a = theano.tensor.dscalar()
b = theano.tensor.dscalar()

s = a + b
f = theano.function([a, b], s)
print(f(1,4))
