''' Implementation of BinaryConnect for learning some random data with binary weights (-1, 1) '''

from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

np.random.seed(42)
rng = RandomStreams(42)


def hard_sigmoid(x):
    # return T.nnet.sigmoid(x)
    # return T.clip((x + 1.) / 2., 0, 1)
    return T.clip(x, -1, 1)


def glorot(n_in, n_hidden):
    return np.sqrt(1.5 / (n_in + n_hidden))


def binarize(W, deterministic=False, stochastic=False):
    assert deterministic or stochastic
    if deterministic:
        return T.round(W) * 2 - 1
    else:
        Wb = T.cast(rng.binomial(n=1, p=(hard_sigmoid(W) + 1) / 2, size=T.shape(W)), theano.config.floatX)
        return T.cast(T.switch(Wb, 1, -1), theano.config.floatX)


def get_weights(*shape):
    return theano.shared(np.random.randn(*shape) * np.sqrt(1.5 // sum(shape)), strict=False)


def get_cost(output, target):
    return 0.5 * ((output - target) ** 2).sum() # MSE


# model
N_IN = 2
N_HIDDEN = 20
N_OUT = 1
LEARNING_RATE = 0.01
# N_TEST = 5

W1 = get_weights(N_IN, N_HIDDEN)
b1 = get_weights(N_HIDDEN)
W2 = get_weights(N_HIDDEN, N_OUT)
b2 = get_weights(N_OUT)

X = T.matrix(name='X', dtype=theano.config.floatX)
y = T.matrix(name='y', dtype=theano.config.floatX)
lr = T.scalar(name='lr', dtype=theano.config.floatX)

W1b = binarize(W1, stochastic=True)
W2b = binarize(W2, stochastic=True)
z1 = hard_sigmoid(T.dot(X, W1b) + b1)
z2 = hard_sigmoid(T.dot(z1, W2b) + b2)

cost = get_cost(y, z2)
updates = [(W1, W1 - lr * T.grad(cost, W1b)),
           (W2, W2 - lr * T.grad(cost, W2b)),
           (b1, b1 - lr * T.grad(cost, b1)),
           (b2, b2 - lr * T.grad(cost, b2))]
grads = [x[1] for x in updates]

train = theano.function([X, y, lr], [cost], updates=updates)
test = theano.function([X], [z2])
# get_grads = theano.function([X, y, lr], grads)
get_binary_weights = theano.function([], [binarize(W1, deterministic=True), binarize(W2, deterministic=True)])

# generate some random data
# X_data = np.asarray(np.random.rand(N_TEST, N_IN) > 0.5, dtype=theano.config.floatX)
# y_data = np.asarray(np.random.rand(N_TEST, N_OUT) > 0.5, dtype=theano.config.floatX)
X_data = np.asarray([[1, -1], [-1, 1], [-1, -1], [1, 1]])
y_data = np.asarray([[1], [1], [-1], [-1]])

# train on the random data
for i in range(100000):
    err = train(X_data, y_data, LEARNING_RATE)[0]
    if i % 1000 == 0:
        # for g in get_grads(X_data, y_data, LEARNING_RATE):
        #     print(g.sum())
        print(err.sum(), LEARNING_RATE)
        LEARNING_RATE *= 0.99

print(test(X_data)[0])
print(y_data)
# print(get_binary_weights())
