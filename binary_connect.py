''' Implementation of BinaryConnect for learning some random data with binary weights (-1, 1) '''

from __future__ import print_function

import theano.tensor as T
from utils import *

# model
N_IN = 20
N_HIDDEN = 40
N_OUT = 1
LEARNING_RATE = 0.1
N_TEST = 50 # batch size

W1 = get_weights('W1', N_IN, N_HIDDEN)
b1 = get_weights('b1', N_HIDDEN)
W2 = get_weights('W2', N_HIDDEN, N_OUT)
b2 = get_weights('b2', N_OUT)

X = T.matrix(name='X', dtype=theano.config.floatX)
y = T.matrix(name='y', dtype=theano.config.floatX)
lr = T.scalar(name='lr', dtype=theano.config.floatX)

# stochastic and deterministic versions of each variable
W1b = binarize(W1 * (np.random.rand(N_IN, N_HIDDEN) * (N_IN + N_HIDDEN)), mode='stochastic')
W2b = binarize(W2 * (np.random.rand(N_HIDDEN, N_OUT) * (N_HIDDEN + N_OUT)), mode='stochastic')
W1d = binarize(W1, mode='deterministic')
W2d = binarize(W2, mode='deterministic')
b1b = binarize(b1, mode='stochastic')
b2b = binarize(b2, mode='stochastic')
b1d = binarize(b1, mode='deterministic')
b2d = binarize(b2, mode='deterministic')

z1 = hard_sigmoid(T.dot(X, W1b) + b1b)
z2 = hard_sigmoid(T.dot(z1, W2b) + b2b)
z1d = hard_sigmoid(T.dot(X, W1d) + b1d)
z2d = hard_sigmoid(T.dot(z1d, W2d) + b2d)

cost = get_cost(y, z2)
params = [W1, W2, b1, b2]
grads = [T.grad(cost, W1b),
         T.grad(cost, W2b),
         T.grad(cost, b1b),
         T.grad(cost, b2b)]
updates = [(p, T.clip(p - lr * g, -1, 1)) for p, g in zip(params, grads)]

train = theano.function([X, y, lr], [cost], updates=updates)
test = theano.function([X], [z2d])
get_grads = theano.function([X, y], grads)
get_binary_weights = theano.function([], [W1d, W2d, b1d, b2d])
get_intermediate_outputs = theano.function([X], [T.dot(X, W1d), z1d, T.dot(z1d, W2d)])

# generate some random data
X_data = np.asarray(np.random.rand(N_TEST, N_IN) > 0.5, dtype=theano.config.floatX) * 2 - 1
y_data = np.asarray(np.random.rand(N_TEST, N_OUT) > 0.5, dtype=theano.config.floatX) * 2 - 1
# assert N_IN == 2 and N_OUT == 1
# X_data = np.asarray([[1, -1], [-1, 1], [-1, -1], [1, 1]])
# y_data = np.asarray([[1], [1], [-1], [-1]])

# train on the random data
print('accuracy before training: ', (test(X_data)[0] == y_data).sum() / float(y_data.shape[0]))

nb_epoch = 10
epoch_size = 3000
for i in range(nb_epoch):
    err = sum([train(X_data, y_data, LEARNING_RATE)[0].sum() for _ in range(epoch_size)]) / float(epoch_size)
    # for g in get_grads(X_data, y_data):
    #     print(g.sum())
    print('epoch: ', i, ' average loss: ', err, ' learning rate: ', LEARNING_RATE)
    LEARNING_RATE *= 0.95

# print(y_data)
# print(test(X_data)[0] - y_data)
print('accuracy after training: ', (test(X_data)[0] == y_data).sum() / float(y_data.shape[0]))
print(get_binary_weights())
# print(get_intermediate_outputs(X_data))
