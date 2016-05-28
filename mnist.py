from __future__ import print_function

import cPickle
import gzip
import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

# load the mnist data (specify correct path)
import utils

mnist_path = '/Users/benjaminbolte/Documents/datasets/mnist.pkl.gz'
f = gzip.open(mnist_path, 'rb')
(train_X, train_y), (valid_X, valid_y), (test_X, test_y) = cPickle.load(f)
f.close()

n_in, n_out = train_X.shape[1], 10
train_y, valid_y, test_y = [utils.to_ohe(y, 10) for y in [train_y, valid_y, test_y]]

# show some data
# plt.imshow(train_X[train_y==5].reshape(-1, 28, 28).mean(axis=0), interpolation='bicubic'); plt.show()


def get_mlp_model(n_in, n_out, n_layers=2, n_hidden=50):
    assert n_layers >= 2, '`n_layers` should be greater than 1 (otherwise it is just an mlp)'

    # initialize weights
    weights = [utils.get_weights('w_1', n_in, n_hidden)]
    weights += [utils.get_weights('w_%d' % i, n_hidden, n_hidden) for i in range(2, n_layers)]
    weights += [utils.get_weights('w_%d' % n_layers, n_hidden, n_out)]

    # initialize biases
    biases = [utils.get_weights('b_%d' % i, n_hidden) for i in range(1, n_layers)]
    biases += [utils.get_weights('b_%d' % n_layers, n_out)]

    # binarized versions
    deterministic_binary_weights = [utils.binarize(w, mode='deterministic') for w in weights]
    deterministic_binary_biases = [utils.binarize(b, mode='deterministic') for b in biases]
    stochastic_binary_weights = [utils.binarize(w, mode='stochastic') for w in weights]
    stochastic_binary_biases = [utils.binarize(b, mode='stochastic') for b in biases]

    # variables
    lr = T.scalar(name='learning_rate')
    X = T.matrix(name='X', dtype=theano.config.floatX)
    y = T.matrix(name='y', dtype=theano.config.floatX)

    # generate outputs of mlps
    d_outs = [utils.hard_sigmoid(T.dot(X, deterministic_binary_weights[0]) + deterministic_binary_biases[0])]
    for w, b in zip(deterministic_binary_weights[1:], deterministic_binary_biases[1:]):
        d_outs.append(utils.hard_sigmoid(T.dot(d_outs[-1], w) + b))
    s_outs = [utils.hard_sigmoid(T.dot(X, stochastic_binary_weights[0]) + stochastic_binary_biases[0])]
    for w, b in zip(stochastic_binary_weights[1:], stochastic_binary_biases[1:]):
        s_outs.append(utils.hard_sigmoid(T.dot(s_outs[-1], w) + b))

    # cost function (see utils)
    cost = utils.get_cost((s_outs[-1]+1.)/2., (y+1.)/2., mode='mse')

    # get the update functions
    params = weights + biases
    grads = [T.grad(cost, p) for p in stochastic_binary_weights + stochastic_binary_biases]
    updates = [(p, T.clip(p - lr * g, -1, 1)) for p, g in zip(params, grads)]

    # generate training and testing functions
    train_func = theano.function([X, y, lr], [cost], updates=updates)
    test_func = theano.function([X], [d_outs[-1]])
    grads_func = theano.function([X, y], grads)
    int_output_func = theano.function([X], s_outs + d_outs)

    return train_func, test_func, grads_func, weights + biases, int_output_func

print('compiling model...')
train_func, test_func, grads_func, weights, int_output_func = get_mlp_model(n_in, n_out, n_layers=2, n_hidden=50)

print('accuracy before training:', (test_func(test_X)[0].argmax(axis=1)==test_y.argmax(axis=1)).mean())

print('training model...')
nb_epoch = 100
nb_subepoch = 5
batch_size = 32
learning_rate = 0.01
for i in range(nb_epoch):
    for j in range(nb_subepoch):
        train_X, train_y = utils.shuffle_together(train_X, train_y)
        batch = lambda x: utils.split_minibatch(x, batch_size)
        err = [train_func(x, y, learning_rate)[0].mean() for x, y in zip(batch(train_X), batch(train_y))]
        print('epoch:', i+1, 'subepoch:', j+1, 'average loss:', sum(err) / float(len(err)), 'learning rate:', learning_rate, 'gradients:',  '.join([str(np.absolute(g).mean()) for g in grads_func(train_X, train_y)]))

    # print(int_output_func(train_X[:5]))
    print('train accuracy:', (test_func(train_X)[0].argmax(axis=1)==train_y.argmax(axis=1)).mean())
    learning_rate *= 0.99

print('accuracy after training:', (test_func(test_X)[0].argmax(axis=1)==test_y.argmax(axis=1)).mean())
