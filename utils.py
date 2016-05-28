import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

np.random.seed(42)
rng = RandomStreams(42)

costs = {
    'mse': lambda output, target: 0.5 * ((output - target) ** 2).mean(),
    'hinge': lambda output, target, hinge_margin: T.maximum(0., hinge_margin - output * target).mean(),
    'squhinge': lambda output, target, hinge_margin: T.sqr(T.maximum(0., hinge_margin - output * target)).mean(),
    'crossent': lambda output, target: T.nnet.categorical_crossentropy(T.clip(T.nnet.softmax(output), 1e-6, 1.0 - 1e-6), target).mean(),
}


def get_cost(output, target, mode='mse', **kwargs):
    assert mode in costs, '`mode` {} not available. supported cost functions are {}'.format(mode, costs.keys())
    return costs[mode](output, target, *kwargs.values())


def get_weights(name, *shape):
    return theano.shared(np.random.randn(*shape), strict=False, name=name)


def binarize(W, mode='stochastic'):
    assert mode in ['deterministic', 'stochastic'], '`mode` must be either "deterministic" or "stochastic"'
    H = T.sqrt(1.5/T.sum(W.shape))
    Wb = (hard_sigmoid(W/H)+1)/2
    if mode == 'deterministic':
        Wb = T.round(Wb)
    else:
        Wb = T.cast(rng.binomial(n=1, p=Wb, size=T.shape(W)), theano.config.floatX)
    return T.cast(T.switch(Wb, H, -H), theano.config.floatX)


def hard_sigmoid(x):
    return T.clip(x, -1, 1)


def to_ohe(x, n):
    return np.eye(n)[x] * 2 - 1


def split_minibatch(x, minibatch_size):
    for i in range(x.shape[0] / minibatch_size):
        yield x[i*minibatch_size:(i+1)*minibatch_size]
