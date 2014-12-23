import itertools

import nose
import numpy as np

from climin import Radagrad

from losses import Quadratic, LogisticRegression, Rosenbrock
from common import continuation


def test_radagrad_lr():
    obj = LogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    lamb = 0.00001
    eta = 0.5
    delta = 0.0001
    k = 18
    opt = Radagrad(obj.pars, obj.fprime, eta, lamb, delta, k, args=args)
    for i, info in enumerate(opt):
        print obj.f(opt.wrt, obj.X, obj.Z)
        if i > 3000:
            break
    assert obj.solved(0.15), 'did not find solution'


def test_adadelta_continue():
    obj = LogisticRegression(n_inpt=2, n_classes=2)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Radagrad(obj.pars, obj.fprime, 0.9, args=args)

    continuation(opt)

if __name__ == '__main__':
    test_radagrad_lr()
