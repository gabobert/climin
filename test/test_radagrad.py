import itertools

import nose
import numpy as np

from climin import Radagrad

from losses import Quadratic, LogisticRegression, Rosenbrock, RegularizedLogisticRegression
from common import continuation

def squared_l2_reg(wrt, lamb):
    return lamb * np.dot(wrt, wrt)

def test_radagrad_lr():
    obj = RegularizedLogisticRegression()
    args = itertools.repeat(((obj.X, obj.Z), {}))
    lamb = 0.00000000001
    eta = 0.2
    delta = 0.0001
    k = obj.pars.shape[0]
    opt = Radagrad(obj.pars, obj.fprime, eta, lamb, delta, k, args=args)

    for i, info in enumerate(opt):
        print obj.f_reg(opt.wrt, obj.X, obj.Z)
        
        if i > 3000:
            break
        
    print obj.score()
#    assert obj.solved(0.15), 'did not find solution'


def test_adadelta_continue():
    obj = LogisticRegression(n_inpt=2, n_classes=2)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Radagrad(obj.pars, obj.fprime, 0.9, args=args)

    continuation(opt)

if __name__ == '__main__':
    test_radagrad_lr()
