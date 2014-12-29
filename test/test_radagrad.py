import itertools

import nose
import numpy as np
import matplotlib.pyplot as plt

from climin import Radagrad

from losses import Quadratic, LogisticRegression, Rosenbrock, RegularizedLogisticRegression
from common import continuation
from climin import Adadelta

def squared_l2_reg(wrt, lamb):
    return lamb * np.dot(wrt, wrt)

def test_radagrad_lr():
    lamb = 0.01
    obj = RegularizedLogisticRegression(n_samples=500, lamb=lamb)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    eta = 0.5
    delta = 0.01
    k = obj.pars.shape[0]
    opt_rada = Radagrad(obj.pars, obj.fprime, eta, lamb, delta, k, args=args)
    opt_ada = Adadelta(obj.pars, obj.fprime_reg, 0.9, args=args)

    rada_loss, ada_loss = [], []

    for i, info in enumerate(opt_rada):
        rada_loss += [obj.f_reg(opt_rada.wrt, obj.X, obj.Z)]
        
        if i > 500:
            break

    for i, info in enumerate(opt_ada):
        ada_loss += [obj.f_reg(opt_ada.wrt, obj.X, obj.Z)]
        
        if i > 500:
            break


    plt.plot(rada_loss, '-r')
    plt.plot(ada_loss, '-b')

    plt.show()

def test_adadelta_continue():
    obj = LogisticRegression(n_inpt=2, n_classes=2)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Radagrad(obj.pars, obj.fprime, 0.9, args=args)

    continuation(opt)

if __name__ == '__main__':
    test_radagrad_lr()
