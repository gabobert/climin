import itertools

import nose
import numpy as np
import matplotlib.pyplot as plt

from climin import Radagrad, Adagrad, AdagradFull

from losses import Quadratic, LogisticRegression, Rosenbrock, RegularizedLogisticRegression
from common import continuation
from climin import Adadelta
import random

def squared_l2_reg(wrt, lamb):
    return lamb * np.dot(wrt, wrt)

def test_radagrad_lr():
    seed = 125
    random.seed(seed)
    n_samples = 100
    n_dim = 40
    n_classes = 3
    obj_rada = LogisticRegression(n_samples=n_samples, n_inpt=n_dim, n_classes=n_classes, seed=seed)
    print obj_rada.X.shape
    obj_ada = LogisticRegression(n_samples=n_samples, n_inpt=n_dim, n_classes=n_classes, seed=seed)
    obj_dada = LogisticRegression(n_samples=n_samples, n_inpt=n_dim, n_classes=n_classes, seed=seed)
    obj_fada = LogisticRegression(n_samples=n_samples, n_inpt=n_dim, n_classes=n_classes, seed=seed)
    ridx = random.sample(xrange(obj_rada.X.shape[0]), obj_rada.X.shape[0])
#     ridx = xrange(obj_rada.X.shape[0])
#     print ridx
    eta = 0.5
    eta_rada = 0.5
    delta = 0.001
#     k = obj_rada.pars.shape[0]
    k = int(np.sqrt(n_dim) + 15)
    print n_dim, k
    opt_rada = Radagrad(obj_rada.pars, obj_rada.fprime, eta_rada, 0.001, delta, k, n_classes=n_classes, args=itertools.repeat(((obj_rada.X[ridx], obj_rada.Z[ridx]), {})))
    opt_ada = Adadelta(obj_ada.pars, obj_ada.fprime, 0.9, args=itertools.repeat(((obj_ada.X[ridx], obj_ada.Z[ridx]), {})))
    opt_dada = Adagrad(obj_dada.pars, obj_dada.fprime, eta, delta, args=itertools.repeat(((obj_dada.X[ridx], obj_dada.Z[ridx]), {})))
    opt_fada = AdagradFull(obj_fada.pars, obj_fada.fprime, eta, 0.001, delta, n_classes=n_classes, args=itertools.repeat(((obj_rada.X[ridx], obj_rada.Z[ridx]), {})))

    rada_loss, ada_loss, dada_loss, fada_loss = [], [], [], []


#     print opt_ada.wrt
#     print obj_ada.pars
    for i, info in enumerate(opt_ada):
#         print info
        ada_loss += [obj_ada.f(opt_ada.wrt, obj_ada.X, obj_ada.Z)]
        
        if i > n_samples * 6:
            break

#     print opt_dada.wrt
#     print obj_dada.pars
    for i, info in enumerate(opt_dada):
#         print info
        dada_loss += [obj_dada.f(opt_dada.wrt, obj_dada.X, obj_dada.Z)]

        if i > n_samples * 6:
            break


#     print opt_rada.wrt
#     print obj_rada.pars
    for i, info in enumerate(opt_rada):
#         print info['args'][0][i]

        rada_loss += [obj_rada.f(opt_rada.wrt, obj_rada.X, obj_rada.Z)]

        if i > n_samples * 6:
            break

    for i, info in enumerate(opt_fada):
#         print info['args'][0][i]

        fada_loss += [obj_rada.f(opt_fada.wrt, obj_fada.X, obj_fada.Z)]

        if i > n_samples * 6:
            break


    plt.plot(rada_loss, '-r')
    plt.plot(ada_loss, '-b')
    plt.plot(dada_loss, '-g')
    plt.plot(fada_loss, '-k')

    plt.show()

def test_adadelta_continue():
    obj = LogisticRegression(n_inpt=2, n_classes=2)
    args = itertools.repeat(((obj.X, obj.Z), {}))
    opt = Radagrad(obj.pars, obj.fprime, 0.9, args=args)

    continuation(opt)

if __name__ == '__main__':
    test_radagrad_lr()
