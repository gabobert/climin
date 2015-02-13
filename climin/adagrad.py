# -*- coding: utf-8 -*-

"""This module provides an implementation of adagrad."""

from __future__ import division

from base import Minimizer
from mathadapt import sqrt, ones_like, clip, zero_like
from scipy.linalg import pinv as scipy_pinv, polar
import numpy as np

class Adagrad(Minimizer):
    """AdaGrad optimizer.

    AdaGrad [duchi, ...]_ is a method that ...

    Let :math:`f'(\\theta_t)` be the derivative of the loss with respect to the
    parameters at time step :math:`t`. In its
    basic form, given a step rate :math:`\\alpha`, a decay term
    :math:`\\gamma` and an offset :math:`\\epsilon` we perform the following
    updates:

    .. math::
       g_t &=& (1 - \\gamma)~f'(\\theta_t)^2 + \\gamma g_{t-1}

    where :math:`g_0 = 0`. Let :math:`s_0 = 0` for updating the parameters:

    .. math::
       \\Delta \\theta_t &=& \\alpha {\sqrt{s_{t-1} + \\epsilon} \over \sqrt{g_t + \\epsilon}}~f'(\\theta_t), \\\\
       \\theta_{t+1} &=& \\theta_t + \\Delta \\theta_t.

    Subsequently we adapt the moving average of the steps:

    .. math::
       s_t &=& (1 - \\gamma)~\\Delta\\theta_t^2 + \\gamma s_{t-1}.

    To extend this with Nesterov's accelerated gradient, we need a momentum
    coefficient :math:`\\beta` and incorporate it by using slightly different
    formulas:

    .. math::
        \\theta_{t + {1 \over 2}} &=& \\theta_t + \\beta \\Delta \\theta_{t-1}, \\\\
       g_t &=& (1 - \\gamma)~f'(\\theta_{t + {1 \over 2}})^2 + \\gamma g_{t-1}, \\\\
       \\Delta \\theta_t &=& \\alpha {\sqrt{s_{t-1} + \\epsilon} \over \sqrt{g_t + \\epsilon}}~f'(\\theta_{t + {1 \over 2}}).

    In its original formulation, the case :math:`\\alpha = 1, \\beta = 0` was
    considered only.

    """

    state_fields = 'n_iter Gt eta delta'.split()

    def __init__(self, wrt, fprime, eta, delta, f=None, args=None):
        """Create a AdaGrad object.

        Parameters
        ----------

        wrt : array_like
            Array that represents the solution. Will be operated upon in
            place.  ``fprime`` should accept this array as a first argument.

        fprime : callable
            Callable that given a solution vector as first parameter and *args
            and **kwargs drawn from the iterations ``args`` returns a
            search direction, such as a gradient.

        args : iterable
            Iterator over arguments which ``fprime`` will be called with.
        """
        super(Adagrad, self).__init__(wrt, args=args)

        self.fprime = fprime
        self.Gt = np.zeros(wrt.shape[0])
        self.eta = eta
        self.delta = delta
        self.f = f


    def _iterate(self):
        for args, kwargs in self.args:

            gradient = self.fprime(self.wrt, *args, **kwargs)
            self.Gt += gradient ** 2

#             print self.wrt
#             print self.Gt
            print self.f(self.wrt, *args, **kwargs)

            self.wrt -= self.eta / np.sqrt(self.Gt + self.delta) * gradient
#             self.wrt -= self.eta * gradient


            self.n_iter += 1

            yield {
                'n_iter': self.n_iter,
                'gradient': gradient,
                'args': args,
                'kwargs': kwargs,
            }
