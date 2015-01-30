# -*- coding: utf-8 -*-

"""This module provides an implementation of full matrix adagrad."""

from __future__ import division

from base import Minimizer
from mathadapt import sqrt, ones_like, clip, zero_like
from scipy.linalg import pinv as scipy_pinv, polar
import numpy as np
from fjlt.SubsampledRandomizedFourrierTransform import SubsampledRandomizedFourrierTransform
from scipy.linalg import sqrtm

class AdagradFull(Minimizer):
    """Full Matrix AdaGrad optimizer.

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

    state_fields = 'n_iter g_avg Gt eta lamb delta'.split()

    def __init__(self, wrt, fprime, eta, lamb, delta, n_classes=None, args=None):
        """Create a AadaGradFull object.

        Parameters
        ----------

        wrt : array_like
            Array that represents the solution. Will be operated upon in
            place.  ``fprime`` should accept this array as a first argument.

        fprime : callable
            Callable that given a solution vector as first parameter and *args
            and **kwargs drawn from the iterations ``args`` returns a
            search direction, such as a gradient.

        step_rate : scalar or array_like, optional [default: 1]
            Value to multiply steps with before they are applied to the
            parameter vector.


        args : iterable
            Iterator over arguments which ``fprime`` will be called with.
        """
        super(AdagradFull, self).__init__(wrt, args=args)

        self.n_classes = n_classes
            
        self.fprime = fprime
        self.g_avg = zero_like(wrt)
        self.Gt = np.zeros((self.g_avg.shape[0], self.g_avg.shape[0]))  # TODO: make eye
        self.eta = eta
        self.lamb = lamb
        self.delta = delta
        self.I_delta = np.diag(np.ones(self.Gt.shape[0]) * delta)

        self.eye_Gt = np.eye(*self.Gt.shape)

        if self.n_classes is not None:
            self.n_param = wrt.shape[0]
            self.n_features = (self.n_param-self.n_classes)/self.n_classes

    def _iterate(self):
        for args, kwargs in self.args:

            gradient = self.fprime(self.wrt, *args, **kwargs)
            
            self.Gt += np.outer(gradient, gradient)
            St = self._my_sqrtm(self.I_delta + self.Gt)
            Ht_inv = np.linalg.inv(St)
            if self.n_classes is None:
                uppro = np.dot(Ht_inv, gradient)
            else:
                vec = np.dot(Ht_inv, gradient)
                uppro = np.r_[np.array([vec[self.n_features * i:self.n_features * (i + 1)] for i in range(self.n_classes)]).flatten(), vec[self.n_param - self.n_classes:]]
            self.wrt -= self.eta * uppro


            self.n_iter += 1

            yield {
                'n_iter': self.n_iter,
                'gradient': gradient,
                'args': args,
                'kwargs': kwargs,
            }

    
    def _my_sqrtm(self, X):
#         tol = 1e-7
        return np.real(sqrtm(X))  # + self.eye_Gt * tol)

