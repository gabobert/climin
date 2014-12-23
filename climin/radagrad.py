# -*- coding: utf-8 -*-

"""This module provides an implementation of radagrad."""

from __future__ import division

from base import Minimizer
from mathadapt import sqrt, ones_like, clip, zero_like
from scipy.linalg import pinv as scipy_pinv, polar
import numpy as np
from fjlt.SubsampledRandomizedFourrierTransform import SubsampledRandomizedFourrierTransform

class Radagrad(Minimizer):
    """RadaGrad optimizer.

    RadaGrad [krummenacher_mcwilliams2014]_ is a method that ...

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

    .. [krummenacher_mcwilliams2014]  Krummenacher, G. and McWilliams, B.
       "RadaGrad: Random Projections for Adaptive Stochastic Optimization","
       preprint http://people.inf.ethz.ch/kgabriel/publications/krummenacher_mcwilliams14radagrad.pdf (2014).
    """

    state_fields = 'n_iter g_avg P_g_avg Gt eta lamb delta'.split()

    def __init__(self, wrt, fprime, eta, lamb, delta, k, args=None):
        """Create a RadaGrad object.

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
        super(Radagrad, self).__init__(wrt, args=args)

        self.fprime = fprime
        self.g_avg = zero_like(wrt)
        self.P_g_avg = np.zeros(k)
        self.Gt = np.zeros((k, k))
        self.eta = eta
        self.lamb = lamb
        self.k = k
        self.delta = delta
        self.I_delta = np.diag(np.ones(self.Gt.shape[0]) * delta)
        self.P_Lamb_inv_P = np.diag(np.ones(k) * wrt.shape[0] / (k * 2 * lamb))
        self.lamb_inv = 1 / (2 * lamb)

        self.srft = SubsampledRandomizedFourrierTransform(k)
        self.srft.fit(wrt)


    def _iterate(self):
        for args, kwargs in self.args:

            t = self.n_iter + 1
            gradient = self.fprime(self.wrt, *args, **kwargs)

            self.g_avg = ((t - 1) / t) * self.g_avg + (1 / t) * gradient
            P_gt = self.srft.transform_1d(gradient)
#             P_gt = gradient
            self.P_g_avg = ((t - 1) / t) * self.P_g_avg + (1 / t) * P_gt
            self.Gt += np.outer(P_gt, P_gt)
            St = self._my_sqrtm(self.Gt)
            Ht_inv = np.linalg.inv(self.I_delta + St)
            Ht_reg_inv = scipy_pinv(Ht_inv + 1 / (t * self.eta) * self.P_Lamb_inv_P)

            uppro = self.srft.inverse_transform_1d(np.dot(Ht_reg_inv, self.P_g_avg))
#             uppro = np.dot(Ht_reg_inv, self.P_g_avg)
            self.wrt = -self.lamb_inv * (self.g_avg - 1 / (t * self.eta) * self.lamb_inv * uppro)


            self.n_iter += 1

            yield {
                'n_iter': self.n_iter,
                'gradient': gradient,
                'args': args,
                'kwargs': kwargs,
            }

    def _my_sqrtm(self, X):
        tol = 1e-7

        L = np.linalg.cholesky(X + np.diag(np.ones(X.shape[0]) * tol))
        U, P = polar(L, side="right")
        return P
