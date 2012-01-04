# -*- coding: utf-8 -*-

import itertools

import scipy
import scipy.linalg
import scipy.optimize

from base import Minimizer
from linesearch import WolfeLineSearch


# Things left to do:
#
# - update initial diagonal inverse hessian as in minfunc
# - figure out what the corrections are
# - damped update


class Lbfgs(Minimizer):

    def __init__(self, wrt, f, fprime, initial_hessian_diag=1,
                 n_factors=10, line_search=None,
                 args=None, stop=1, verbose=False):
        super(Lbfgs, self).__init__(wrt, args=args, stop=stop, verbose=verbose)

        self.f = f
        self.fprime = fprime
        self.initial_hessian_diag = initial_hessian_diag
        self.n_factors = n_factors
        if line_search is not None:
            self.line_search = line_search
        else:
            self.line_search = WolfeLineSearch(
                wrt, self.f_with_x, self.fprime_with_x)

    def f_with_x(self, x, *args, **kwargs):
        old = self.wrt.copy()
        self.wrt[:] = x
        res = self.f(*args, **kwargs)
        self.wrt[:] = old
        return res

    def fprime_with_x(self, x, *args, **kwargs):
        old = self.wrt.copy()
        self.wrt[:] = x
        res = self.fprime(*args, **kwargs)
        self.wrt[:] = old
        return res

    def inv_hessian_dot_gradient(self, grad_diffs, steps, grad, hessian_diag, 
                                 idxs):
        grad = grad.copy()  # We will change this.
        n_current_factors = len(idxs)

        # TODO: find a good name for this variable.
        rho = scipy.empty(n_current_factors)

        # TODO: vectorize this function
        for i in idxs:
            rho[i] = 1 / scipy.inner(grad_diffs[i], steps[i])

        # TODO: find a good name for this variable as well.
        alpha = scipy.empty(n_current_factors)

        for i in idxs[::-1]:
            alpha[i] = rho[i] * scipy.inner(steps[i], grad)
            grad -= alpha[i] * grad_diffs[i]
        z = hessian_diag * grad

        # TODO: find a good name for this variable (surprise!)
        beta = scipy.empty(n_current_factors)

        for i in idxs:
            beta[i] = rho[i] * scipy.inner(grad_diffs[i], z)
            z += steps[i] * (alpha[i] - beta[i])

        return z

    def __iter__(self):
        args, kwargs = self.args.next()
        grad = self.fprime(*args, **kwargs)
        grad_m1 = scipy.zeros(grad.shape)
        factor_shape = self.n_factors, self.wrt.shape[0]
        grad_diffs = scipy.zeros(factor_shape)
        steps = scipy.zeros(factor_shape)
        hessian_diag = self.initial_hessian_diag

        # We need to keep track in which order the different statistics
        # from different runs are saved. 
        #
        # Why?
        #
        # Each iteration, we save statistics such as the difference between
        # gradients and the actual steps taken. This are then later combined
        # into an approximation of the Hessian. We call them factors. Since we
        # don't want to create a new matrix of factors each iteration, we
        # instead keep track externally, which row of the matrix corresponds
        # to which iteration. `idxs` now is a list which maps its i'th element
        # to the corresponding index for the array. Thus, idx[i] contains the
        # rowindex of the for the (n_factors - i)'th iteration prior to the
        # current one.
        idxs = []

        for i, (next_args, next_kwargs) in enumerate(self.args):
            if i > 0 and i % self.stop == 0:
                loss = self.f(*args, **kwargs)
                yield dict(loss=loss)

            # If the gradient is exactly zero, we stop. Otherwise, the
            # updates will lead to NaN errors because the direction will
            # be zero.
            if (grad == 0.0).all():
                break

            if i == 0:
                direction = -grad
            else:
                sTgd = scipy.inner(step, grad_diff)
                if sTgd > 1E-10:
                    # Determine index for the current update. 
                    if not idxs:
                        # First iteration.
                        this_idx = 0
                    elif len(idxs) < self.n_factors:
                        # We are not "full" yet. Thus, append the next idxs.
                        this_idx = idxs[-1] + 1
                    else:
                        # we are full and discard the first index.
                        this_idx = idxs.pop(0)

                    idxs.append(this_idx)
                    grad_diffs[this_idx] = grad_diff
                    steps[this_idx] = step
                    hessian_diag = sTgd / scipy.inner(grad_diff, grad_diff)
                else:
                    print 'skipping update,', sTgd
                direction = self.inv_hessian_dot_gradient(
                    grad_diffs, steps, -grad, hessian_diag, idxs)

            steplength = self.line_search.search(direction, args, kwargs)
            step = steplength * direction
            self.wrt += step

            # Prepare everything for the next loop.
            args, kwargs = next_args, next_kwargs
            grad_m1[:], grad[:] = grad, self.line_search.grad

            grad_diff = grad - grad_m1