'''
Created on Oct 4, 2012

@author: Georgiana Dinu, Pham The Nghia, Antonin Delpeuch
'''

import numpy as np
import logging
import scipy.linalg as splinalg
from sparsesvd import sparsesvd
from warnings import warn
from time import time
import sys
from math import sqrt, log10
from numpy.linalg import LinAlgError
from composes.matrix.matrix import Matrix
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.utils.matrix_utils import assert_same_shape
from composes.utils.matrix_utils import padd_matrix
import composes.utils.log_utils as log

logger = logging.getLogger(__name__)

class Linalg(object):
    """
    Contains a set of linear algebra utilities defined to work both with sparse and
    with dense matrices as an input (i.e. with objects of type SparseMatrix/DenseMatrix).

    Implements:
        svd,
        nmf (LIN algorithm, add citation here!),
        pinv,
        ordinary least squares regression,
        ridge regression
    """

    _NMF_ALPHA = 1.0
    _NMF_BETA = 0.1
    _NMF_MAX_ITER = 20
    _NMF_MAX_ITER_SUBPROB = 15
    _NMF_MIN_TOL = 0.001
    _NMF_TOL = _NMF_MIN_TOL
    _NMF_TOL_DECREASE_FACTOR = 0.5
    _NMF_TIME_LIMIT = 7000

    _SVD_TOL = 1e-12

    @staticmethod
    def svd(matrix_, reduced_dimension):
        """
        Performs SVD decomposition.

        If the rank is smaller than the requested reduced dimension,
        reduction to rank is performed. Dense SVD uses Linalg._SVD_TOL to decide
        the rank of the matrix.


        Args:
           matrix_: input of type Matrix
           reduced_dimension: int, the desired reduced dimension

        Returns:
            U,S,V of the decomposition X = USV^T. U, V: Matrix type,
            S: ndarray of singular values.

        """
        log.print_info(logger, 4, "In SVD..reducing to dim %d" % reduced_dimension)
        log.print_matrix_info(logger, matrix_, 5, "Input matrix:")

        #TODO: IMPORTANT!! do the sign normalization COLUMN-wise!!!not
        #for the full matrix at once!!
        if reduced_dimension == 0:
            raise ValueError("Cannot reduce to dimensionality 0.")

        if isinstance(matrix_, SparseMatrix):
            result =  Linalg._sparse_svd(matrix_, reduced_dimension)
        elif isinstance(matrix_, DenseMatrix):
            result =  Linalg._dense_svd(matrix_, reduced_dimension)
        else:
            raise TypeError("expected Matrix type, received %s" % type(matrix_))

        log.print_matrix_info(logger, result[0], 5, "Resulting matrix U:")
        return result

    @staticmethod
    def ridge_regression(matrix_a , matrix_b, lambda_, intercept=False):
        #log.print_info(logger, "In Ridge regression..", 4)
        #log.print_matrix_info(logger, matrix_a, 5, "Input matrix A:")
        #log.print_matrix_info(logger, matrix_b, 5, "Input matrix B:")
        """
        Performs Ridge Regression.

        This method uses the general formula:
            :math:`X = (A^t A + \\lambda I)^{-1}A^t B`
        to solve the problem:
            :math:`X = argmin(||AX - B||_2 + \\lambda||X||_2)`

        Args:
            matrix_a: input matrix A, of type Matrix
            matrix_b: input matrix A, of type Matrix
            lambda_: scalar, lambda parameter
            intercept: bool. If True intercept is used. Optional, default False.

        Returns:
            solution X of type Matrix

        """
        
        matrix_a._assert_same_type(matrix_b)
        # TODO: check out where to define this assert
        assert_same_shape(matrix_a, matrix_b, 0)

        matrix_type = type(matrix_a)
        dim = matrix_a.shape[1]

        if intercept:
            matrix_a = matrix_a.hstack(matrix_type(np.ones((matrix_a.shape[0],
                                                             1))))
        lambda_diag = (lambda_ ) * matrix_type.identity(dim)

        if intercept:
            lambda_diag = padd_matrix(padd_matrix(lambda_diag, 0, 0.0), 1, 0.0)

        matrix_a_t = matrix_a.transpose()
        try:
            tmp_mat = Linalg.pinv(((matrix_a_t * matrix_a) + lambda_diag))
        except np.linalg.LinAlgError:
            print "Warning! LinAlgError"
            tmp_mat = matrix_type.identity(lambda_diag.shape[0])

        tmp_res = tmp_mat * matrix_a_t
        result = tmp_res * matrix_b

        #S: used in generalized cross validation, page 244 7.52 (YZ also used it)
        # S is defined in 7.31, page 232
        # instead of computing the matrix and then its trace, we can compute
        # its trace directly
        # NOTE when lambda = 0 we get out trace(S) = rank(matrix_a)

        dist = (matrix_a * result - matrix_b).norm()
        S_trace = matrix_a_t.multiply(tmp_res).sum()
        return result, S_trace, dist

    # This quantity is the "P_mu" from the article (Ji and Ye 2009)
    @staticmethod # numpy inputs
    def _intermediate_cost(matrix_a, matrix_b, new_W, old_W, mu):
        diff_W = new_W - old_W
        grad_f = Linalg._fitness_gradient(matrix_a, matrix_b, old_W)
        return np.real((Linalg._fitness(matrix_a, matrix_b, old_W) +
                np.trace(np.dot(diff_W.transpose(), grad_f)) +
                (mu/2)*Linalg._frobenius_norm_squared(diff_W)))

    @staticmethod # numpy inputs
    def _frobenius_norm_squared(W):
        W2 = np.multiply(W,W)
        return np.real(np.sum(W2))

    @staticmethod # numpy inputs
    def _fitness(inputs, outputs, A):
        return Linalg._frobenius_norm_squared(np.dot(inputs, A) - outputs)

    @staticmethod # numpy inputs
    def _fitness_gradient(inputs, outputs, A):
        return 2* (np.dot(inputs.transpose(), np.dot(inputs,A)) - np.dot(inputs.transpose(), outputs))


    @staticmethod # numpy inputs
    def _tracenorm(W):
        U, s, V = np.linalg.svd(W)
        return sum(s)

    @staticmethod # numpy inputs
    def _next_tracenorm_guess(matrix_a, matrix_b, lmbd, mu, current_W, at_times_a):
        # Computes the next estimate of A using the first gradient algorithm
        # from (Ji and Ye 2009)
        p = np.shape(matrix_a)[0]
        q = np.shape(matrix_a)[1]
        r = np.shape(matrix_b)[1]
        W = current_W

        matrix_a_t = np.transpose(matrix_a)
        utu = at_times_a
        uv = np.dot(matrix_a_t, matrix_b)

        gradient = np.dot(utu,W) - uv
        C = W - (1/mu) * gradient
        if lmbd == 0:
            return C, 0
        U, s, V = np.linalg.svd(C)

        s = s - (lmbd/(2*mu))*np.ones(np.shape(s)[0])
        sz = np.array([s, np.zeros(np.shape(s)[0])])
        final_s = sz.max(0)
        ssum = sum(final_s)
        lu = np.shape(U)[1]
        lv = np.shape(V)[0]
        S = np.zeros((lu, lv), dtype=complex)
        rk = min(lu, lv)
        S[:rk,:rk] = np.diag(final_s)
        return np.dot(U, np.dot(S, V)), ssum

    @staticmethod
    def _kronecker_product(matrix_a):
        """
        Computes the sum over the lines of A of the kronecker product of the line
        with itself
        """
        result = np.zeros([matrix_a.shape[1],matrix_a.shape[1]])
        for i in range(matrix_a.shape[0]):
            vec = matrix_a[i,:]
            vec_norm2 = Linalg._frobenius_norm_squared(vec)
            vec = np.array(vec)
            outer_prod = np.dot(vec.transpose(),vec)
            result += (1/vec_norm2) * outer_prod
        return result

    @staticmethod
    def kronecker_product(matrix_a):
        """
        Public version (using the Matrix interface) of the _kronecker_product function
        """
        m = DenseMatrix(matrix_a).mat
        return DenseMatrix(Linalg._kronecker_product(m))

    @staticmethod
    def tracenorm_regression(matrix_a , matrix_b, lmbd, iterations, intercept=False):
        #log.print_info(logger, "In Tracenorm regression..", 4)
        #log.print_matrix_info(logger, matrix_a, 5, "Input matrix A:")
        #log.print_matrix_info(logger, matrix_b, 5, "Input matrix B:")
        """
        Performs Trace Norm Regression.

        This method uses approximate gradient descent
        to solve the problem:
            :math:`X = argmin(||AX - B||_2 + \\lambda||X||_*)`
        where :math:`||X||_*` is the trace norm of :math:`X`, the sum of its
        singular values.
        It is implemented for dense matrices only.
        The algorithm is the Extended Gradient Algorithm from (Ji and Ye, 2009).

        Args:
            matrix_a: input matrix A, of type Matrix
            matrix_b: input matrix A, of type Matrix. If None, it is defined as matrix_a
            lambda_: scalar, lambda parameter
            intercept: bool. If True intercept is used. Optional, default False.

        Returns:
            solution X of type Matrix

        """

        if intercept:
            matrix_a = matrix_a.hstack(matrix_type(np.ones((matrix_a.shape[0],
                                                             1))))
        if matrix_b == None:
            matrix_b = matrix_a

        
        # TODO remove this
        matrix_a = DenseMatrix(matrix_a).mat
        matrix_b = DenseMatrix(matrix_b).mat

        # Matrix shapes
        p = matrix_a.shape[0]
        q = matrix_a.shape[1]
        assert_same_shape(matrix_a, matrix_b, 0)

        # Initialization of the algorithm
        W = (1.0/p)* Linalg._kronecker_product(matrix_a)

        # Sub-expressions reused at various places in the code
        matrix_a_t = matrix_a.transpose()
        at_times_a = np.dot(matrix_a_t, matrix_a)

        # Epsilon: to ensure that our bound on the Lipschitz constant is large enough
        epsilon_lbound = 0.05
        # Expression of the bound of the Lipschitz constant of the cost function
        L_bound = (1+epsilon_lbound)*2*Linalg._frobenius_norm_squared(at_times_a)
        # Current "guess" of the local Lipschitz constant
        L = 1.0
        # Factor by which L should be increased when it happens to be too small
        gamma = 1.2
        # Epsilon to ensure that mu is increased when the inequality hold tightly
        epsilon_cost = 0.00001
        # Real lambda: resized according to the number of training samples (?)
        lambda_ = lmbd*p
        # Variables used for the accelerated algorithm (check the original paper)
        Z = W
        alpha = 1.0
        # Halting condition
        epsilon = 0.00001
        last_cost = 1
        current_cost = -1
        linalg_error_caught = False

        costs = []
        iter_counter = 0
        while iter_counter < iterations and (abs((current_cost - last_cost)/last_cost)>epsilon) and not linalg_error_caught:
            sys.stdout.flush()
            # Cost tracking
            try:
                next_W, tracenorm = Linalg._next_tracenorm_guess(matrix_a, matrix_b, lambda_, L, Z, at_times_a)
            except LinAlgError:
                print "LinAlgError caught in trace norm regression"
                linalg_error_caught = True
                break

            last_cost = current_cost
            current_fitness = Linalg._fitness(matrix_a, matrix_b, next_W)
            current_cost = current_fitness + lambda_ * tracenorm
            if iter_counter > 0: # The first scores are messy
                cost_list =  [L, L_bound, current_fitness, current_cost]
                costs.append(cost_list)

            while (current_fitness + epsilon_cost >=
                    Linalg._intermediate_cost(matrix_a, matrix_b, next_W, Z, L)):
                if L > L_bound:
                    print "Trace Norm Regression: numerical error detected at iteration "+str(iter_counter)
                    break
                L = gamma * L
                try:
                    next_W, tracenorm = Linalg._next_tracenorm_guess(matrix_a, matrix_b, lambda_, L, Z, at_times_a)
                except LinAlgError:
                    print "LinAlgError caught in trace norm regression"
                    linalg_error_caught = True
                    break

                last_cost = current_cost
                current_fitness = Linalg._fitness(matrix_a, matrix_a, next_W)
                current_cost = current_fitness + lambda_*tracenorm

            if linalg_error_caught:
                break

            previous_W = W
            W = next_W
            previous_alpha = alpha
            alpha = (1.0 + sqrt(1.0 + 4.0*alpha*alpha))/2.0
            Z = W
            # Z = W + ((alpha - 1)/alpha)*(W - previous_W)
            iter_counter += 1

        sys.stdout.flush()
        W = np.real(W)
        return DenseMatrix(W), costs

    @classmethod
    def lstsq_regression(cls, matrix_a, matrix_b, intercept=False):
        """
        Performs Least Squares Regression.

        Solves the problem:

        :math:`X = argmin(||AX - B||_2)`

        Args:
            matrix_a: input matrix A, of type Matrix
            matrix_b: input matrix A, of type Matrix
            intercept: bool. If True intercept is used. Optional, False by default.

        Returns:
            solution X of type Matrix

        """

        matrix_a._assert_same_type(matrix_b)
        # TODO: check out where to define this assert
        assert_same_shape(matrix_a, matrix_b, 0)

        if intercept:
            matrix_a = matrix_a.hstack(type(matrix_a)(np.ones((matrix_a.shape[0],
                                                             1))))
        if isinstance(matrix_a, DenseMatrix):
            result = Linalg._dense_lstsq_regression(matrix_a, matrix_b)
        else:
            result = Linalg._sparse_lstsq_regression(matrix_a, matrix_b)

        return result

    @staticmethod
    def _dense_lstsq_regression(matrix_a , matrix_b):
        return DenseMatrix(Linalg._numpy_lstsq_regression(matrix_a, matrix_b))
        #return DenseMatrix(Linalg._scipy_lstsq_regression(matrix_a, matrix_b))

    @staticmethod
    def _sparse_lstsq_regression(matrix_a , matrix_b, intercept=False):
        return Linalg.ridge_regression(matrix_a, matrix_b, 0.0)[0]
        #return SparseMatrix(Linalg._dense_lstsq_regression(DenseMatrix(matrix_a),
        #                                      DenseMatrix(matrix_b)))

    @staticmethod
    def _numpy_lstsq_regression(matrix_a, matrix_b, rcond=-1):
        return np.linalg.lstsq(matrix_a.mat, matrix_b.mat, rcond)[0]

    @staticmethod
    def _scipy_lstsq_regression(matrix_a, matrix_b):
        return splinalg.lstsq(matrix_a.mat, matrix_b.mat)[0]

    @staticmethod
    def _sparse_svd(matrix_, reduced_dimension):
        #svds from scipy.sparse.linalg
        #RAISES ValueError if the rank is smaller than reduced_dimension + 1
        #TODO : fix this or replace with svdsparse
        #??? eIGENVALUES ARE NOT SORTED!!!!!!
        #IF EVER USE THIS; FIX THE PROBLEMS
        #u, s, vt = svds(matrix_.mat, False, True)
        """
        Patch

        Problem: sparsesvd sometimes returns fewer dimensions that requested.
        It will be no longer needs when sparsesvd will allow
        SVDLIBC parameters as an input (kappa parameter of SVDLIBC has to be
        larger than the default. e.g. 1E-05 instead of 1E-06)

        Current fix: ask for more dimensions and remove the unnecessary ones.
        """

        extra_dims = int(reduced_dimension/10)

        ut, s, vt = sparsesvd(matrix_.mat.tocsc(), reduced_dimension + extra_dims)

        u = SparseMatrix(ut.transpose())
        v = SparseMatrix(vt.transpose())

        no_cols = min(u.shape[1], reduced_dimension)
        u = u[:, 0:no_cols]
        v = v[:, 0:no_cols]

        Linalg._check_reduced_dim(matrix_.shape[1], u.shape[1], reduced_dimension)

        if not u.is_mostly_positive():
            u = -u
            v = -v

        return u, s[0:no_cols], v

    @staticmethod
    def _dense_svd(matrix_, reduced_dimension):

        print "Running dense svd"
        u, s, vt = np.linalg.svd(matrix_.mat, False, True)
        rank = len(s[s > Linalg._SVD_TOL])

        no_cols = min(u.shape[1], reduced_dimension, rank)
        u = DenseMatrix(u[:,0:no_cols])
        s = s[0:no_cols]
        v = DenseMatrix(vt[0:no_cols,:].transpose())

        Linalg._check_reduced_dim(matrix_.shape[1], u.shape[1], reduced_dimension)

        if not u.is_mostly_positive():
            u = -u
            v = -v

        return u, s, v

    @staticmethod
    def _check_reduced_dim(no_columns, reduced_dim, requested_reduced_dim):
        if requested_reduced_dim > no_columns:
            warn("Number of columns smaller than the reduced dimensionality requested: %d < %d. Truncating to %d dimensions (rank)." % (no_columns, requested_reduced_dim, reduced_dim))
        elif reduced_dim != requested_reduced_dim:
            warn("Returning %d dimensions instead of %d." % (reduced_dim, requested_reduced_dim))

    @staticmethod
    def _nmf_nlssubprob(v, w, w_t, h_init, tol, maxiter):
        """
        h, grad: output solution and gradient
        iteration: #iterations used
        v, w: constant matrices
        h_init: initial solution
        tol: stopping tolerance
        maxiter: limit of iterations
        """
        h = h_init
        w_t_v = w_t * v
        w_t_w = w_t * w

        alpha = Linalg._NMF_ALPHA
        beta = Linalg._NMF_BETA

        #sub_loop_time = time()

        for iteration in xrange(1, maxiter):
            grad = w_t_w * h - w_t_v

            # search step size
            for inner_iter in xrange(1, 20):
                hn = h - alpha * grad
                hn = hn.get_non_negative()
                d = hn - h
                gradd = grad.multiply(d).sum()
                dQd = (w_t_w * d).multiply(d).sum()
                suff_decr = 0.99 * gradd + 0.5 * dQd < 0
                if inner_iter == 1:
                    decr_alpha = not suff_decr
                    hp = h
                if decr_alpha:
                    if suff_decr:
                        h = hn
                        break
                    else:
                        alpha = alpha * beta
                else:
                    if not suff_decr or hp.all_close(hn):
                        h = hp
                        break
                    else:
                        alpha = alpha / beta
                        hp = hn

        return h, grad, iteration

    @staticmethod
    def nmf(v, w_init, h_init):
        """
        Performs Non-negative Matrix Factorization.

        It solves the problem:
        :math:`W,H = argmin(||X - WH||_2)` such that W and H are non-negative matrices.

        Args:
            w_init: initial value for matrix W, type Matrix
            h_init: initial value for matrix H, type Matrix

        Returns:
            W, H <Matrix>: where W, H solve the NMF problem stated above.

        """

        log.print_info(logger, 4, "In NMF..reducing to dim %d" % w_init.shape[1])
        log.print_matrix_info(logger, w_init, 5, "W init matrix:")
        log.print_matrix_info(logger, h_init, 5, "H init matrix:")

        if not isinstance(v, Matrix):
            raise TypeError("expected Matrix type, received %s" % type(v))
        w = w_init
        h = h_init
        init_time = time()

        wt = w.transpose()
        ht = h.transpose()
        vt = v.transpose()
        gradW = (w * (h * ht)) - (v * ht)
        gradH = ((wt * w) * h) - (wt * v)

        gradW_norm = gradW.norm()
        gradH_norm = gradH.norm()
        initgrad = sqrt(pow(gradW_norm, 2) + pow(gradH_norm, 2))

        #print 'Init gradient norm %f' % initgrad
        tolW = max(Linalg._NMF_MIN_TOL, Linalg._NMF_TOL) * initgrad
        tolH = tolW

        #loop_time = init_time
        for iteration in xrange(1, Linalg._NMF_MAX_ITER):
            log.print_info(logger, 5, "Iteration: %d(%d)" % (iteration, Linalg._NMF_MAX_ITER))

            if time() - init_time > Linalg._NMF_TIME_LIMIT:
                break

            w, gradW, iterW = Linalg._nmf_nlssubprob(vt, h.transpose(), h,
                                              w.transpose(), tolW,
                                              Linalg._NMF_MAX_ITER_SUBPROB)
            old_w = w
            w = w.transpose()
            gradW = gradW.transpose()

            if iterW == 1:
                tolW = Linalg._NMF_TOL_DECREASE_FACTOR * tolW

            h, gradH, iterH = Linalg._nmf_nlssubprob(v, w, old_w, h, tolH,
                                              Linalg._NMF_MAX_ITER_SUBPROB)

            if iterH == 1:
                tolH = Linalg._NMF_TOL_DECREASE_FACTOR * tolH

        log.print_matrix_info(logger, w, 5, "Return W matrix:")
        log.print_matrix_info(logger, h, 5, "Return H matrix:")
        return w, h

    @staticmethod
    def pinv(matrix_):
        """
        Computes the pseudo-inverse of a matrix.

        Args:
            matrix_: input matrix, of type Matrix

        Returns:
            Pseudo-inverse of input matrix, of type Matrix

        Raises:
            TypeError, if input is not of type Matrix
        """
        if isinstance(matrix_, SparseMatrix):
            return Linalg._sparse_pinv(matrix_)
        elif isinstance(matrix_, DenseMatrix):
            return Linalg._dense_pinv(matrix_)
        else:
            raise TypeError("expected Matrix type, received %s" % type(matrix_))

    @staticmethod
    def _dense_pinv(matrix_):
        return DenseMatrix(np.linalg.pinv(matrix_.mat))

    @staticmethod
    def _sparse_pinv(matrix_):
        # TODO: implement pinv
        return SparseMatrix(np.linalg.pinv(matrix_.mat.todense()))
