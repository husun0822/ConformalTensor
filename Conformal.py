from LRIsing import *
from SimIsing import *
import wquantiles as wq
import pandas as pd
import tensorly as tl
from functools import reduce
from tensorly.cp_tensor import cp_to_tensor


def EntryCI(ncs_cal, w_cal, w_test, target_q=[0.90], two_sided=False):
    """
    Construct entry-wise confidence interval.

    :param ncs_cal: 1-D numpy array, calibration set non-conformity score
    :param w_cal: 1-D numpy array, calibration set un-normalized weight
    :param w_test: 1-D numpy array, test set un-normalized weight
    :param target_q: list of float, target nominal coverage level
    :param two_sided: bool, if True, the confidence interval is set to be asymmetric
    :return: CI_tensor: a tensor of size Q * N * 2, with Q = len(target_q), N = w_test.shape[0], and this tensor contains the corresponding non-conformity score for constructing the confidence band
    """

    # sort the calibration set score
    sum_w_cal = w_cal.sum()
    sorted_index = np.argsort(ncs_cal)
    ncs_cal, w_cal = ncs_cal[sorted_index], w_cal[sorted_index]
    cumsum_w_cal = np.cumsum(w_cal) / sum_w_cal  # empirical CDF
    ncs_cal, cumsum_w_cal = np.concatenate((np.array([0]), ncs_cal)), np.concatenate(
        (np.array([0]), cumsum_w_cal)
    )

    if not two_sided:
        CI_tensor = np.zeros(
            (len(target_q), w_test.shape[0])
        )  # non-conformity score cutoff tensor
        for i, q in enumerate(target_q):
            query_quantile = q / (1 - (w_test / (w_test + sum_w_cal)))
            CI_tensor[i, query_quantile >= 1] = np.inf

            # find the empirical quantile
            qs = query_quantile[query_quantile < 1]
            qs_index = np.searchsorted(cumsum_w_cal, qs)
            CI_tensor[i, query_quantile < 1] = ncs_cal[qs_index]
    else:
        CI_tensor = np.zeros(
            (len(target_q), w_test.shape[0], 2)
        )  # non-conformity score cutoff tensor
        for i, q in enumerate(target_q):
            query_quantile_left = 0.5 * (1 - q) / (1 - (w_test / (w_test + sum_w_cal)))
            query_quantile_right = 0.5 * (1 + q) / (1 - (w_test / (w_test + sum_w_cal)))
            CI_tensor[i, query_quantile_right >= 1, 1] = np.inf

            # find the empirical left quantile
            qs = query_quantile_left
            qs_index = np.searchsorted(cumsum_w_cal, qs, side="right")
            CI_tensor[i, :, 0] = ncs_cal[qs_index]

            # find the empirical right quantile
            qs = query_quantile_right[query_quantile_right < 1]
            qs_index = np.searchsorted(cumsum_w_cal, qs)
            CI_tensor[i, query_quantile_right < 1, 1] = ncs_cal[qs_index]

    return CI_tensor


class TC_RGrad:
    """
    A class for tensor completion with Riemannian Gradient Descent (RGrad).
    """

    def __init__(self, true_X=None):
        """
        Intialize the class.

        :param true_X: optional true tensor, used for evaluation only.
        """
        self.true_X = true_X

    def fit(self, X, rank, method="tensor-train", config=None):
        """
        A functional wrapper for fitting Tensor Completion.

        :param X: input tensor with missing entries encoded as nan.
        :param rank: rank of the completed tensor.
        :param method: low-rank assumption of the completed tensor, could either be tensor-train or tucker or CP.
        :param config: other tensor completion configs, including the following optional fields:
            'seed':int, random seed for initialization
            'beta': float, the stepsize for the RGrad, default is 0.1
            'max_iter': int, the maximum number of iterations, default is 200
            'thres': float, the threshold on relative squared error (RSE) change that stops the algorithm early if converged, default is 1e-4
            'verbose': bool, if True, print out the iteration counter, default is True
        :return: a completed tensor in self.B and also self.AIC and self.BIC for the information criteria of the fit
        """

        if method == "tensor-train" and not X.ndim == len(rank) + 1:
            raise Exception(
                "Wrong tensor-train rank! Please provide a rank parameter with length equal to the number of modes minus 1."
            )

        if method == "tucker" and not X.ndim == len(rank):
            raise Exception(
                "Wrong tucker rank! Please provide a rank parameter with length equal to the number of modes."
            )

        if method == "CP" and not isinstance(rank, int):
            raise Exception("Wrong CP rank! Please provide an integer.")

        if method == "tensor-train":
            self.fit_TT(X, rank, config)
        elif method == "tucker":
            self.fit_TK(X, rank, config)
        elif method == "CP":
            self.fit_CP(X, rank, config)
        else:
            raise ValueError(
                "Unknown method! Please choose from 'tensor-train' or 'tucker' or 'CP'."
            )

    def fit_TT(self, X, rank, config):
        """
        Tensor completion with low tensor-train rank tensors.

        :param X: input tensor with missing entries encoded as nan.
        :param rank: rank of the completed tensor.
        :param config: other tensor completion configs, including the following optional fields:
            'seed':int, random seed for initialization
            'beta': float, the stepsize for the RGrad, default is 0.1
            'max_iter': int, the maximum number of iterations, default is 200
            'thres': float, the threshold on relative squared error (RSE) change that stops the algorithm early if converged, default is 1e-4
            'verbose': bool, if True, print out the iteration counter, default is True
        :return: a completed tensor in self.B and also self.AIC and self.BIC for the information criteria of the fit
        """

        d = X.shape
        K = X.ndim
        if K < 3:
            raise TypeError("Input tensor should contain at least three modes!")
        tt_rank = [1] + list(rank) + [1]  # full tensor-train rank

        # ----- Initialization ----- #
        if "seed" in config:
            np.random.seed(config["seed"])
        X_zero = np.copy(X)
        X_zero[np.isnan(X)] = 0.0
        T = tensor_train(
            X_zero + np.random.normal(loc=0, scale=0.2, size=d), rank=tt_rank
        )  # T is a tensorly tensor-train object, and is randomly initialized for the algorithm
        B = tt_to_tensor(T)
        W = np.logical_not(np.isnan(X)).astype(int)  # missingness mask

        # evaluate initial loss
        residual = X - B
        loss_hist = [0.5 * np.sum(residual[W == 1] ** 2)]

        # ----- Riemannian Gradient Descent ----- #
        beta = 0.1 if "beta" not in config else config["beta"]  # step size
        l, delta = 0, 1
        l_max = 200 if "max_iter" not in config else config["max_iter"]
        threshold = 1e-4 if "thres" not in config else config["thres"]
        verbose = True if "verbose" not in config else config["verbose"]
        RSE_hist = []

        if self.true_X is not None:
            RSE_X = RSE(X, self.true_X)
            RSE_hist.append(RSE_X)

        while (l <= l_max) and (delta > threshold):
            if l % 100 == 0 and l > 0 and verbose:
                print(f"iter {l}...")

            # Step I: compute vanilla gradient
            G = -residual
            G[W == 0] = 0

            # Step II: tangent space projected gradient descent
            if K > 3:
                # compute projected gradient
                projG = np.zeros_like(G)

                # compute first K-1 projection components
                for k in range(K - 1):
                    TT_factors = [f for f in T]
                    LT = left_unfold(TT_factors[k])
                    orthogonal_component = np.eye(tt_rank[k] * d[k]) - LT @ LT.T
                    left_part, right_part = tt_left_part(T, k - 1), tt_right_part(
                        T, k + 1
                    )
                    LY = (
                        orthogonal_component
                        @ np.kron(left_part, np.eye(d[k])).T
                        @ tensor_sep(G, k)
                        @ right_part.T
                        @ np.linalg.inv(right_part @ right_part.T)
                    )
                    TT_factors[k] = left_unfold_rev(LY, TT_factors[k].shape)
                    projG += tt_to_tensor(TT_factors)

                # compute the last projection components
                TT_factors = [f for f in T]
                LY = np.kron(tt_left_part(T, K - 2), np.eye(d[K - 1])).T @ tensor_sep(
                    G, K - 1
                )
                TT_factors[-1] = left_unfold_rev(LY, TT_factors[-1].shape)
                projG += tt_to_tensor(TT_factors)
            elif K <= 3:
                # compute projected gradient faster for 3-mode tensors
                T1, T2, T3 = T[0].squeeze(), T[1], T[2].squeeze()
                T3_T3T = T3 @ T3.T

                # update Y1
                T2_1 = unfold(T2, mode=0)
                C = unfold(mode_dot(G, T3, mode=2), mode=0)
                U1 = unfold(mode_dot(T2, T3.T, mode=2), mode=0)
                Y1 = T1.T @ C
                Y1 = C - T1 @ Y1
                Y1 = Y1 @ T2_1.T @ inv(U1 @ U1.T)

                # update Y2
                T2_3 = unfold(T2, mode=2)
                C = unfold(multi_mode_dot(G, [T1.T, T3], modes=[0, 2]), mode=2).T
                Y2 = T2_3 @ C
                Y2 = C - T2_3.T @ Y2
                Y2 = Y2 @ inv(T3_T3T)
                Y2 = np.reshape(Y2, (rank[0], d[1], rank[1]))

                # update Y3
                u, s, v = np.linalg.svd(T1, full_matrices=False)
                left_tensor = mode_dot(T2, (v.T @ np.diag(s)).T, mode=0)
                right_tensor = mode_dot(G, u.T, mode=0)
                Y3 = unfold(left_tensor, mode=2) @ unfold(right_tensor, mode=2).T
                projG = (
                    multi_mode_dot(T2, [Y1, T3.T], modes=[0, 2])
                    + multi_mode_dot(Y2, [T1, T3.T], modes=[0, 2])
                    + multi_mode_dot(T2, [T1, Y3.T], modes=[0, 2])
                )

                # descent along the projected gradient direction
            B_tilde = B - beta * projG

            # Step III: retraction
            T = tensor_train(B_tilde, rank=tt_rank)
            B_new = tt_to_tensor(T)

            # track convergence
            delta = ((B_new - B) ** 2).sum() / (B**2).sum()
            B = np.copy(B_new)
            l += 1
            if self.true_X is not None:
                RSE_X = RSE(X, self.true_X)
                RSE_hist.append(RSE_X)

            # evaluate initial loss
            residual = X - B
            loss_hist.append(0.5 * np.sum(residual[W == 1] ** 2))

        if verbose:
            print(f"RGrad terminates at iter {l}.")

        # return model parameters & fitting history
        self.B = B
        self.loss_hist = loss_hist
        self.RSE_hist = RSE_hist

        # entry-wise uncertainty estimator
        self.S = None

        # compute model BIC
        n_param = 0
        for k in range(K):
            n_param += d[k] * tt_rank[k] * tt_rank[k + 1]
            if k < K - 1:
                n_param -= tt_rank[k + 1] ** 2  # adjust for left-orthogonality
        d_obs = np.sum(W)
        sigma_hat = np.sqrt(2 * loss_hist[-1] / (d_obs - 1))
        NLL = d_obs * np.log(sigma_hat) + loss_hist[-1] / (sigma_hat**2)
        self.BIC = 2 * NLL + n_param * np.log(np.prod(d))
        self.AIC = 2 * NLL + 2 * n_param

    def fit_TK(self, X, rank, config):
        """
        Tensor completion with low Tucker rank tensors.

        :param X: input tensor with missing entries encoded as nan.
        :param rank: rank of the completed tensor.
        :param config: other tensor completion configs, including the following optional fields:
            'seed':int, random seed for initialization
            'beta': float, the stepsize for the RGrad, default is 0.1
            'max_iter': int, the maximum number of iterations, default is 200
            'thres': float, the threshold on relative squared error (RSE) change that stops the algorithm early if converged, default is 1e-4
            'verbose': bool, if True, print out the iteration counter, default is True
        :return: a completed tensor in self.B and also self.AIC and self.BIC for the information criteria of the fit
        """

        d = X.shape
        K = X.ndim
        if K < 3:
            raise TypeError("Input tensor should contain at least three modes!")
        tk_rank = list(rank)

        # ----- Initialization ----- #
        if "seed" in config:
            np.random.seed(config["seed"])
        X_zero = np.copy(X)
        X_zero[np.isnan(X)] = 0.0
        C, U = tucker(
            X_zero + np.random.normal(loc=0, scale=0.2, size=d), rank=tk_rank
        )  # T is a tensorly tensor-train object, a randomly initialized tensor
        B = tucker_to_tensor((C, U))
        W = np.logical_not(np.isnan(X)).astype(int)

        # evaluate initial loss
        residual = X - B
        loss_hist = [0.5 * np.sum(residual[W == 1] ** 2)]

        # ----- Riemannian Gradient Descent ----- #
        beta = 0.1 if "beta" not in config else config["beta"]  # step size
        l, delta = 0, 1
        l_max = 200 if "max_iter" not in config else config["max_iter"]
        threshold = 1e-4 if "thres" not in config else config["thres"]
        verbose = True if "verbose" not in config else config["verbose"]
        RSE_hist = []

        while (l <= l_max) and (delta > threshold):
            if l % 100 == 0 and l > 0 and verbose:
                print(f"iter {l}...")

            # Step I: compute vanilla gradient
            G = -residual
            G[W == 0] = 0

            # Step II: tangent space projected gradient descent
            UT = [mat.T for mat in U]
            G_core = tucker_to_tensor((G, UT))
            projG = tucker_to_tensor((G_core, U))

            for k in range(K):
                Uk = copy.deepcopy(U)
                Uk[k] = np.eye(d[k])
                UkT = [mat.T for mat in Uk]
                C_k = unfold(C, k)
                C_pinv = C_k.T @ inv(C_k @ C_k.T)
                Uk[k] = (
                    unfold(tucker_to_tensor((G, UkT)), k) - U[k] @ unfold(G_core, k)
                ) @ C_pinv
                projG += tucker_to_tensor((C, Uk))

            # descent along the projected gradient direction
            B_tilde = B - beta * projG

            # Step III: retraction
            C, U = tucker(B_tilde, rank=tk_rank)
            B_new = tucker_to_tensor((C, U))

            # track convergence
            delta = ((B_new - B) ** 2).sum() / (B**2).sum()
            B = np.copy(B_new)
            l += 1
            if self.true_X is not None:
                RSE_X = RSE(X, self.true_X)
                RSE_hist.append(RSE_X)

            # evaluate initial loss
            residual = X - B
            loss_hist.append(0.5 * np.sum(residual[W == 1] ** 2))

        if verbose:
            print(f"RGrad terminates at iter {l}.")

        # return model parameters & fitting history
        self.B = B
        self.loss_hist = loss_hist
        self.RSE_hist = RSE_hist

        # compute model BIC
        d_obs = np.sum(W)
        sigma_hat = np.sqrt(2 * loss_hist[-1] / (d_obs - 1))
        NLL = d_obs * np.log(sigma_hat) + loss_hist[-1] / (sigma_hat**2)
        n_param = np.prod(np.array(tk_rank))
        for k in range(K):
            n_param += (d[k] - tk_rank[k]) * tk_rank[k]
        self.BIC = 2 * NLL + n_param * np.log(np.prod(d))
        self.AIC = 2 * NLL + 2 * n_param

        # element-wise uncertainty point estimate
        v_list = []
        for U_k in U:
            v_list.append(np.linalg.norm(U_k @ U_k.T, axis=1))
        self.S = (
            reduce(np.multiply.outer, v_list) * sigma_hat * np.sqrt(np.prod(d) / d_obs)
        )

    def fit_CP(self, X, rank, config):
        """
        Fit CP decomposition for tensor completion. CP decomposition is fitted with tensorly package instead of RGrad.

        :param X: input tensor with missing entries encoded as nan.
        :param rank: rank of the completed tensor.
        :param config: other tensor completion configs, including the following optional fields:
            'seed':int, random seed for initialization
            'beta': float, the stepsize for the RGrad, default is 0.1
            'max_iter': int, the maximum number of iterations, default is 200
            'thres': float, the threshold on relative squared error (RSE) change that stops the algorithm early if converged, default is 1e-4
        :return: a completed tensor in self.B and also self.AIC and self.BIC for the information criteria of the fit
        """

        mask = ~np.isnan(X)
        X_filled = np.nan_to_num(X, nan=0.0)

        # Fit CP decomposition
        n_iter_max = 1000 if "max_iter" not in config.keys() else config["max_iter"]
        tol = 1e-7 if "thres" not in config.keys() else config["thres"]
        cp_tensor = parafac(
            X_filled, rank=rank, mask=mask, n_iter_max=n_iter_max, tol=tol
        )
        X_completed = cp_to_tensor(cp_tensor)
        self.B = X_completed  # the completed tensor
        self.S = None  # the entrywise analytical uncertainty (to be added)

        # compute AIC/BIC
        X_obs, X_hat_obs = X[mask], X_completed[mask]

        # Compute residual sum of squares (RSS)
        rss = np.sum((X_obs - X_hat_obs) ** 2)
        n_obs = np.sum(mask)
        sigma2 = rss / n_obs

        # Compute log-likelihood
        LL = -0.5 * n_obs * np.log(sigma2) - rss / (2 * sigma2)

        # Number of parameters in CP rank-r decomposition:
        # For CP, parameters = r * (I + J + K) if X is 3D tensor of shape I x J x K
        tensor_shape = X.shape
        n_params = rank * sum(tensor_shape)  # excludes scaling ambiguity

        # Compute AIC and BIC
        self.AIC = 2 * n_params - 2 * LL
        self.BIC = np.log(n_obs) * n_params - 2 * LL


class ConfTC:
    def __init__(
        self,
        g_func="product",
        h_func="logit",
        g_params=None,
        h_params=None,
        nb="1-nb",
        true_B=None,
        q=0.7,
    ):
        """
        A conformalized tensor completion class.

        :param g_func: a bivariate function specifying the coupling strength of neighboring entries
        :param h_func: a univariate function specifying the individual entry missing propensity
        :param g_params: hyperparameter for the g_func
        :param h_params: hyperparameter for the h_func
        :param nb: neighborhood structure, currently supports the following options
            - "1-nb": directly neighbors that differ their indices on only 1 axis
            - "1-nb-spatial": direct neighbors on the spatial dimensions, not including the temporal dimension (last axis)
            - "cube-nb": all entries within +/- 1 distance on each axis are included
        :param true_B: the ground truth tensor parameter, used for evaluation purpose
        :param q: splitting probability for train and calibration set
        """

        self.TenIsing = TensorIsing(
            g_func=g_func, h_func=h_func, g_params=g_params, h_params=h_params, nb=nb
        )
        self.g_func = g_func
        self.h_func = h_func
        self.g_params = g_params
        self.h_params = h_params
        self.nb = nb
        self.true_B = true_B
        self.comp_model = None
        self.decomp_model = None
        self.q = q

    def fit(
        self,
        X,
        train,
        tc_params,
        dc_params,
        weight="unweighted",
        normalized_ncs=False,
        two_sided=False,
        invtemp=1 / 15,
        target_q=[0.90],
        seed=1,
    ):
        """
        Fit Conformalized Tensor Completion
        """

        d = X.shape
        if "true_X" not in tc_params:
            raise Exception("Need the ground truth tensor for validation!")

        # train-calibration split
        W = np.logical_not(np.isnan(X))
        cal = np.logical_and(W == 1, train == 0)
        test = 1 - (train + cal)
        train, cal, test = train.astype(int), cal.astype(int), test.astype(int)
        W = 2 * W - 1
        W_train = np.copy(W)
        W_train[np.logical_not(train)] = -1

        # run tensor completion
        X_train = np.copy(X)
        X_train[cal == 1] = np.nan
        comp_model = TC_RGrad(true_X=tc_params["true_X"])
        comp_model.fit(
            X_train,
            rank=tc_params["rank"],
            method=tc_params["method"],
            config={"max_iter": 500, "thres": 1e-6, "verbose": False, "seed": 2024},
        )
        X_hat = comp_model.B
        S_hat = comp_model.S
        self.comp_model = comp_model

        # get non-conformity score
        if two_sided:
            ncs = (X - X_hat)[cal == 1]
        else:
            ncs = np.abs((X - X_hat)[cal == 1])

        if normalized_ncs and S_hat is not None:
            ncs = ncs / S_hat[cal == 1]

        # non-conformity score cutoff for all test points
        if two_sided:
            CI_tensor = np.zeros((len(target_q), np.sum(test), 2))
        else:
            CI_tensor = np.zeros((len(target_q), np.sum(test)))

        if weight == "unweighted":
            # unweighted conformal prediction
            ncs = np.concatenate((ncs, np.array([np.inf])))
            if two_sided:
                qs_left = np.quantile(ncs, q=[(1 - qs) / 2 for qs in target_q])
                qs_right = np.quantile(ncs, q=[(1 + qs) / 2 for qs in target_q])
                for i, q in enumerate(qs_left):
                    CI_tensor[i, :, 0] = q
                for i, q in enumerate(qs_right):
                    CI_tensor[i, :, 1] = q
            else:
                qs = np.quantile(ncs, q=target_q)
                for i, q in enumerate(qs):
                    CI_tensor[i, :] = q
        elif weight == "oracle":
            P = self.TenIsing.CondProb(W, self.true_B, invtemp=invtemp)
            p_cal = np.clip(P[cal == 1], 1e-4, 1 - 1e-4)
            w_cal = (1 - p_cal) / p_cal
            p_test = np.clip(P[test == 1], 1e-4, 1 - 1e-4)
            w_test = (1 - p_test) / p_test
            CI_tensor = EntryCI(
                ncs, w_cal, w_test, target_q=target_q, two_sided=two_sided
            )
        elif weight == "RGrad":
            decomp_model = LRIsing(
                g_func=self.g_func,
                h_func=self.h_func,
                g_params=self.g_params,
                h_params=self.h_params,
                nb=self.nb,
                true_B=self.true_B,
                true_temp=1 / invtemp,
            )
            decomp_model.fit(
                W_train,
                rank=dc_params["rank"],
                q=self.q,
                method=dc_params["method"],
                config={
                    "max_iter": 2000,
                    "thres": 1e-3,
                    "beta": 0.1,
                    "verbose": False,
                    "print_frequency": 1000,
                    "backtracking": False,
                    "armijo_threshold": 1e-4,
                    "projected_grad": False,
                },
            )
            P = self.TenIsing.CondProb(W, decomp_model.B, invtemp=invtemp)
            p_cal = np.clip(P[cal == 1], 1e-4, 1 - 1e-4)
            w_cal = (1 - p_cal) / p_cal
            p_test = np.clip(P[test == 1], 1e-4, 1 - 1e-4)
            w_test = (1 - p_test) / p_test
            CI_tensor = EntryCI(
                ncs, w_cal, w_test, target_q=target_q, two_sided=two_sided
            )
            self.decomp_model = decomp_model
        elif "External" in weight:
            if "fitted_B" not in dc_params:
                raise Exception(
                    "Please provide pre-fitted tensor parameter when using weight='External-Bernoulli'!"
                )

            g_model = "zero" if "Bernoulli" in weight else "product"
            T_Ising = TensorIsing(
                g_func=g_model,
                h_func=self.h_func,
                g_params=self.g_params,
                h_params=self.h_params,
                nb=self.nb,
            )
            P = T_Ising.CondProb(W, dc_params["fitted_B"], invtemp=invtemp)
            p_cal = np.clip(P[cal == 1], 1e-4, 1 - 1e-4)
            w_cal = (1 - p_cal) / p_cal
            p_test = np.clip(P[test == 1], 1e-4, 1 - 1e-4)
            w_test = (1 - p_test) / p_test
            CI_tensor = EntryCI(
                ncs, w_cal, w_test, target_q=target_q, two_sided=two_sided
            )

        if normalized_ncs and S_hat is not None:
            if two_sided:
                CI_tensor[:, :, 0] = CI_tensor[:, :, 0] * np.tile(
                    S_hat[test == 1], (len(target_q), 1)
                )
                CI_tensor[:, :, 1] = CI_tensor[:, :, 1] * np.tile(
                    S_hat[test == 1], (len(target_q), 1)
                )
            else:
                CI_tensor = CI_tensor * np.tile(S_hat[test == 1], (len(target_q), 1))

        # evaluate the coverage and confidence interval width
        cov_probs, inf_probs, widths = [], [], []
        test_residual = (tc_params["true_X"] - X_hat)[test == 1]
        for i, q in enumerate(target_q):
            if two_sided:
                left_cutoff, right_cutoff = CI_tensor[i, :, 0], CI_tensor[i, :, 1]
                cov_probs.append(
                    np.mean(
                        np.logical_and(
                            test_residual >= left_cutoff, test_residual <= right_cutoff
                        )
                    )
                )
                inf_probs.append(
                    np.mean(
                        np.logical_or(np.isinf(right_cutoff), np.isinf(left_cutoff))
                    )
                )
                is_finite = np.logical_and(
                    np.logical_not(np.isinf(right_cutoff)),
                    np.logical_not(np.isinf(left_cutoff)),
                )
                widths.append(np.mean(right_cutoff[is_finite] - left_cutoff[is_finite]))
            else:
                cutoff = CI_tensor[i, :]
                cov_probs.append(np.mean(np.abs(test_residual) <= cutoff))
                inf_probs.append(np.mean(np.isinf(cutoff)))
                widths.append(2 * np.mean(cutoff[np.logical_not(np.isinf(cutoff))]))

        return cov_probs, widths, inf_probs
