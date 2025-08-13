import sys
import numpy as np
import pandas as pd
import wquantiles as wq
from LRIsing import *
from Conformal import TC_RGrad, ConfTC, EntryCI
from scipy.io import loadmat


def BinaryTEC(day=1, rank=3, model="Bernoulli", neighbor="1-nb", q=0.7):
    """
    code for conformalized tensor completion with the TEC data
    """
    np.random.seed(day)
    method = model

    # configure the algorithm
    if model == "Bernoulli":
        g_func, h_func, model_abbr = "zero", "logit", "bern"
    elif model == "Ising-NoExt":
        g_func, h_func, model_abbr = "product", "zero", "ising_noext"
    elif model == "Ising":
        g_func, h_func, model_abbr = "product", "logit", "ising"
    else:
        raise Exception(
            "Unknown specification of the model. Please choose from 'Bernoulli', 'Ising-NoExt' and 'Ising'."
        )

    if neighbor == "1-nb":
        nb_abbr = "1nb"
    elif neighbor == "1-nb-spatial":
        nb_abbr = "1nb_spat"
    else:
        raise Exception(
            "Unknown neighborhood of the model. Please choose from '1-nb' and '1-nb-spatial'."
        )

    # load the TEC data
    day = str(day)
    day = day if len(day) == 2 else "0" + day
    data = loadmat(f"../data_JCGS/application/1709{day}.mat")
    mad, vista, dtec = data["mad"], data["vista"], data["dtec"]
    mad, vista, dtec = (
        np.flip(mad, axis=0),
        np.flip(vista, axis=0),
        np.flip(dtec, axis=0),
    )  # flip for easier visualization

    # train, calibration split
    W = np.logical_not(np.isnan(mad))
    train_mask = np.random.uniform(size=W.shape) <= q  # training set mask
    train, cal = np.logical_and(W, train_mask), np.logical_and(
        W, np.logical_not(train_mask)
    )
    W, train, cal = 2 * W.astype(int) - 1, train.astype(int), cal.astype(int)

    # run the binary tensor decomposition with Riemannian gradient descent (RGrad)
    model = LRIsing(
        g_func=g_func,
        h_func=h_func,
        g_params={"temp": 15, "scale": 1 / 5},
        h_params=None,
        nb=neighbor,
        true_B=None,
    )
    model.fit(
        W,
        rank=(rank,) * 2,
        method="tensor-train",
        config={"max_iter": 500, "thres": 1e-6, "beta": 0.1},
    )

    res = [[day, rank, method, neighbor, model.AIC, model.BIC]]
    df = pd.DataFrame(
        data=res, columns=["day", "rank", "model", "neighbor", "AIC", "BIC"]
    )
    df.to_csv(
        f"../results_JCGS/application/summary_statistics/decomposition_model/{day}_{rank}_{model_abbr}_{nb_abbr}.csv",
        index=False,
    )
    np.savez(
        f"../results_JCGS/application/decomposition_result/{day}_{rank}_{model_abbr}_{nb_abbr}.npz",
        B=model.B,
        cal=cal,
    )


def CompleteTEC(day=1, rank=3, q=0.7, method="tensor-train"):
    np.random.seed(day)

    # load the TEC data
    day = str(day)
    day = day if len(day) == 2 else "0" + day
    data = loadmat(f"../data_JCGS/application/1709{day}.mat")
    mad, vista, dtec = data["mad"], data["vista"], data["dtec"]
    mad, vista, dtec = (
        np.flip(mad, axis=0),
        np.flip(vista, axis=0),
        np.flip(dtec, axis=0),
    )  # flip for easier visualization

    # get the training data
    W = np.logical_not(np.isnan(mad))
    train_mask = np.random.uniform(size=vista.shape) <= q  # training set mask
    train = np.logical_and(W, train_mask).astype(int)
    X_train = np.copy(vista)
    X_train[train == 0] = np.nan

    # box-cox transformation
    X_train = (X_train**0.1 - 1) / 0.1
    X_train_mean, X_train_sd = np.mean(X_train[train == 1]), np.std(X_train[train == 1])
    X_train = (X_train - X_train_mean) / X_train_sd

    # fit the tensor completion model
    working_rank = (rank,) * 2 if method == "tensor-train" else (rank,) * 3
    comp_model = TC_RGrad(true_X=None)
    comp_model.fit(
        X_train,
        rank=working_rank,
        method=method,
        config={"max_iter": 500, "thres": 1e-6, "verbose": False, "seed": 2024},
    )

    Xhat = comp_model.B
    Xhat = Xhat * X_train_sd + X_train_mean
    Xhat = (Xhat * 0.1 + 1) ** 10

    # save the completion result
    method_abbr = "tt" if method == "tensor-train" else "tk"
    df = pd.DataFrame(
        data=[[day, rank, method, comp_model.AIC, comp_model.BIC]],
        columns=["day", "rank", "method", "AIC", "BIC"],
    )
    df.to_csv(
        f"../results_JCGS/application/summary_statistics/completion_model/{day}_{rank}_{method_abbr}.csv",
        index=False,
    )
    np.savez(
        f"../results_JCGS/application/completion_result/{day}_{rank}_{method_abbr}.npz",
        Xhat=Xhat,
    )


# def ConformalTEC(day=1, target_coverage=np.linspace(0.8, 0.99, 20)):
#     '''
#     code for conformal inference with the TEC data
#     '''
#     # look up table for the best-fitting rank
#     df = pd.read_csv(f"../data_JCGS/application/decomposition_rank_table.csv")
#     du = pd.read_csv(f"../data_JCGS/application/completion_rank_table.csv")
#
#     # load the TEC data
#     day = str(day)
#     day = day if len(day) == 2 else "0" + day
#     data = loadmat(f"../data_JCGS/application/1709{day}.mat")
#     mad, vista, dtec = data['mad'], data['vista'], data['dtec']
#     mad, vista, dtec = np.flip(mad, axis=0), np.flip(vista, axis=0), np.flip(dtec,
#                                                                              axis=0)  # flip for easier visualization
#     W = 2 * np.logical_not(np.isnan(mad)).astype(int) - 1  # binary missingness tensor
#
#     # load the tensor completion result
#     # data = np.load(f"../results_JCGS/application/completion_result/{day}_3_tt.npz")
#     completion_rank = du.loc[(du.day == int(day)) & (du.method == "tensor-train"), "rank"].values[0]
#     data = np.load(f"../results_JCGS/application/completion_result/{day}_{completion_rank}_tt.npz")
#     Xhat = data["Xhat"]
#
#     # test ncs
#     test_ncs = np.abs(Xhat - vista)[W == -1]
#
#     # record the result
#     cov_prob, width = {}, {}
#     output = []
#
#     for model, neighbor in [("Bernoulli", "1-nb"), ("Ising", "1-nb"), ("Ising", "1-nb-spatial"), ("unweighted", "N/A")]:
#         if model == "unweighted":
#             bandwidth = list(np.quantile(ncs, q=target_coverage))
#             cover = [np.mean(test_ncs <= h) for h in bandwidth]
#             cov_prob["unweighted"] = cover
#             width["unweighted"] = bandwidth
#             model_name = "unweighted"
#         else:
#             best_rank = df.loc[(df.day == int(day)) & (df.model == model) & (df.neighbor == neighbor), "rank"].values[0]
#
#             # load the fitted best-AIC model
#             model_abbr = "bern" if model == "Bernoulli" else "ising"
#             nb_abbr = "1nb" if neighbor == "1-nb" else "1nb_spat"
#             data = np.load(
#                 f"../results_JCGS/application/decomposition_result/{day}_{best_rank}_{model_abbr}_{nb_abbr}.npz")
#             B, cal = data["B"], data["cal"]
#
#             # compute non-conformity score
#             ncs = np.abs(Xhat - vista)[cal == 1]
#             ncs = np.concatenate([ncs, np.array([np.inf])])
#
#             # compute the weight
#             g_func = "product" if model == "Ising" else "zero"
#             tsing = TensorIsing(g_func=g_func, h_func="logit", g_params={"temp": 15, "scale": 1 / 5}, h_params=None,
#                                 nb=neighbor)
#             P = tsing.CondProb(W, B)
#             P = np.clip(P, 1e-4, 1 - 1e-4)
#             w_cal = ((1 - P) / P)[cal == 1]
#             w_test = np.amax(((1 - P) / P)[W == -1])
#             w = np.concatenate([w_cal, np.array([w_test])])
#             w = w / w.sum()
#
#             # compute the weighted conformal prediction result
#             cover, bandwidth = [], []
#             for q in target_coverage:
#                 qs = wq.quantile_1D(data=ncs, weights=w, quantile=q)
#                 cover.append(np.mean(test_ncs <= qs))
#                 bandwidth.append(qs)
#
#             if model == "Bernoulli":
#                 model_name = "Bernoulli"
#             elif neighbor == "1-nb" and model == "Ising":
#                 model_name = "Ising-st"
#             elif neighbor == "1-nb-spatial" and model == "Ising":
#                 model_name = "Ising-sp"
#
#             cov_prob[model_name] = cover
#             width[model_name] = bandwidth
#
#         # format the output
#         output.append([int(day), model_name] + cover + bandwidth)
#
#     # save the result
#     res = pd.DataFrame(data=output,
#                        columns=["day", "model"] + [f"X{qlevel}_pct" for qlevel in range(80, 100)] + [f"X{qlevel}_width"
#                                                                                                      for qlevel in
#                                                                                                      range(80, 100)])
#     res.to_csv(f"../results_JCGS/application/{day}_coverage_result.csv", index=False)


def ConformalTEC(day=1):
    """
    conformal inference on TEC data
    """

    final_res = []

    # load the TEC data
    day = str(day)
    day = day if len(day) == 2 else "0" + day
    data = loadmat(f"../data_JCGS/application/1709{day}.mat")
    mad, vista, dtec = data["mad"], data["vista"], data["dtec"]
    mad, vista, dtec = (
        np.flip(mad, axis=0),
        np.flip(vista, axis=0),
        np.flip(dtec, axis=0),
    )  # flip for easier visualization
    W = 2 * np.logical_not(np.isnan(mad)).astype(int) - 1

    # train-calibration split
    np.random.seed(int(day))
    mask = (np.random.uniform(size=W.shape) <= 0.7).astype(int)
    train, cal = np.logical_and(W == 1, mask == 1).astype(int), np.logical_and(
        W == 1, mask == 0
    ).astype(int)
    test = 1 - (train + cal)
    W_train = np.copy(W)
    W_train[train == 0] = -1

    # tensor completion
    X_train = np.copy(mad)
    X_train[W == -1] = np.nan
    X_train[cal == 1] = np.nan
    comp_model = TC_RGrad(true_X=None)
    comp_model.fit(
        X_train,
        (3, 3),
        method="tensor-train",
        config={"max_iter": 500, "thres": 1e-6, "beta": 0.1, "verbose": False},
    )
    print("Tensor completion finished.")

    # ----- Unweighted Conformal Prediction ----- #
    Xhat = comp_model.B
    ncs_cal = np.abs(Xhat - vista)[cal == 1]
    test_residual = np.abs(Xhat - vista)[test == 1]
    qs = np.quantile(
        np.concatenate([ncs_cal, np.array([np.inf])]), q=np.arange(0.80, 1, 0.01)
    )
    cov_prob = []
    for q in qs:
        cov_prob.append(np.mean(test_residual <= q))

    final_res.append(
        [int(day), "unweighted"]
        + cov_prob
        + [
            np.abs(np.arange(0.80, 1, 0.01) - np.array(cov_prob)).mean(),
            np.nan,
            np.nan,
            np.nan,
        ]
    )

    # ----- Binary Tensor Decomposition ----- #
    # Ising & Bernoulli model
    for prob_model in ["Ising", "Bernoulli"]:
        g_func = "product" if prob_model == "Ising" else "zero"
        K = 2 if prob_model == "Ising" else 3
        decomp_method = "tensor-train" if prob_model == "Ising" else "tucker"

        for r in range(2, 16):
            print(f"fitting {prob_model} with r={r}...")
            # binary tensor decomposition step
            decomp_model = LRIsing(g_func=g_func, h_func="logit", nb="1-nb")
            decomp_model.fit(
                W_train,
                rank=(r,) * K,
                q=0.7,
                method=decomp_method,
                config={
                    "max_iter": 2000,
                    "thres": 1e-3,
                    "beta": 0.1,
                    "max_iter_gamma": 500,
                    "thres_gamma": 1e-3,
                    "thres_temp": 1e-4,
                    "init_gamma": -np.log(0.8),  # set a priori based on external data
                    "fit_temperature": False,
                    "verbose": False,
                    "print_frequency": 100,
                },
            )

            # conformal inference step
            P = decomp_model.TenIsing.CondProb(
                W, decomp_model.B, invtemp=np.exp(decomp_model.gamma_hist[-1])
            )
            P = np.clip(P, 1e-4, 1 - 1e-4)
            p_cal = P[cal == 1]
            w_cal = (1 - p_cal) / p_cal
            p_test = P[test == 1]
            w_test = (1 - p_test) / p_test
            CI_tensor = EntryCI(ncs_cal, w_cal, w_test, np.arange(0.80, 1, 0.01))
            cov_prob = []
            for i, q in enumerate(np.arange(0.80, 1, 0.01)):
                cov_prob.append(np.mean(test_residual <= CI_tensor[i, :]))

            final_res.append(
                [int(day), prob_model]
                + cov_prob
                + [
                    np.abs(np.arange(0.80, 1, 0.01) - np.array(cov_prob)).mean(),
                    r,
                    decomp_model.AIC,
                    decomp_model.BIC,
                ]
            )

    print(f"binary tensor decomposition finished.")

    # format the output
    df = pd.DataFrame(
        data=final_res,
        columns=["day", "model"]
        + [f"{q}pct" for q in range(80, 100)]
        + ["miscoverage", "rank", "AIC", "BIC"],
    )
    df.to_csv(f"../results_JCGS/application/{day}_result.csv", index=False)


if __name__ == "__main__":
    # fit low-rank Ising model to the tensor sample
    # tasks = list(itertools.product([i for i in range(1,11)],
    #                                [r for r in range(2,91)],
    #                                ["tensor-train", "tucker"]))
    taskid = int(sys.argv[1])
    # c = tasks[taskid]
    # BinaryTEC(day=c[0], rank=c[1], model=c[2], neighbor=c[3], q=0.7)
    # df = pd.read_csv("../data_JCGS/application/completion_rank_table.csv")
    # CompleteTEC(day=taskid, rank=3, method="tensor-train")
    ConformalTEC(taskid)
