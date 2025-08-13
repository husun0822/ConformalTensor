import sys, time
import pandas as pd
from LRIsing import *
from SimIsing import *
from scipy.io import loadmat
from Conformal import ConfTC, TC_RGrad


def SimulateTrainCalSplit(d=40, rank=3, g_func="product", q=0.7):
    data = np.load(f"../data_JCGS/simulation/missingness/{g_func}_{d}_{rank}.npz")
    B, sample = data["B"], data["sample"]
    train_mask = np.zeros_like(sample)

    for i in range(sample.shape[0]):
        W = sample[i]
        np.random.seed(i)
        mask = np.random.uniform(size=W.shape) <= q
        train_mask[i, :, :, :] = np.logical_and(mask == 1, W == 1)

    np.savez(
        f"../data_JCGS/simulation/missingness/{g_func}_{d}_{rank}.npz",
        B=B,
        sample=sample,
        train=train_mask,
    )


def SimulateMissingness(
    tensor_size, tensor_rank, method="tensor-train", g_func="zero", h_func="logit"
):
    # create simulator object
    sim = SimIsing(
        d=tensor_size,
        g_func=g_func,
        h_func=h_func,
        g_params={"temp": 15, "scale": 1 / 2},
        h_params=None,
        nb="1-nb",
    )

    # simulate fix B
    B = sim.sim_B(rank=tensor_rank, method=method, seed=2024, tensor_type="checkerbox")

    # simulate random sample with block gibbs
    sample = sim.sim_sample(
        method="Block-Gibbs",
        method_params={"max_iter": 40000, "burn_in": 10000, "N_sample": 30},
        seed=2024,
    )

    # save the true parameter and sample
    d, rank = tensor_size[0], tensor_rank[0]
    np.savez(
        f"../data_JCGS/simulation/missingness/{g_func}_{d}_{rank}.npz",
        B=B,
        sample=sample,
    )


def ModelSelection(
    d=40, rank=3, g_func="product", misspecified=False, method="tensor-train", q=0.7
):
    # load the data
    data = np.load(f"../data_JCGS/simulation/missingness/{g_func}_{d}_{rank}.npz")
    B, sample = data["B"], data["sample"]
    K = 2 if method == "tensor-train" else 3

    # fit RGrad to each sample
    res = []

    for i in range(sample.shape[0]):
        # print(f"fitting sample {i}...")
        W = sample[i]
        np.random.seed(i)
        mask = np.random.uniform(size=W.shape) <= q
        train, cal = np.logical_and(mask == 1, W == 1), np.logical_and(
            mask == 0, W == 1
        )
        W_train = np.copy(W)
        W_train[train == False] = -1  # only keep the training entries

        for r in range(2, 16):
            # fit RGrad
            if not misspecified:
                model = LRIsing(
                    g_func=g_func,
                    h_func="logit",
                    g_params=None,
                    h_params=None,
                    nb="1-nb",
                    true_B=B,
                    true_temp=15,
                )
            else:
                g_func_fit = "product" if g_func == "zero" else "zero"
                model = LRIsing(
                    g_func=g_func_fit,
                    h_func="logit",
                    g_params=None,
                    h_params=None,
                    nb="1-nb",
                    true_B=B,
                    true_temp=15,
                )
            model.fit(
                W_train,
                rank=(r,) * K,
                q=q,
                method=method,
                config={
                    "max_iter": 2000,
                    "thres": 1e-3,
                    "beta": 0.1,
                    "max_iter_gamma": 3000,
                    "thres_gamma": 1e-4,
                    "thres_temp": 1e-4,
                    "init_gamma": -np.log(model.true_temp),
                    "fit_temperature": False,
                    "verbose": True,
                    "print_frequency": 100,
                },
            )
            RMSE_B = np.sqrt(np.mean((model.B - B) ** 2))
            res.append(
                [
                    g_func,
                    d,
                    rank,
                    i,
                    r,
                    model.RSE_hist[-1],
                    RMSE_B,
                    model.l,
                    model.BIC,
                    model.AIC,
                ]
            )

    df = pd.DataFrame(
        data=res,
        columns=[
            "g_func",
            "tensor_size",
            "tensor_rank",
            "sample",
            "fit_rank",
            "RSE",
            "RMSE",
            "iteration",
            "BIC",
            "AIC",
        ],
    )
    if not misspecified:
        if method == "tucker":
            df.to_csv(
                f"../results_JCGS/simulation/compare_method/RGrad_Tucker_{g_func}_{d}.csv",
                index=False,
            )
        else:
            df.to_csv(
                f"../results_JCGS/simulation/{g_func}_{d}_{rank}.csv", index=False
            )
    else:
        if method == "tucker":
            df.to_csv(
                f"../results_JCGS/simulation/compare_method/misspecified_RGrad_Tucker_{g_func}_{d}.csv",
                index=False,
            )
        else:
            df.to_csv(
                f"../results_JCGS/simulation/misspecified_{g_func}_{d}_{rank}.csv",
                index=False,
            )


def ConformalInference(
    d=40,
    rank=3,
    g_func="product",
    noise="constant",
    normalized_ncs=False,
    two_sided=False,
):
    rank_df = pd.read_csv("../data_JCGS/simulation/RGrad_rank.csv", index_col=False)
    g_model = "Ising" if g_func == "product" else "Bernoulli"

    data = np.load(f"../data_JCGS/simulation/missingness/{g_func}_{d}_{rank}.npz")
    B, sample, train = data["B"], data["sample"], data["train"]
    simdata = SimData(d=B.shape)
    X_true, X_noise = simdata.sim_X(
        rank=(3, 3),
        seed=2024,
        noise=noise,
        sigma=1.0,
        true_B=B,
        SNR=2.0,
        tensor_type="checkerbox",
    )
    X_true = X_true + X_noise  # noisy tensor observation

    output = []
    for i in range(sample.shape[0]):
        # mask the tensor
        X = np.copy(X_true)
        W = sample[i]
        X[W == -1] = np.nan
        working_rank = rank_df.loc[
            (rank_df.model == g_model)
            & (rank_df.d == d)
            & (rank_df.iter == i)
            & (rank_df.misspecified == False)
            & (rank_df.r == rank),
            "AIC_rank",
        ].values[0]

        res = []
        for weight in ["unweighted", "oracle", "RGrad"]:
            CTC = ConfTC(
                g_func=g_func,
                h_func="logit",
                g_params={"temp": 15, "scale": 1 / 2},
                nb="1-nb",
                true_B=B,
                q=0.7,
            )
            cov_prob, width, inf_prob = CTC.fit(
                X,
                train[i],
                tc_params={"true_X": X_true, "rank": (3, 3, 3), "method": "tucker"},
                dc_params={"rank": (working_rank,) * 2, "method": "tensor-train"},
                weight=weight,
                target_q=np.linspace(0.80, 0.99, 20),
                invtemp=1 / 15,
                seed=0,
                normalized_ncs=normalized_ncs,
                two_sided=two_sided,
            )

            res.append(
                [g_func, d, rank, weight, noise, normalized_ncs, two_sided, i]
                + list(cov_prob)
                + list(width)
                + list(inf_prob)
            )
        output += res
    output = pd.DataFrame(
        data=output,
        columns=[
            "model",
            "d",
            "r",
            "weight",
            "noise",
            "normalized_ncs",
            "two_sided",
            "iter",
        ]
        + [f"{q}pct" for q in range(80, 100)]
        + [f"{q}pct_width" for q in range(80, 100)]
        + [f"{q}pct_infprob" for q in range(80, 100)],
    )

    ncs = "_normalized" if normalized_ncs else ""
    twoside = "_twosided" if two_sided else ""
    output.to_csv(
        f"../results_JCGS/simulation/conformal/{g_func}_{noise}_{d}_{rank}{ncs}{twoside}.csv",
        index=False,
    )


def ModelPrefit(d=40, g_func="product"):
    rank_df = pd.read_csv(
        "../data_JCGS/simulation/all_method_rank_table.csv", index_col=False
    )
    g_model = "Ising" if g_func == "product" else "Bernoulli"
    method = "Tucker-Bernoulli"

    # load the training data
    data = np.load(f"../data_JCGS/simulation/missingness/{g_func}_{d}_3.npz")
    B, sample, train = data["B"], data["sample"], data["train"]

    fitB = np.zeros_like(sample, dtype="float")
    for i in range(sample.shape[0]):
        # prepare binary tensor
        W = sample[i]
        W_train = np.copy(W)
        train_mask = train[i]
        W_train[train_mask == 0] = -1

        # find the best-fitting rank
        working_rank = rank_df.loc[
            (rank_df.method == method)
            & (rank_df["sample"] == (i + 1))
            & (rank_df.model == g_model)
            & (rank_df.d == d),
            "rank",
        ].values[0]

        # fit the model
        decomp_model = LRIsing(
            g_func="zero", h_func="logit", nb="1-nb", true_B=B, true_temp=15
        )
        decomp_model.fit(
            W_train,
            rank=(working_rank,) * 3,
            q=0.7,
            method="tucker",
            config={
                "max_iter": 2000,
                "thres": 1e-3,
                "beta": 0.1,
                "max_iter_gamma": 3000,
                "thres_gamma": 1e-4,
                "thres_temp": 1e-4,
                "init_gamma": -np.log(decomp_model.true_temp),
                "fit_temperature": False,
                "verbose": False,
                "print_frequency": 1000,
            },
        )

        # store the output
        fitB[i] = decomp_model.B

    # save the fitted model
    np.save(f"../data_JCGS/simulation/missingness/Tucker_{g_func}_{d}.npy", fitB)


def ConformalCompete(d=40, rank=3, g_func="product", noise="constant"):
    g_model = "Ising" if g_func == "product" else "Bernoulli"

    data = np.load(f"../data_JCGS/simulation/missingness/{g_func}_{d}_{rank}.npz")
    fitTucker = np.load(f"../data_JCGS/simulation/missingness/Tucker_{g_func}_{d}.npy")
    fitGCP = loadmat(f"../data_JCGS/simulation/missingness/GCP_{g_func}_{d}.mat")
    B, sample, train, fitGCP = data["B"], data["sample"], data["train"], fitGCP["fitB"]
    simdata = SimData(d=B.shape)
    X_true, X_noise = simdata.sim_X(
        rank=(3, 3),
        seed=2024,
        noise=noise,
        sigma=1.0,
        true_B=B,
        SNR=2.0,
        tensor_type="checkerbox",
    )
    X_true = X_true + X_noise  # noisy tensor observation

    output = []
    for i in range(sample.shape[0]):
        # mask the tensor
        X = np.copy(X_true)
        W = sample[i]
        X[W == -1] = np.nan

        res = []
        for method in ["GCP", "Tucker"]:
            if method == "GCP":
                fitted_B = fitGCP[i]
            elif method == "Tucker":
                fitted_B = fitTucker[i]
            CTC = ConfTC(g_func=g_func, h_func="logit", nb="1-nb", true_B=B, q=0.7)
            cov_prob, width, inf_prob = CTC.fit(
                X,
                train[i],
                tc_params={"true_X": X_true, "rank": (3, 3, 3), "method": "tucker"},
                dc_params={
                    "rank": (3, 3),
                    "method": "tensor-train",
                    "fitted_B": fitted_B,
                },
                weight="External-Bernoulli",
                target_q=np.linspace(0.80, 0.99, 20),
                invtemp=1 / 15,
                seed=0,
            )

            res.append(
                [g_func, d, rank, method, noise, i]
                + list(cov_prob)
                + list(width)
                + list(inf_prob)
            )
        output += res
    output = pd.DataFrame(
        data=output,
        columns=["model", "d", "r", "weight", "noise", "iter"]
        + [f"{q}pct" for q in range(80, 100)]
        + [f"{q}pct_width" for q in range(80, 100)]
        + [f"{q}pct_infprob" for q in range(80, 100)],
    )
    output.to_csv(
        f"../results_JCGS/simulation/conformal/compete_{g_func}_{noise}_{d}_{rank}.csv",
        index=False,
    )


def TimingExperiment(d=40, rank=3, g_func="zero", q=0.7):
    # load the data
    data = np.load(
        f"../data_JCGS/simulation/missingness/pre_revision_ver/{g_func}_{d}_{rank}.npz"
    )
    B, sample = data["B"], data["sample"]
    K = 2

    # fit RGrad to each sample
    res = []

    for i in range(sample.shape[0]):
        # train-calibration split
        W = sample[i]
        np.random.seed(i)
        mask = np.random.uniform(size=W.shape) <= q
        train, cal = np.logical_and(mask == 1, W == 1), np.logical_and(
            mask == 0, W == 1
        )
        W_train = np.copy(W)
        W_train[train == False] = -1  # only keep the training entries

        # specify a model
        model = LRIsing(
            g_func=g_func,
            h_func="logit",
            g_params={"temp": 15, "scale": 1 / 2},
            h_params=None,
            nb="1-nb",
            true_B=B,
            true_temp=15,
        )

        # I. fit RGrad with fixed stepsize
        start = time.time()
        model.fit(
            W_train,
            rank=(rank,) * K,
            q=q,
            method="tensor-train",
            config={
                "seed": i,
                "max_iter": 2000,
                "thres": 1e-4,
                "beta": 0.1,
                "backtracking": False,
                "armijo_threshold": 1e-4,
                "projected_grad": False,
            },
        )
        end = time.time()
        res.append(
            [
                g_func,
                d,
                rank,
                i,
                "RGrad",
                model.RSE_hist[-1],
                np.sqrt(np.mean((model.B - B) ** 2)),
                model.l,
                end - start,
                model.BIC,
                model.AIC,
            ]
        )

        # II. fit RGrad with linearized line search
        start = time.time()
        model.fit(
            W_train,
            rank=(rank,) * K,
            q=q,
            method="tensor-train",
            config={
                "seed": i,
                "max_iter": 2000,
                "thres": 1e-4,
                "beta": 10,
                "backtracking": True,
                "armijo_threshold": 1e-4,
                "projected_grad": False,
            },
        )
        end = time.time()
        res.append(
            [
                g_func,
                d,
                rank,
                i,
                "RGrad_adaptive",
                model.RSE_hist[-1],
                np.sqrt(np.mean((model.B - B) ** 2)),
                model.l,
                end - start,
                model.BIC,
                model.AIC,
            ]
        )

        # III. fit projected gradient descent
        start = time.time()
        model.fit(
            W_train,
            rank=(rank,) * K,
            q=q,
            method="tensor-train",
            config={
                "seed": i,
                "max_iter": 2000,
                "thres": 1e-4,
                "beta": 0.1,
                "backtracking": False,
                "armijo_threshold": 1e-4,
                "projected_grad": True,
            },
        )
        end = time.time()
        res.append(
            [
                g_func,
                d,
                rank,
                i + 1,
                "projected_GD",
                model.RSE_hist[-1],
                np.sqrt(np.mean((model.B - B) ** 2)),
                model.l,
                end - start,
                model.BIC,
                model.AIC,
            ]
        )

    res = pd.DataFrame(
        res,
        columns=[
            "g",
            "d",
            "r",
            "i",
            "optim",
            "RSE",
            "RMSE",
            "total_iter",
            "runtime",
            "BIC",
            "AIC",
        ],
    )
    res.to_csv(f"../results_JCGS/simulation/time_and_conv/{g_func}_{d}_{rank}.csv")


def misspecified_CTC(d=40, rank=3, g_func="product", noise="constant"):
    # get the RGrad rank from AIC/BIC selection result
    rank_df = pd.read_csv("../data_JCGS/simulation/RGrad_rank.csv", index_col=False)
    g_model = "Ising" if g_func == "product" else "Bernoulli"

    # load the simulated missingness data
    data = np.load(f"../data_JCGS/simulation/missingness/{g_func}_{d}_{rank}.npz")
    B, sample, train = data["B"], data["sample"], data["train"]

    # simulate a simple data tensor
    simdata = SimData(d=B.shape)
    X_true, X_noise = simdata.sim_X(
        rank=(5, 5, 5),
        method="tucker",
        seed=2024,
        noise=noise,
        sigma=1.0,
        true_B=B,
        SNR=2.0,
        tensor_type="checkerbox",
    )

    # iterate over all tensor missingness masks
    output = []
    for i in range(sample.shape[0]):
        # mask the tensor
        X = np.copy(X_true)
        W = sample[i]
        X[W == -1] = np.nan
        working_rank = rank_df.loc[
            (rank_df.model == g_model)
            & (rank_df.d == d)
            & (rank_df.iter == i)
            & (rank_df.misspecified == False)
            & (rank_df.r == rank),
            "AIC_rank",
        ].values[0]

        res = []
        for tucker_rank in list(range(1, 11)):
            CTC = ConfTC(
                g_func=g_func,
                h_func="logit",
                g_params={"temp": 15, "scale": 1 / 2},
                nb="1-nb",
                true_B=B,
                q=0.7,
            )

            cov_prob, width, inf_prob = CTC.fit(
                X,
                train[i],
                tc_params={
                    "true_X": X_true,
                    "rank": (tucker_rank,) * 3,
                    "method": "tucker",
                },
                # tensor completion parameter
                dc_params={"rank": (working_rank,) * 2, "method": "tensor-train"},
                # binary mask decomposition parameter
                weight="RGrad",
                target_q=np.linspace(0.80, 0.99, 20),
                invtemp=1 / 15,
                seed=0,
            )

            res.append(
                [g_func, d, rank, "RGrad", noise, i, tucker_rank]
                + list(cov_prob)
                + list(width)
                + list(inf_prob)
            )

        output += res

    # format output dataframe
    output = pd.DataFrame(
        data=output,
        columns=[
            "model",
            "d",
            "r",
            "weight",
            "noise",
            "iter",
            "tensor_completion_rank",
        ]
        + [f"{q}pct" for q in range(80, 100)]
        + [f"{q}pct_width" for q in range(80, 100)]
        + [f"{q}pct_infprob" for q in range(80, 100)],
    )
    output.to_csv(
        f"../results_JCGS/simulation/conformal/misspecified_tensor_completion_experiment/{g_func}_{noise}_{d}_{rank}_misspecified_tensor_completion.csv",
        index=False,
    )


if __name__ == "__main__":
    # simulate missingness tensor
    # tasks = list(itertools.product([40], [3], ["zero","product"]))
    # tasks = list(itertools.product([40,60,80,100], [3,5,7,9], ["zero","product"]))
    # tasks = list(itertools.product([40, 60], ["zero", "product"], [True, False]))
    # tasks = list(itertools.product([40, 60], ["zero", "product"]))
    tasks = list(
        itertools.product(
            [40, 60, 80, 100],
            [3, 5, 7, 9],
            ["zero", "product"],
            ["constant", "adversarial"],
        )
    )

    # Filter out tasks where both boolean values are False
    # tasks = [task for task in raw_tasks if not (task[-2] == False and task[-1] == False)]

    idx = int(sys.argv[1])
    c = tasks[idx]

    # npz2mat(f"../data_JCGS/simulation/missingness/pre_revision_ver/{c[2]}_{c[0]}_{c[1]}.npz")
    # TimingExperiment(d=c[0], rank=c[1], g_func=c[2], q=0.7)
    # SimulateTrainCalSplit(d=c[0], rank=c[1], g_func=c[2])
    # SimulateMissingness(tensor_size=(c[0],)*3, tensor_rank=(c[1],)*2, g_func=c[2])
    # ModelSelection(d=c[0], rank=c[1], g_func=c[2], misspecified=c[3], method="tensor-train")
    # ModelSelection(d=c[0], rank=3, g_func=c[1], misspecified=c[2], method="tucker", q=0.7)
    # ConformalInference(d=c[0], rank=c[1], g_func=c[2], noise=c[3], normalized_ncs=c[4], two_sided=c[5])
    # ModelPrefit(d=c[0], g_func=c[1])
    # ConformalCompete(d=c[0], rank=c[1], g_func=c[2], noise=c[3])
    misspecified_CTC(d=c[0], rank=c[1], g_func=c[2], noise=c[3])
