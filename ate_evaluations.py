import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
import doubleml as dml
from statsmodels.api import WLS, add_constant
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from psmpy import PsmPy


@dataclass
class CausalSetUp:
    index: np.ndarray = None
    confounders: List[np.ndarray] = None
    outcome: np.ndarray = None
    treatment: np.ndarray = None


class ATEEvaluations:
    def __init__(
        self,
        seed,
        outcome_type: Literal["binary", "continuous"],
        data: Optional[CausalSetUp] = None,
    ):
        self.outcome_type = outcome_type
        self.data = data if data is not None else None
        self.seed = seed
        self.synth_data: CausalSetUp = None 
        self.known_confounders_coef = None 
        self.known_effect = None
        self.num_confounders: int = (
            None if self.data is None else len(data.confounders.T)
        )

    def data_to_df(self, data: Optional[CausalSetUp] = None):
        if data is None or isinstance(data, CausalSetUp):
            data = asdict(self.synth_data)
        for i, arr in enumerate(data["confounders"].T):
            f = f"X{i+1}"
            data[f] = arr
        data.pop("confounders")
        return pd.DataFrame(data)

    def data_to_causalsetup(self, data: pd.DataFrame):
        cols = ["index", "treatment", "outcome"]
        ncols = len(data.columns) - len(cols)
        confounders = data[[f"X{i+1}" for i in range(ncols)]].to_numpy()
        return CausalSetUp(
            index=data["index"],
            confounders=confounders,
            outcome=data["outcome"],
            treatment=data["treatment"],
        )

    def dml(
        self,
        data: CausalSetUp,
        model1=None,
        model2=None,
        model3=None,
    ):
        learner = RandomForestRegressor(
            n_estimators=100,
            max_features=self.num_confounders,
            max_depth=5,
            min_samples_leaf=2,
        )
        model1 = learner if model1 is None else model1
        model2 = learner if model2 is None else model2
        model3 = learner if model3 is None else model3
        dml_data = dml.DoubleMLData.from_arrays(
            data.confounders, data.outcome, data.treatment
        )
        dml_model = dml.DoubleMLPLR(dml_data, model1, model2, n_folds=5, n_rep=5)
        # dml_model = dml.DoubleMLPLR(dml_data, model1, model2, model3, n_folds=3, score="IV-type", n_rep=5)
        dml_model.fit(n_jobs_cv=10)
        return dml_model.coef[0], dml_model.se[0], dml_model.pval[0]

    def inverse_prob(self, data: Optional[CausalSetUp] = None):
        if data is None or not isinstance(data, CausalSetUp):
            data = self.synth_data
        elif data is not None and isinstance(data, pd.DataFrame):
            data = self.data_to_causalsetup()
        ps_model = LogisticRegression().fit(data.confounders, data.treatment)
        ps = ps_model.predict_proba(data.confounders)[:, 1]

        # Compute IPW weights
        weights = data.treatment / ps + (1 - data.treatment) / (1 - ps)

        # Include treatment in regression
        X_ipw = np.column_stack((data.confounders, data.treatment))
        X_ipw = add_constant(X_ipw)

        # Weighted regression
        model = WLS(data.outcome, X_ipw, weights=weights).fit()
        return model.params[-1], model.bse[-1], model.pvalues[-1]

    def propensity_score2(self, data: CausalSetUp):
        df = self.data_to_df(data)
        X = df.drop(columns=["outcome", "treatment", "index"])
        y = df["treatment"]
        model = LogisticRegression(solver="lbfgs", max_iter=1000)
        model.fit(X, y)
        df["propensity_score"] = model.predict_proba(X)[:, 1]
        treated = df[df["treatment"] == 1].copy()
        control = df[df["treatment"] == 0].copy()
        nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        nn.fit(control[["propensity_score"]])
        distances, indices = nn.kneighbors(treated[["propensity_score"]])
        treated["matched_ID"] = control.iloc[indices.flatten()]["index"].values
        treated["outcome_untreated"] = control.iloc[indices.flatten()]["outcome"].values
        diff = treated["outcome"] - treated["outcome_untreated"]
        ate, std = diff.mean(), diff.std()
        return ate

    def propensity_score(self, data: CausalSetUp):
        df = self.data_to_df(data)
        psm = PsmPy(df, treatment="treatment", indx="index", exclude=["outcome"])
        # TODO: it might be best to create my own psm library since this one has a lot of issues
        try:
            psm.logistic_ps(balance=True)
        except ValueError:
            psm.logistic_ps(balance=False)
        psm.knn_matched(matcher="propensity_score", replacement=True)
        matched = psm.df_matched.merge(df[["outcome", "index"]], on="index")
        if self.outcome_type == "binary":
            pair_diffs = matched[matched["treatment"] == 1]
            try:
                pair_diffs["outcome_untreated"] = [
                    matched[matched["index"] == row["matched_ID"]]["outcome"].iloc[0]
                    for _, row in pair_diffs.iterrows()
                ]
            except IndexError:
                pair_diffs["outcome_untreated"] = [
                    matched[matched["matched_ID"] == row["index"]]["outcome"].iloc[0]
                    for _, row in pair_diffs.iterrows()
                ]
            diff = pair_diffs["outcome"] - pair_diffs["outcome_untreated"]
            ate, std = diff.mean(), diff.std()
            return ate
        elif self.outcome_type == "continuous":
            pass

    def mean_diff(self, data: Optional[CausalSetUp] = None):
        if data is None:
            data = self.synth_data
        if self.outcome_type == "binary":
            return (
                data.outcome[data.treatment == 1].mean()
                - data.outcome[data.treatment == 0].mean()
            )
        elif self.outcome_type == "continuous":
            model = LinearRegression()
            model.fit(data.treatment.reshape(-1, 1), data.outcome)
            return model.coef_[0]

    def create_synth(
        self,
        n_samples: int,
        n_confounders: int,
        known_effect: float,
        coe_range: int = 10,
    ):
        np.random.seed(self.seed)
        self.known_effect = known_effect
        causal_setup = CausalSetUp()
        causal_setup.index = np.array([i for i in range(n_samples)])
        causal_setup.confounders = np.random.randn(n_samples, n_confounders)
        if self.outcome_type == "binary":
            causal_setup.treatment = ((np.random.randn(n_samples)) > 0).astype(int)
        elif self.outcome_type == "continuous":
            causal_setup.treatment = np.random.randn(n_samples)
        self.known_confounders_coef = [
            np.random.randint(low=1, high=coe_range) for _ in range(n_confounders)
        ]
        coe_confounders = np.sum(
            np.array(
                [
                    (self.known_confounders_coef[i] * x).T
                    for i, x in enumerate(causal_setup.confounders.T)
                ]
            ),
            axis=0,
        )
        causal_setup.outcome = (
            known_effect * causal_setup.treatment
            + coe_confounders
            + np.random.randn(n_samples)
        )
        self.synth_data = causal_setup
        self.num_confounders = len(causal_setup.confounders.T)
        return self

    def evaluate(self, data: Optional[CausalSetUp] = None):
        # all the methods to find ATE based on outcome_type
        if data is None:
            data = self.synth_data
        results_ate = {
            "known-effect": self.known_effect,
            "mean-diff": self.mean_diff(data),
            "dml": self.dml(data)[0],
            "psw": self.propensity_score2(data),
            "ipw": self.inverse_prob(data)[0],
        }
        return results_ate

    def evaluates(
        self,
        confounders: List[int],
        n_samples: int,
        known_effect: float,
        coe_range: int,
    ):
        # do n_confounders evaluate on all methods
        return [
            self.evaluate(
                self.create_synth(
                    n_samples=n_samples,
                    n_confounders=n_confounders,
                    known_effect=known_effect,
                    coe_range=coe_range,
                ).synth_data
            )
            for n_confounders in confounders
        ]

    def affichage(
        self, results: List[Dict], confounders: List[int], ke, sample, coerange
    ):
        rs = dict(
            zip(
                [k for k in results[0].keys()],
                [[] for _ in results[0].keys()],
            )
        )
        for result in results:
            for k, v in result.items():
                rs[k].append(v)
        results = rs
        plt.figure()
        plt.plot(
            confounders,
            results["known-effect"],
            marker="*",
            linestyle=":",
            color="yellow",
            label="known-effect",
        )
        plt.plot(
            confounders,
            results["mean-diff"],
            marker="o",
            linestyle="--",
            color="red",
            label="mean-diff",
        )
        plt.plot(
            confounders,
            results["dml"],
            marker="o",
            linestyle="-",
            color="steelblue",
            label="dml",
        )
        plt.plot(
            confounders,
            results["psw"],
            marker="o",
            linestyle="-",
            color="darkorange",
            label="psm",
        )
        plt.plot(
            confounders,
            results["ipw"],
            marker="o",
            linestyle="-",
            color="green",
            label="ipw",
        )
        plt.title(
            f"Comparison of causal inference methods for confounders [{confounders[0]}, {confounders[-1]}] \n known_effect={ke} | samples={sample} | coef_range={coerange}"
        )
        plt.xlabel("confounders count")
        plt.ylabel("ATE")
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"./exports/eval_coerange{coerange}_ke{ke}_s{sample}.png")

    def export(self):
        pass


if __name__ == "__main__":
    ate_eval = ATEEvaluations(seed=47, outcome_type="binary", data=None)
    coeranges = [2, 4, 8, 16, 32, 64]
    coeranges = [2, 4, 8, 16, 32, 64, 128, 256]
    confounders = [i for i in range(3, 20)]
    if False:
        results = []
        mean_diffs = []
        # for coerange in coeranges:
        for confounder in confounders:
            synth_data = ate_eval.create_synth(
                n_samples=1000, n_confounders=confounder, known_effect=4, coe_range=16
            ).synth_data
            mean_diffs.append(ate_eval.mean_diff(synth_data))
            # results.append(ate_eval.propensity_score(synth_data))
            results.append(ate_eval.dml(synth_data)[0])
        print([float(round(i, 3)) for i in mean_diffs])
        print([float(round(i, 3)) for i in results])
    # testing evaluates()
    if True:
        confounders = [i for i in range(3, 100)]
        coeranges = [64, 128, 256]
        coeranges = [2, 4, 8, 16, 32, 64, 128, 256]
        for coerange in coeranges:
            results = ate_eval.evaluates(
                confounders=confounders,
                n_samples=1000,
                known_effect=4,
                coe_range=coerange,
            )
            ate_eval.affichage(
                results, confounders, coerange=coerange, ke=4, sample=1000
            )

    print("END")
