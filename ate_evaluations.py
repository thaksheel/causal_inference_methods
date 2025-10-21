import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass, asdict
from typing import Dict, List, Literal, Optional
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
        data: Optional[pd.DataFrame] = None,
    ):
        self.outcome_type = outcome_type
        if data is not None:
            self.data = data
        self.seed = seed
        self.synth_data: CausalSetUp = None
        self.known_confounders_coef = None
        self.known_effect = None

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
        model1=RandomForestRegressor(),
        model2=RandomForestRegressor(),
    ):
        dml_data = dml.DoubleMLData.from_arrays(
            data.confounders, data.outcome, data.treatment
        )
        dml_model = dml.DoubleMLPLR(dml_data, model1, model2)
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

    def propensity_score(self, data: CausalSetUp):
        # FIXME: I do not think this is correct implementation
        df = self.data_to_df(data)
        psm = PsmPy(df, treatment="treatment", indx="index", exclude=["outcome"])
        try:
            psm.logistic_ps(balance=True)
        except ValueError:
            psm.logistic_ps(balance=False)
        psm.knn_matched(matcher="propensity_logit", replacement=False)
        matched = psm.df_matched.merge(df[["outcome", "index"]], on="index")
        if self.outcome_type == "binary":
            treated = matched[matched["treatment"] == 1]["outcome"]
            control = matched[matched["treatment"] == 0]["outcome"]
            ate = treated.mean() - control.mean()
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
        return self

    def evaluate(self, data: Optional[CausalSetUp] = None):
        # all the methods to find ATE based on outcome_type
        if data is None:
            data = self.synth_data
        results_ate = {
            "known-effect": self.known_effect,
            "mean-diff": self.mean_diff(data),
            "dml": self.dml(data)[0],
            "psw": self.propensity_score(data),
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
            label="psw",
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
    if False:
        synth_data = ate_eval.create_synth(
            n_samples=750, n_confounders=5, known_effect=1.5, coe_range=7
        ).synth_data
        diff = ate_eval.mean_diff(synth_data)
        df = ate_eval.data_to_df()
        results = ate_eval.evaluate()
    # testing evaluates()
    confounders = [i for i in range(3, 100)]
    for coerange in [2, 4, 8, 16, 32, 64, 128, 256]:
        results = ate_eval.evaluates(
            confounders=confounders, n_samples=1000, known_effect=4, coe_range=coerange
        )
        ate_eval.affichage(results, confounders, coerange=coerange, ke=4, sample=1000)

    print("END")
