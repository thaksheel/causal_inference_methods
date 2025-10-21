import pandas as pd
import numpy as np
from psmpy import PsmPy


data = pd.read_excel("./data/synth5k.xlsx")
data["index"] = data.index
Y = data[["index", "Y"]] 

# Step 1: Initialize PSM
psm = PsmPy(data, treatment="D", indx="index", exclude=["Y"])

# Step 2: Estimate propensity scores using logistic regression
psm.logistic_ps(balance=True)

# Step 3: Perform matching (e.g., k-NN on logit scores)
psm.knn_matched(matcher="propensity_score", replacement=False)

# Step 4: Retrieve matched dataset
matched = psm.df_matched.merge(Y, on="index")

# Step 5: Estimate ATE
treated = matched[matched["D"] == 1]["Y"]
control = matched[matched["D"] == 0]["Y"]
ate = treated.mean() - control.mean()

print("Estimated ATE (PSM):", ate)

# FIXME: I do not think this is good enough since it mainly removes the unmatched from the dataset 