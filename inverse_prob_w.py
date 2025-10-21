from sklearn.linear_model import LogisticRegression
from statsmodels.api import WLS, add_constant
import pandas as pd 
import numpy as np 

df = pd.read_excel("./data/synth5k.xlsx")


# Define confounders and treatment
X = df[["X1", "X2", "X3"]]
d = df["D"]
y = df["Y"]

# Step 1: Estimate propensity scores
ps_model = LogisticRegression().fit(X, d)
ps = ps_model.predict_proba(X)[:, 1]

# Step 2: Compute IPW weights
weights = d / ps + (1 - d) / (1 - ps)

# Step 3: Include treatment in regression
X_ipw = pd.concat([X, d], axis=1)
X_ipw = add_constant(X_ipw)

# Step 4: Weighted regression
model = WLS(y, X_ipw, weights=weights).fit()

# Step 5: ATE is the coefficient on D
print(model.summary())
print("\nEstimated ATE (IPW):", model.params["D"])