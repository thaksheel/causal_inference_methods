import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import doubleml as dml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(42)
n_samples = 2500 
n_features = 3 

X = np.random.randn(n_samples, n_features)
D = (np.random.randn(n_samples)) > 0
D = D.astype(int)
Y = 3 * X[:, 0] + 2 * X[:, 1] + 5 * X[:, 2] + 1.5 * D + np.random.randn(n_samples)

data = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(n_features)])
data["Y"] = Y
data["D"] = D

X = data[["X1", "X2", "X3"]]  
y = data["Y"]  
d = data["D"]  
# data.to_excel("./data/synth5k.xlsx", index=False)

ml_model = RandomForestRegressor(n_estimators=100)
dml_data = dml.DoubleMLData.from_arrays(X.values, y.values, d.values)
dml_model = dml.DoubleMLPLR(dml_data, ml_model, ml_model)
dml_model.fit()

print("mean difference:", Y[D == 1].mean() - Y[D == 0].mean())
print("Model Summary:\n", dml_model.summary)


print("END")
