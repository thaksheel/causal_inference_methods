# causal_inference_methods
Comparing different causal inference methods such as propensity scores, inverse probability, DoubleML, and text as a treatment

How to use: 
- Use the ate_evaluations.py which has the class to evaluate Propensity score matching, inverse probability, and DoubleML methods compared to mean difference and a known effect from synthetic data where you can control the number of confounders and the range of coefficients they can get
- See paper for similar set up: https://arxiv.org/abs/2502.19898
- However, the above paper made a mistake in the DoubleML python library and could not obtain an estimate for the known effect in their experiement. This mistake was corrected in the ate_evaluation.py file 
