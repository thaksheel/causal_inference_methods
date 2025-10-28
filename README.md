# causal_inference_methods
Comparing different causal inference methods such as propensity scores, inverse probability, DoubleML, and text as a treatment

How to use: 
- Use the ate_evaluations.py which has the class to evaluate Propensity score matching, inverse probability, and DoubleML methods compared to mean difference and a known effect from synthetic data where you can control the number of confounders and the range of coefficients they can get
- See paper for similar set up: https://arxiv.org/abs/2502.19898
- However, the above paper made a mistake in the DoubleML python library and could not obtain an estimate for the known effect in their experiement. This mistake was corrected in the ate_evaluation.py file 

Issues: 
- IPW is performing much better than expected when increasing number of confounders and the coefficient range. However, DML is not performing well with high coefficients range and high number of confounders. This is challenging its formulation since DML was created for high dimensional data

Log: 
- 10/28/2025: DML and Propensity Score matching has been udpated and improved

Upcoming changes: 
- Add GPI package function by Imai for Causal Representation learning 
