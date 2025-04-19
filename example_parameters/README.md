All results were named in a similar theme. Starting with "res", followed by the abbriviation of the optimizer
- "GA" for non-dominated sorting Genetic Algorithm,
- "SQP" for Sequential least sQuares Programming,
- "DE" for Differential Evolution.
Next the network version is added, dictating what source/method was used to pick the mRNA (Targets) of the network. "HitG" indicates
the mRNAs where chosen based on their associated speed, then follows the amount of mRNAs in the network. Afterwards some relevant 
specifications of the algorithms are noted. The last two numbers indicate the start data and the duration of optimization.

The results from DE and GA optimization are saved as numpy files. The SQP results as a csv.