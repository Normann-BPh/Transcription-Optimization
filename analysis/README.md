The scripts in this directory were used to analyse the parameters and their predictions.

# `constraint_violartions.py`
This script checks each parameter-set for violations of the following constraints:
- $0 \le \alpha \le 1$ with each $\alpha$-set under the constraint: $\sum_i \alpha_i = 1$
- $-2 \le \beta \le 2$ with each $\beta$-set under the constraint: $\sum_j \beta_j = 1$


Required are the dictionary with the postion of the parameters, the dictionary of the network and any number of parameter-sets. 

The script does NOT automatically SAVE the results.

# `error_calculation.py`
This script calculates the error (MSA and RMSD) of the predictions of each parameter-set.

Required are the dictionary with the postion of the parameters, the dictionary of the network and any number of parameter-sets. 
Uses functions for the `functions.py` script.

The script does NOT automatically SAVE the results.

# `physicality.py`
This script classifies the effect and the impact of the parameters of each set and compares the effect (inhibtion, activation or nothing)
to the CollecTRI classification.

The method uses the 'Sub_Sum', which is the term in the model equation associated with a single TF, to categorize. The 'Sub_Sum' is 
defined as follows
$S_{TF} = \alpha_{i,j} \cdot TF_{i,j} \cdot (\beta^0_j + \sum_q^k \beta_j^q \cdot [PSite]_j^k).$

The latest (Colloqium) method decided by the following statements
$E' = \begin{cases}
	\text{I}, \quad \text{if } \bar{S}_{TF} < 0, \\
	\text{A}, \quad \text{if } 0 < \bar{S}_{TF}, \\
	\text{0}, \quad \text{if } \bar{S}_{TF} = 0,
	\end{cases}$
- Inhibition
- Activation
- n0ne

$E'' = \begin{cases}
	\text{r}, \quad \text{if } \bar{S}_{TF} < \bar{TF}, \\
	\text{e}, \quad \text{if } \bar{TF} < \bar{S}_{TF}, \\
	\text{n}, \quad \text{if } \bar{S}_{TF} = \bar{TF}.
	\end{cases}$
- repressed
- enhanced
- normal

The old method is also included in the script, but commented out.

The script DOES automatically SAVE the results.
