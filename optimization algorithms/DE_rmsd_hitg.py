import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from pymoo.core.problem import ElementwiseProblem
from multiprocessing.pool import ThreadPool
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize
from pymoo.termination.ftol import SingleObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator

with open("../dicts/Target_TFs_HitG.json", "r") as f:
    Target_TFs = json.load(f)

mRNA_df = pd.read_pickle("../pickle/mRNA_collectri_HitG_50_df.pkl")
Prot_time_df = pd.read_pickle("../pickle/Prot_time_collectri_HitG_50_df.pkl")

with open('../dicts/SQP/SQP_HitG_50_rmsd_alpha_dict.json','r') as f:
    alpha_pos_dict = json.load(f)
with open('../dicts/SQP/SQP_HitG_50_rmsd_beta_dict.json','r') as f:
    beta_pos_dict = json.load(f)
with open('../dicts/SQP/SQP_HitG_50_rmsd_error_dict.json','r') as f:
    error_pos_dict = json.load(f)

with open('../dicts/GA/GA_HitG_50_alpha_dict.json','r') as f:
    alpha_pos_dict_G = json.load(f)
with open('../dicts/GA/GA_HitG_50_beta_dict.json','r') as f:
    beta_pos_dict_G = json.load(f)


last_fm = list(error_pos_dict.items())[-1][-1][-1]
last_fp = list(error_pos_dict.items())[-1][-1][0]
n_var = list(beta_pos_dict.items())[-1][-1][-1]
boundsl = np.zeros(n_var) # lower bound for alpha and errors
boundsl[list(beta_pos_dict.items())[0][-1][0]:] = -2 # lower bound for beta 
boundsu = np.ones(n_var) # upper bound for alpha
boundsu[0:last_f] = 20 # upper bound for errors
boundsu[list(beta_pos_dict.items())[0][-1][0]:] = 2 # upper bound for beta 

the_prot_dict = dict(zip(list(Prot_time_df.columns),[np.array([np.array([site[time] for site in Prot_time_df[T][1:] if np.ndim(site) > 0]) for time in range(9)]) for T in Prot_time_df.columns]))

class MyProblem(ElementwiseProblem):
	
    def __init__(self, **kwargs):
        self.alpha_dict = alpha_pos_dict
        self.beta_dict = beta_pos_dict
        self.error_dict = error_pos_dict
        self.mRNA = mRNA_df
        self.Prot = Prot_time_df
        self.TTFs = Target_TFs
        self.TPD = the_prot_dict
        
        super().__init__(n_var = n_var, 
                        n_obj = 1, 
                        n_eq_constr = len(mRNA_df.columns)+len(alpha_pos_dict)+len(beta_pos_dict), 
                        xl = boundsl, 
                        xu = boundsu)

    def DE_error_sum(self, x):
        return np.sum(x[:last_fm])
    
    def DE_sin_con_RNA(self,x):
        fun_res = np.array([])
        for R in self.mRNA.columns:
            begin_alpha, alpha_end = self.alpha_dict[R]
            alpha = x[begin_alpha:alpha_end]
            # ---
            begin_fp,begin_fm = self.error_dict[R]
            fp = x[begin_fp]
            fm = x[begin_fm]
            # ---
            unit_ = np.arange(len(self.TTFs[R]*9)).reshape(len(self.TTFs[R]),9)
            unit = np.zeros_like(unit_,dtype=float)
            for T in self.TTFs[R]:
                begin_beta, end_beta = self.beta_dict[T]
                dex = self.TTFs[R].index(T)
                if end_beta - begin_beta > 1:
                    P = np.sum(x[begin_beta+1:end_beta]*self.TPD[T],axis=1)
                    unit[dex] = alpha[dex] * np.array(self.Prot[T][0]) * (x[begin_beta] + P)
                else:
                    unit[dex] = alpha[dex] * np.array(self.Prot[T][0]) * x[begin_beta]
            unit = np.sqrt((np.sum((np.array(mRNA_df[R])-np.sum(unit,axis=0))**2))/9) - fp + fm
            fun_res = np.append(fun_res,unit)
        return fun_res

    def DE_con_alpha(self,x):
        fun_res = np.array([])
        # ---
        for a in self.alpha_dict:
            unit = 1 - sum(x[self.alpha_dict[a][0]:self.alpha_dict[a][1]])
            fun_res = np.append(fun_res,unit)
        return fun_res

    def DE_con_beta(self,x):
        fun_res = np.array([])
        # ---
        for b in self.beta_dict:
            unit = 1 - sum(x[self.beta_dict[b][0]:self.beta_dict[b][1]])
            fun_res = np.append(fun_res,unit)
        return fun_res
    
    def _evaluate(self, x, out):        
        out["F"] = self.DE_error_sum(x)
        out["H"] = np.concatenate((self.DE_sin_con_RNA(x),self.DE_con_alpha(x),self.DE_con_beta(x)),axis=0)


## thread init
n_threads = 16 ###### ATTENTION ###### 16 threads
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)
problem = MyProblem(elementwise_runner=runner)

pop_init = LatinHypercubeSampling()#LHS

## parameter for DE and termination
pop_size = 200
gen = 200
CR = 0.9
WF = 0.8
#ftol_termination = RobustTermination(SingleObjectiveSpaceTermination(tol=0.005, n_skip = 20), period=2) ### maybe add a max_gen ?! ###
#time_termination = get_termination("time", "30:00:00")


algorithm = DE(sampling = pop_init,
                variant = "DE/best/1/bin",
                CR = CR,
                F = WF
                )

d1 = datetime.today()
print(d1.strftime('%d-%m-%Y'))

results = minimize(problem,
                    algorithm,
                    termination = get_termination('n_gen',100000),#ftol_termination,#
                    seed = 1,
                    save_history = False,
                    verbose = True,
                    return_least_infeasible=True,
                    )

## end
Time = results.exec_time
print('Threads:', Time)
pool.close()

X = results.X
F = results.F
CV = results.CV

print(F, CV)

pop = algorithm.pop_size
n_gen = results.algorithm.n_gen

name = '_rmsd_best1bin_{}_{}_pop{}_CR{}_WF{}_FRS_{}_{}'.format(n_threads,n_gen,pop,CR,WF,d1.strftime('%d-%m-%Y'),int(Time))