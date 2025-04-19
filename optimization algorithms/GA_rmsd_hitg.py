import sys
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
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator

with open("../dicts/Target_TFs_HitG.json", "r") as f:
    Target_TFs = json.load(f)

mRNA_df = pd.read_pickle("../pickle/mRNA_collectri_HitG_50_df.pkl")
Prot_time_df = pd.read_pickle("../pickle/Prot_time_collectri_HitG_50_df.pkl")

with open('../dicts/GA/GA_HitG_50_alpha_dict.json','r') as f:
    alpha_pos_dict = json.load(f)
with open('../dicts/GA/GA_HitG_50_beta_dict.json','r') as f:
    beta_pos_dict = json.load(f)

n_var = list(beta_pos_dict.items())[-1][-1][-1] # last beta entry equals number of parameters

boundsl = np.zeros(n_var) # lower bound for alpha
boundsl[list(beta_pos_dict.items())[0][1][0]:] = -2 # lower bound for beta 
boundsu = np.ones(n_var) # upper bound for alpha
boundsu[list(beta_pos_dict.items())[0][1][0]:] = 2 # upper bound for beta 

the_prot_dict = dict(zip(list(Prot_time_df.columns),[np.array([np.array([site[time] for site in Prot_time_df[T][1:] if np.ndim(site) > 0]) for time in range(9)]) for T in Prot_time_df.columns]))

class MyProblem(ElementwiseProblem):
	
    def __init__(self, **kwargs):
        self.alpha_dict = alpha_pos_dict
        self.beta_dict = beta_pos_dict
        self.mRNA = mRNA_df
        self.Prot = Prot_time_df
        self.TTFs = Target_TFs
        self.TPD = the_prot_dict
        
        super().__init__(n_var = n_var, 
                        n_obj = len(mRNA_df.columns), 
                        n_eq_constr = len(alpha_pos_dict)+len(beta_pos_dict), 
                        xl = boundsl, 
                        xu = boundsu)

    def _evaluate(self, x, out):
        out["F"] = self.GA_rmsd(x)
        out["H"] = np.concatenate((self.GA_con_alpha(x),self.GA_con_beta(x)),axis=0)

    def GA_con_alpha(self, x):
        H_res = np.array([])
        for a in self.alpha_dict:
            # for g ------------------------------------------------------
            unit = 1 - sum(x[self.alpha_dict[a][0]:self.alpha_dict[a][1]])
            H_res = np.append(H_res,unit)
        return H_res
    
    def GA_con_beta(self, x):
        H_res = np.array([])
        for b in self.beta_dict:
            # for g ------------------------------------------------------
            unit = 1 - sum(x[self.beta_dict[b][0]:self.beta_dict[b][1]])
            H_res = np.append(H_res,unit)
        return H_res
    
    def GA_rmsd(self, x):
        F_res = np.array([])
        for R in self.mRNA.columns:
            alpha_start,alpha_end = self.alpha_dict[R]
            alpha = x[alpha_start:alpha_end]
            unit_ = np.arange(len(self.TTFs[R])*9).reshape(len(self.TTFs[R]),9)
            unit = np.zeros_like(unit_,dtype=float)
            for T in self.TTFs[R]:
                beta_start,beta_end = self.beta_dict[T]
                dex = self.TTFs[R].index(T)
                if beta_end-beta_start > 1:
                    P = np.sum(x[beta_start+1:beta_end]*self.TPD[T],axis=1)
                    unit[dex] = alpha[dex]* np.array(self.Prot[T][0]) * (x[beta_start] + P)
                else:
                    unit[dex] = alpha[dex] * np.array(self.Prot[T][0]) * x[beta_start]
            unit = np.sqrt((np.sum((np.array(self.mRNA[R])-np.sum(unit,axis=0))**2))/9)
            F_res = np.append(F_res,unit)
        return F_res


n_threads = 16 
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)

problem = MyProblem(elementwise_runner=runner)

pop_init = FloatRandomSampling() #FRS

pop_size = 100
sbx = 0.2
pm = 0.4

algorithm = NSGA2(pop_size = pop_size,
                    n_offsprings = 200,
                    sampling = pop_init,
                    crossover = SBX(prob = 0.8, eta = sbx),
                    mutation = PM(eta = pm),
                    eliminate_duplicates = True
                    )

ftol_termination = RobustTermination(MultiObjectiveSpaceTermination(tol=0.005, n_skip = 1000), period=10)
#time_termination = get_termination("time", "00:00:30")

d1 = datetime.today()
print(d1.strftime('%d-%m-%Y'))

results = minimize(problem,
    algorithm,
    termination = get_termination('n_gen',50),#ftol_termination,#
    seed = 1,
    save_history = False,
    verbose = True,
    return_least_infeasible=True,
    )

Time = results.exec_time
print('Threads:', Time)
pool.close()

X = results.X
print(X.shape)
F = results.F
CV = results.CV

num_gen = results.algorithm.n_gen
n_off = algorithm.n_offsprings

name = 'rmsd_{}_pop{}_off{}_gen{}_SBX{}_PM{}_FRS_{}_T{}'.format(n_threads,pop_size,n_off,num_gen,sbx,pm,d1.strftime('%d-%m-%Y'),int(Time))