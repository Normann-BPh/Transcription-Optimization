import json
import numpy as np
import pandas as pd
import sys
import functions
import pickle
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
#from pyrecorder.writers.video import Video
#from pyrecorder.recorder import Recorder
from pymoo.core.callback import Callback
import matplotlib.pyplot as plt
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator

with open("../dicts/Target_TFs_kinases.json", "r") as f:
    Target_TFs = json.load(f)
mRNA_df = pd.read_pickle("../pickle/mRNA_collectri_kinases_df.pkl")
with open("../dicts/TF_Targets_kinases.json", "r") as f:
    TF_Targets = json.load(f)
Prot_time_df = pd.read_pickle("../pickle/Prot_time_collectri_kinases_df.pkl")
with open("../dicts/non_optimizables_kinases.json", "r") as f:
    non_opti = json.load(f)

with open('../dicts/GA/GA_kinases_alpha_dict.json','r') as f:
    alpha_pos_dict = json.load(f)
with open('../dicts/GA/GA_kinases_beta_dict.json','r') as f:
    beta_pos_dict = json.load(f)

d1 = datetime.today()
print(d1.strftime('%d-%m-%Y'))

n_var = list(beta_pos_dict.items())[-1][-1][-1]
boundsl = np.zeros(n_var)
boundsl[list(beta_pos_dict.items())[0][1][0]:] = -2
boundsu = np.ones(n_var)
boundsu[list(beta_pos_dict.items())[0][1][0]:] = 2


class MyProblem(ElementwiseProblem):
	
    def __init__(self, **kwargs):
        self.alpha_dict = alpha_pos_dict
        self.beta_dict = beta_pos_dict
        self.mRNA = mRNA_df
        self.Prot = Prot_time_df
        self.TTFs = Target_TFs
        
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
            begin_of_alpha,end_of_alpha = self.alpha_dict[R]
            alpha = x[begin_of_alpha:end_of_alpha]
            unit_ = np.arange(len(self.TTFs[R])*9).reshape(len(self.TTFs[R]),9)
            unit = np.zeros_like(unit_,dtype=float)
            for T in self.TTFs[R]:
                begin_of_beta,end_of_beta = self.beta_dict[T]
                dex = self.TTFs[R].index(T)
                if end_of_beta-begin_of_beta > 1:
                    P = np.sum([(x[(begin_of_beta+(list(self.Prot[T][1:]).index(site)))]*np.array(site)) for site in self.Prot[T][1:] if np.ndim(site) > 0],axis=0)
                    unit[dex] = alpha[dex]* np.array(self.Prot[T][0]) * (x[begin_of_beta] + P)
                else:
                    unit[dex] = alpha[dex] * np.array(self.Prot[T][0]) * x[begin_of_beta]
            unit = np.sqrt((np.sum((np.array(self.mRNA[R])-np.sum(unit,axis=0))**2))/9)
            F_res = np.append(F_res,unit)
        return F_res

class MyCallback(Callback):
    
    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.opt = []
    
    def notify(self, algorithm):
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt.append(algorithm.opt[0].F)

n_threads = 8 ###### ATTENTION ###### 8 threads
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)

problem = MyProblem(elementwise_runner=runner)

bias = 'no'
if bias == 'yes':
    print('bias is on')
    link = './results/res_GA_kinases_X_multi_single_8_2bound_pop200_SBX2_PM2_FRS_ftol_T37551.npy'
    X = np.load(link)
    pop_init = Population.new('X', X)
    Evaluator().eval(problem, biased_pop)
    time = link[link.find('T'):link.find('.npy')]
    name_ = '_biased_w_' + str(time) + '_'
else:
    pop_init = FloatRandomSampling() #FRS
    name_ = '_'

pop_size = 200
gen = None
sbx = 0.2
pm = 0.2

algorithm = NSGA2(pop_size = pop_size,
                    n_offsgrings = 200,
                    sampling = pop_init,
                    crossover = SBX(prob = 0.9, eta = sbx),
                    mutation = PM(eta = pm),
                    eliminate_duplicates = True,
                    callback = MyCallback()
                    )

ftol_termination = RobustTermination(MultiObjectiveSpaceTermination(tol=0.005, n_skip = 20), period=2) ### maybe add a max_gen ?! ###
#time_termination = get_termination("time", "30:00:00")

print(n_var)

results = minimize(problem,
    algorithm,
    termination = get_termination('n_gen',100000),#ftol_termination,#
    seed = 1,
    save_history = False, ###### ATTENTION ###### history is off
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


name = name_+'rmsd_{n}_pop{p}_SBX{SBX}_PM{PM}_FRS_{date}_T{T}'.format(T=int(Time),p=pop_size,n=n_threads,SBX=sbx,PM=pm,date=d1.strftime('%d-%m-%Y'))

np.save("./results/res_GA_kinases{name}.npy".format(name=name),X,allow_pickle=True)
with open("./results/cal_GA_kinases{name}.pkl".format(name=name),'wb') as f:
    pickle.dump(results.algorithm.callback,f)