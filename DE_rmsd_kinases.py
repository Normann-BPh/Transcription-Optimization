import json
import numpy as np
import pandas as pd
import sys
import pickle
import functions
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

with open("../dicts/Target_TFs_kinases.json", "r") as f:
    Target_TFs = json.load(f)
mRNA_df = pd.read_pickle("../pickle/mRNA_collectri_kinases_df.pkl")
with open("../dicts/TF_Targets_kinases.json", "r") as f:
    TF_Targets = json.load(f)
Prot_time_df = pd.read_pickle("../pickle/Prot_time_collectri_kinases_df.pkl")
with open("../dicts/non_optimizables_kinases.json", "r") as f:
    non_opti = json.load(f)

with open('../dicts/SQP/SQP_kinases_rmsd_alpha_dict.json','r') as f:
    alpha_pos_dict = json.load(f)
with open('../dicts/SQP/SQP_kinases_rmsd_beta_dict.json','r') as f:
    beta_pos_dict = json.load(f)
with open('../dicts/SQP/SQP_kinases_rmsd_error_dict.json','r') as f:
    error_pos_dict = json.load(f)

with open('../dicts/GA/GA_kinases_alpha_dict.json','r') as f:
    alpha_pos_dict_G = json.load(f)
with open('../dicts/GA/GA_kinases_beta_dict.json','r') as f:
    beta_pos_dict_G = json.load(f)

d1 = datetime.today()
print(d1.strftime('%d-%m-%Y'))

## bounds and variable count
last_fm = list(error_pos_dict.items())[-1][-1][-1]
last_fp = list(error_pos_dict.items())[-1][-1][0]
n_var = list(beta_pos_dict.items())[-1][-1][-1]
boundsl = np.zeros(n_var)
boundsl[list(beta_pos_dict.items())[0][-1][0]:] = -2
boundsu = np.ones(n_var)
boundsu[:last_fm] = 20


class MyProblem(ElementwiseProblem):
	
    def __init__(self, **kwargs):
        self.alpha_dict = alpha_pos_dict
        self.beta_dict = beta_pos_dict
        self.error_dict = error_pos_dict
        self.mRNA = mRNA_df
        self.Prot = Prot_time_df
        self.TTFs = Target_TFs
        
        
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
            begin_alpha, end_alpha = self.alpha_dict[R]
            alpha = x[begin_alpha:end_alpha]
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
                    P = np.sum([(x[(begin_beta+(list(self.Prot[T][1:]).index(site)))]*np.array(site)) for site in self.Prot[T][1:] if np.ndim(site) > 0],axis=0)
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
        out["H"] = np.concatenate((self.DE_sin_con_RNA(x),self.DE_con_alpha(x),self.DE_con_beta(x)),axis=0) ##functions.SQP_fun_sin_new_new(x,error_pos_dict,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs)
    
class MyCallback(Callback):
    
    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.opt = []
    
    def notify(self, algorithm):
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt.append(algorithm.opt[0].F)

## thread init
n_threads = 8 ###### ATTENTION ###### 8 threads
pool = ThreadPool(n_threads)
runner = StarmapParallelization(pool.starmap)
problem = MyProblem(elementwise_runner=runner)

## bias yes/no

## to determine best initial guess from prev runs
def best_mean(x_):
    last_mean = [10, None, None]
    for x in x_:
        F = functions.GA_Objectivs_sin_new(x,alpha_pos_dict_G,beta_pos_dict_G,mRNA_df,Prot_time_df,Target_TFs)
        maen = np.mean(np.abs(F))
        if maen < last_mean[0]:
            last_mean = [maen, x, F]
    return last_mean

bias = False
if bias == True:
    print('bias is on')
    link = './results/res_GA_kinases_X_multi_single_8_2bound_pop200_SBX2_PM2_FRS_ftol_T37551.npy'
    BIAS = np.load(link)
    maen = best_mean(BIAS)
    print(n_var,n_var-last_fm,maen[1].shape)
    error_part = np.zeros(last_fm)
    print(error_part.shape)
    error_part[:last_fp] =  np.array([entry if entry > 0 else 0 for entry in maen[2]])
    error_part[last_fp:last_fm] = np.array([entry if entry < 0 else 0 for entry in maen[2]]) ##error_part[:last_fp] * -2
    maen_dict = dict(sorted([(str(np.mean(functions.GA_Objectivs_sin_new(BIAS[i],alpha_pos_dict_G,beta_pos_dict_G,mRNA_df,Prot_time_df,Target_TFs))),np.insert(BIAS[i],0,error_part)) for i in range(200)]))
    X = np.array([maen_dict[key] for key in maen_dict])
    initial_guess = np.concatenate((error_part,maen[1]),axis=0)
    print(X.shape)
    print('init_g: ',initial_guess.shape)
    pop_init = Population.new('X', X)
    Evaluator().eval(problem, pop_init)
    time = link[link.find('T'):link.find('.npy')]
    name_ = '_biased_w_' + str(time) + '_'
else:
    print('bias is off')
    pop_init = LatinHypercubeSampling()#LHS
    name_ = '_'

## parameter for DE and termination
pop_size = 200 ## instead of int(n_var/2)
gen = 2
CR = 0.2
FF = 0.8
## sampling with pop_init form T37551 (pop_size = 200), F = 0.8 and CR = 0.2 results in error_sum ~ 27 and CV ~ 35 after 200 gen 
## + dither --> no effect; + DE/rand/1/bin --> error_sum ~ 29 and CV ~ 46; + DE/best/1/bin --> no effect
## sampling with pop_init form T37551 (pop_size = 200), F = 0.8 and CR = 0.2 results in error_sum ~ 28 and CV ~ 29 after 300 gen 

#ftol_termination = RobustTermination(SingleObjectiveSpaceTermination(tol=0.005, n_skip = 20), period=2) ### maybe add a max_gen ?! ###
#time_termination = get_termination("time", "30:00:00")


algorithm = DE(##pop_size = pop_size,
                ##n_offsgrings = None, ### from 200 to none for next try ###
                sampling = pop_init,
                ##variant = "DE/best/1/bin", ###### ATTENTION ###### no variant
                CR = CR,
                F = FF,
                ##dither = 'vector',
                callback = MyCallback()
                )

results = minimize(problem,
                    algorithm,
                    termination = get_termination('n_gen',100000),#ftol_termination,#
                    seed = 1,
                    save_history = False, ###### ATTENTION ###### history is off
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

name = name_ + 'rmsd_robust_{n}_100K_pop{p}CR{CR}_F{FF}_FRS_{date}_T{T}'.format(p=pop_size,n=n_threads,CR=CR,FF=FF,T=int(Time),date=d1.strftime('%d-%m-%Y'))
np.save("./results/res_DE_kinases{name}.npy".format(name=name),X,allow_pickle=True)
with open("./results/cal_DE_kinases{name}.pkl".format(name=name),'wb') as f:
    pickle.dump(results.algorithm.callback,f)
