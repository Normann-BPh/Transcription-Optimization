import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import minimize, Bounds
from scipy.optimize import NonlinearConstraint

with open("../dicts/Target_TFs_HitG.json", "r") as f:
    Target_TFs = json.load(f)

mRNA_df = pd.read_pickle("../pickle/mRNA_collectri_HitG_50_df.pkl")
Prot_time_df = pd.read_pickle("../pickle/Prot_time_collectri_HitG_df.pkl")

with open('../dicts/SQP/SQP_HitG_50_rmsd_alpha_dict.json','r') as f:
    alpha_pos_dict = json.load(f)
with open('../dicts/SQP/SQP_HitG_50_rmsd_beta_dict.json','r') as f:
    beta_pos_dict = json.load(f)
with open('../dicts/SQP/SQP_HitG_50_rmsd_error_dict.json','r') as f:
    error_pos_dict = json.load(f)

last_f = list(error_pos_dict.items())[-1][-1][-1]

n_var = list(beta_pos_dict.items())[-1][-1][-1] # last beta entry equals number of parameters

def error_sum(x):
    return np.sum(x[0:last_f])
def error_sum_grad(x):
    grad = np.zeros(n_var)
    grad[0:last_f] = 1
    return grad

boundsl = np.zeros(n_var) # lower bound for alpha and errors
boundsl[list(beta_pos_dict.items())[0][-1][0]:] = -2 # lower bound for beta 
boundsu = np.ones(n_var) # upper bound for alpha
boundsu[0:last_f] = 20 # upper bound for errors
boundsu[list(beta_pos_dict.items())[0][-1][0]:] = 2 # upper bound for beta 
bounds = Bounds(boundsl,boundsu)

the_prot_dict = dict(zip(list(Prot_time_df.columns),[np.array([np.array([site[time] for site in Prot_time_df[T][1:] if np.ndim(site) > 0]) for time in range(9)]) for T in Prot_time_df.columns]))

def SQP_fun_sin_new_new(x,error_pos_dict,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs):
    fun_res = np.array([])
    for R in mRNA_df.columns:
        begin_alpha, end_alpha = alpha_pos_dict[R]
        alpha = x[begin_alpha:end_alpha]
        # ---
        begin_fp,begin_fm = error_pos_dict[R]
        fp = x[begin_fp]
        fm = x[begin_fm]
        # ---
        unit_ = np.arange(len(Target_TFs[R]*9)).reshape(len(Target_TFs[R]),9)
        unit = np.zeros_like(unit_,dtype=float)
        for T in Target_TFs[R]:
            begin_beta, end_beta = beta_pos_dict[T]
            dex = Target_TFs[R].index(T)
            if end_beta - begin_beta > 1:
                P = np.sum(x[begin_beta+1:end_beta]*the_prot_dict[T],axis=1)
                unit[dex] = alpha[dex] * np.array(Prot_time_df[T][0]) * (x[begin_beta] + P)
            else:
                unit[dex] = alpha[dex] * np.array(Prot_time_df[T][0]) * x[begin_beta]
        unit = np.sqrt((np.sum((np.array(mRNA_df[R])-np.sum(unit,axis=0))**2))/9) - fp + fm
        fun_res = np.append(fun_res,unit)
    # ---
    for a in alpha_pos_dict:
        unit = 1 - sum(x[alpha_pos_dict[a][0]:alpha_pos_dict[a][1]])
        fun_res = np.append(fun_res,unit)
    # ---
    for a in beta_pos_dict:
        unit = 1 - sum(x[beta_pos_dict[a][0]:beta_pos_dict[a][1]])
        fun_res = np.append(fun_res,unit)
    return fun_res


callback_ndarray = np.array([])
steps = [0]
def callback_fun(x): # to keep trach of iterations and function value
    global callback_ndarrays
    global steps
    print(steps[0],error_sum(x))
    steps[0] +=1

# define the equality constraints. a jacobian can be added as well using the key 'jac': ...
eq_cons = {'type': 'eq',
            'fun': lambda x: SQP_fun_sin_new_new(x,error_pos_dict,alpha_pos_dict,beta_pos_dict,mRNA_df,Prot_time_df,Target_TFs),
			}

def run_opti(x0,maxiter):
    '''
    defined for simplicity in calling the algorithm.
    initializes and performs the actual optimization
    '''
    res = minimize(error_sum, x0, method='SLSQP', jac=error_sum_grad, 
               constraints=[eq_cons], options={'ftol':0.005, 'disp':True, 'maxiter':maxiter},
               bounds=bounds,callback=callback_fun)
    return res

x0 = np.random.randint(boundsl*10,boundsu*10,size=n_var)/100 # make initial guess inside the bounds

maxiter = 100

restarts = 0
best_sum = 100 # set to have a comparable value for following while-loop. needs to be higher than expected best result

d1 = datetime.today()
print(d1.strftime('%d-%m-%Y')) # get starting time of algorithm

while steps[0] < 1000:
    res = run_opti(x0,maxiter)
    exit_mode = res.message
    
    if res.success:
        best_res = res
        n_restarts = restarts
        break
    else:
        x0 = x0+np.random.uniform(-0.5,0.5) # perturbes the last initial value by randomly substracting or adding 0.5. 
        restarts += 1
    if error_sum(res.x) < best_sum: # should the algorithm find no successful solution at all the best so far is used instead
        best_res = res
        n_restarts = restarts
        best_sum = error_sum(res.x)

print(best_res.x)
print(n_restarts)
res = best_res

d2 = datetime.today()
print(d2.strftime('%d-%m-%Y'))
dif = d2 - d1
Time = dif.seconds + (dif.days * 3600 * 24)

d = {'x':res.x, 'x0':x0}
res_df = pd.DataFrame(data=d)

name = 'SQP_HitG_50_rmsd_{}_restarts{}_steps{}_nit{}_{}_{}_b'.format(maxiter,n_restarts,steps[0],best_res.nit,d1.strftime('%d-%m-%Y'),Time)