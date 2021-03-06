import numpy as np
import pandas as pd
import os
import lmfit

global KRAK_VALS, KRAKATOA_YEAR
KRAK_VALS = {}
KRAKATOA_YEAR = 1883

# set constants
data_dir = 'data'
ERF_data = pd.read_csv(os.path.join(data_dir, 'SSPs/','ERF_ssp585_1750-2500.csv'))
ERF_data = ERF_data.set_index('year')
ERF = np.array(ERF_data.loc[1850:2020]['total'])

    
start_point = 1961-1850
end_point = 1990 -1850

rho = 1000 # density of water kgm-3
c_p = 4218 # specific heat of water Jkg-1K-1
kap = 1e-4 # vertical diffusivity m2s-1

h_u = 100 # upper ocean height m
h_d = 900 # deep ocean height m

gamma = (2*kap*c_p*rho)/(h_u+h_d) # prop constant for heat transfer to deep ocean Wm-2K-1

C_u = rho*c_p*h_u # specific heat of upper ocean Jm-2K-1
C_d = rho*c_p*h_d # specific heat of deep ocean Jm-2K-1

dt = 365*24*60*60 # seconds in year


# Solved second order differential equation to find expression for T_u:
# T_u = Aexp(lambda1*t) + Bexp(lambda2*t) + F/alpha
# where lambda1,2 are found using quadratic formula from homogenous 2nd order ODE solution, and
# A and B are constants, where A + B = -F/alpha (from inhomogenous solution)

def upper_ocean_temp(t, alpha, F=None, krakatwoa=False):
    if type(F) != np.array and type(F) != np.ndarray:
        F = ERF
    T_u = np.zeros(t)
    T_d = np.zeros(t)
    for i in range(t-1):
        if krakatwoa:
            if i == 200:
                F[i] += (KRAK_VALS[KRAKATOA_YEAR] * 2)
            if i == 201:
                F[i] += (KRAK_VALS[KRAKATOA_YEAR+1] * 2)
            if i == 202:
                F[i] += (KRAK_VALS[KRAKATOA_YEAR+2] * 2)
            if i == 203:
                F[i] += (KRAK_VALS[KRAKATOA_YEAR+3] * 2)
        T_u[i+1] = (1/C_u)*(F[i] - (alpha+gamma)*T_u[i] + T_d[i]*gamma)*dt + T_u[i]
        T_d[i+1] = (gamma/C_d)*(T_u[i]-T_d[i])*dt + T_d[i]
    T_u = T_u - np.mean(T_u[start_point:end_point])
    return T_u


def get_opt_model(temp_anom, F, t=171):
    alpha_val, opt_error = opt_alpha(temp_anom=temp_anom, F=F, t=t)
    return alpha_val, opt_error
    

def opt_alpha(temp_anom, F, t=171):
    mod = lmfit.Model(upper_ocean_temp, F=F)
    params = mod.make_params(alpha=1)
    fit_result = mod.fit(temp_anom, params, t=t)
    return fit_result.params['alpha'].value, fit_result.params['alpha'].stderr
