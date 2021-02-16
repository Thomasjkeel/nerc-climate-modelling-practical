import numpy as np
import pandas as pd
import os
# set constants
#  data_dir = '../data'
#  ERF_data = pd.read_csv(os.path.join(data_dir, 'ERF_ssp585_1750-2500.csv'))
#  ERF_data = ERF_data.set_index('year')
#  ERF = ERF_data.loc[1850:2020]['total']

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

def upper_ocean_temp(alpha, F):
    # if type(F) != np.array and type(F) != np.ndarray:
    #     F = ERF
    T_u = np.zeros(len(F))
    T_d = np.zeros(len(F))

    for i in range(len(F)-1):
        T_u[i+1] = (1/C_u)*(F[i] - (alpha+gamma)*T_u[i] + T_d[i])*dt + T_u[i]
        T_d[i+1] = (gamma/C_d)*(T_u[i]-T_d[i])*dt + T_d[i]
    
    return T_u
