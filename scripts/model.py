import numpy as np
from lmfit import Model

# set constants

rho = 100 # density of water kgm-3
c_p = 4218 # specific heat of water Jkg-1K-1
kap = 1e-4 # vertical diffusivity m2s-1

h_u = 100 # upper ocean height m
h_d = 900 # deep ocean height m

gamma = (2*kap*c_p*rho)/(h_u+h_d) # prop constant for heat transfer to deep ocean Wm-2K-1

C_u = rho*c_p*h_u # specific heat of upper ocean Jm-2K-1
C_d = rho*c_p*h_d # specific heat of deep ocean Jm-2K-1


# Solved second order differential equation to find expression for T_u:
# T_u = Aexp(lambda1*t) + Bexp(lambda2*t) + F/alpha
# where lambda1,2 are found using quadratic formula from homogenous 2nd order ODE solution, and
# A and B are constants, where A + B = -F/alpha (from inhomogenous solution)

def surface_ocean_temp(t, A, B, F, alpha):
    # function for upper ocean temperature, T_u, with variable inputs for constants A, B, F and alpha
    
    # terms for quadratic equation to solve second order ODE solution for T_u
    a = C_u/gamma
    b = (alpha+gamma)/gamma + C_u/C_d
    c = (alpha+gamma)/C_d - gamma/C_d
    
    # exponential coefficients for T_u solution using quadratic formula
    lambda1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    lambda2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    
    t_s = t*365*24*60*60 # convert time to seconds
    
    T_u = A * np.exp(lambda1*t_s) + B * np.exp(lambda2*t_s) + F/alpha
    
    return T_u


# array for time, in years and seconds

t = np.array(range(0,171), dtype='int64')


# Now make a model that fits function for upper ocean temperature to the temperature anomaly data
# in order to find parameters for function that give the best fit using least squares approach

mod = Model(surface_ocean_temp)
params = mod.make_params(A=-1,B=-1,F=2,alpha=1)

# need to constrain the parameters as A + B = -F/alpha
params['F'].expr = '-alpha*(A+B)'

# Radiative forcing must be positive
params['F'].min = 0
# Alpha is 1.04+-0.36 from CMIP6 (in slides)
params['alpha'].min = 0.68
params['alpha'].max = 1.40

result = mod.fit(temp_anom, params, t=t, method='least_squares')

