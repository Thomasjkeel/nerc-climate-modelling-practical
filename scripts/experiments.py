import os 
import numpy as np
from lmfit import Model
from model import surface_ocean_temp

data_path = os.path.join('data','hadCRUT_data.txt')

# array for time, in years and seconds
t = np.array(range(0,171), dtype='int64')

## get data
HadCRUT_data = np.loadtxt(filename, delimiter=None, dtype=str) 

years = np.array([])
temp_anom = np.array([])
for row in range(171):
    years = np.append(years, float(HadCRUT_data[row][0]))
    temp_anom = np.append(temp_anom, float(HadCRUT_data[row][1]))

# convert to temperature anomaly from 1850
temp_anom = temp_anom - temp_anom[0]


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
