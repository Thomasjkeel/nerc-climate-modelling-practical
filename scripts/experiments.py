import os 
import numpy as np
from lmfit import Model
from model import surface_ocean_temp
import matplotlib.pyplot as plt

data_dir = data
filename = 'hadCRUT_data.txt'

data_path = os.path.join(data, filename)

# array for time, in years and seconds
t = np.array(range(0,171), dtype='int64')

## load in data (Move to get_data_func)
HadCRUT_data = np.loadtxt(data_path, delimiter=None, dtype=str) 

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


# quick plot of the temperature anomaly
years = t + 1850

plt.plot(years,temp_anom)
plt.xlabel('Year')
plt.xlim(1850,2025)
plt.ylabel('Temperature Anomaly (K)')
plt.show()

result = mod.fit(temp_anom, params, t=t, method='least_squares')

result.plot()
plt.show()

# Now make prediction for the next 100 years...

t_fut = np.array(range(0,271), dtype='int64')
years_fut = t_fut + 1850

# best fit parameters from model

A = result.params['A'].value
B = result.params['B'].value
F = result.params['F'].value
alpha = result.params['alpha'].value


T_fut = T(t=t_fut, A=A, B=B, F=F, alpha=alpha)

plt.plot(years_fut, T_fut, label='model')
plt.plot(years, temp_anom, label='HadCRUT data')

plt.xlabel('Year', fontsize=12)
plt.ylabel('Temperature Anomaly (K)', fontsize=12)

plt.legend()

plt.show()