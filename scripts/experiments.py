import os 
import pandas as pd
import numpy as np
import lmfit 
from model import surface_ocean_temp
import matplotlib.pyplot as plt

## load in data (Move to get_data_func)
def load_data(data_path):
    ## TODO: will be extend to allow for getting climate model data on the fly
    data = np.loadtxt(data_path, delimiter=None, dtype=str) 
    return data

def calc_anomaly(data, num_years):
    years = np.array([])
    anom = np.array([])
    for row in range(num_years):
        years = np.append(years, float(data[row][0]))
        anom = np.append(anom, float(data[row][1]))

    # convert to temperature anomaly from 1850
    anom = anom - anom[0]
    return anom


def fit_model(data, model_used, t):
    # Now make a model that fits function for upper ocean temperature to the temperature anomaly data
    # in order to find parameters for function that give the best fit using least squares approach
    
    mod = lmfit.Model(model_used)
    params = mod.make_params(A=-1, B=-1, alpha=1) ## parameters needed for best guess
    fitted_model = mod.fit(data, params, t=t, method='least_squares')

    return fitted_model, params 


def plot_model(years, model,  label, ax=None, fig=None):
    if not ax:
        fig, ax = plt.subplots(1)
    plt.plot(years, model, label=label)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Temperature Anomaly (K)', fontsize=12)
    plt.legend()
    return fig, ax


def main():
    ## GLOBALS
    data_dir = '../data'
    filename = 'hadCRUT_data.txt'
    data_path = os.path.join(data_dir, filename)
    # array for time, in years and seconds
    t = np.array(range(0,171), dtype='int64')
    years = t + 1850
    t_fut = np.array(range(0,271), dtype='int64')
    years_fut = t_fut + 1850

    ERF_data = pd.read_csv(os.path.join(data_dir, 'ERF_ssp585_1750-2500.csv'))
    ERF_data = ERF_data.set_index('year')
    ERF = ERF_data.loc[1850:2020]['total']

    data_used = load_data(data_path)
    temp_anom = calc_anomaly(data_used, num_years=171)
    
    fitted_model, params = fit_model(temp_anom, surface_ocean_temp, t=t)

    ## SET CONSTRAINTS
    # # need to constrain the parameters as A + B = -F/alpha
    # params['F'].expr = '-alpha*(A+B)'
    # Radiative forcing must be positive
    # params['F'].min = 0
    # Alpha is 1.04+-0.36 from CMIP6 (in slides)
    params['alpha'].min = 0.68
    params['alpha'].max = 1.40
    # best fit parameters from model
    A = fitted_model.params['A'].value
    B = fitted_model.params['B'].value
    # F = fitted_model.params['F'].value
    ERF_fut = np.array(ERF_data.loc[1850:2120]['total'] * 1.25)
    
    alpha = fitted_model.params['alpha'].value
    projection = surface_ocean_temp(t=t_fut, A=A, B=B, F=ERF_fut, alpha=alpha)
    print(projection)
    ## plot and save ouputs
    fig, ax = plot_model(years_fut, projection, label='model')
    fig, ax = plot_model(years, temp_anom, label='HadCRUT data', fig=fig, ax=ax)
    fig.savefig('../outputs/test2.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()