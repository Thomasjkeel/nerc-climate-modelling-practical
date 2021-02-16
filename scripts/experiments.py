import os 
import pandas as pd
import numpy as np
import lmfit 
from model import surface_ocean_temp as upper_ocean_temp
import matplotlib.pyplot as plt

## GLOBALS
FORCING_SENSITIVITY = 1

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


def load_forcing_data(filename):
    ERF_data = pd.read_csv(filename)
    ERF_data = ERF_data.set_index('year')
    ERF = np.array(ERF_data.loc[1850:2020]['total']) * FORCING_SENSITIVITY
    ERF_fut = np.array(ERF_data.loc[1850:2120]['total'] * FORCING_SENSITIVITY)
    return ERF, ERF_fut

def main():
    # array for time, in years and seconds
    t = np.array(range(0,171), dtype='int64')
    years = t + 1850
    t_fut = np.array(range(0,271), dtype='int64')
    years_fut = t_fut + 1850

    ## file locations
    data_dir = '../data'
    filename = 'hadCRUT_data.txt'
    path_to_ssp_forcings = os.path.join(data_dir, 'SSPs/')

    ## load data and calc temperature anomaly
    data_path = os.path.join(data_dir, filename)
    model_data_used = load_data(data_path)
    temp_anom = calc_anomaly(model_data_used, num_years=171)
    
    ## initialise_plot
    fig, ax = plt.subplots(1)

    ## run model under different forcing scenarios
    for scen_file in os.listdir(path_to_ssp_forcings):
        print(scen_file)
        forcing_scenario_path = os.path.join(path_to_ssp_forcings, scen_file)
        ERF, ERF_fut = load_forcing_data(forcing_scenario_path)

    
        fitted_model, params = fit_model(temp_anom, upper_ocean_temp, t=t)

        ## SET CONSTRAINTS
        # Alpha is 1.04+-0.36 from CMIP6 (in slides)
        params['alpha'].min = 0.68
        params['alpha'].max = 1.40
        # best fit parameters from model
        A = fitted_model.params['A'].value
        B = fitted_model.params['B'].value
        
        alpha = fitted_model.params['alpha'].value
        projection = upper_ocean_temp(t=t_fut, A=A, B=B, F=ERF_fut, alpha=alpha)

        ## plot and save ouputs
        fig, ax = plot_model(years_fut, projection, label='model', fig=fig, ax=ax)
        fig, ax = plot_model(years, temp_anom, label='HadCRUT data', fig=fig, ax=ax)
    fig.savefig('../outputs/test3.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()