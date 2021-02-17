import os 
import pandas as pd
import numpy as np
from scripts import model
from scripts.model import KRAK_VALS, KRAKATOA_YEAR
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

    return anom


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
    KRAK_VALS[1883] = ERF_data['volcanic'][KRAKATOA_YEAR]
    KRAK_VALS[1884] = ERF_data['volcanic'][KRAKATOA_YEAR+1]
    KRAK_VALS[1885] = ERF_data['volcanic'][KRAKATOA_YEAR+2]
    KRAK_VALS[1886] = ERF_data['volcanic'][KRAKATOA_YEAR+3]

    past_volcanic_record = len(ERF_data['volcanic'].loc[1850:2024]) 
    new_vals = list(ERF_data['total'].loc[:2024].values)
    new_vals.extend(ERF_data['total'].loc[2025:2024+past_volcanic_record].values + ERF_data['volcanic'].loc[1850:2024].values)
    new_vals.extend(ERF_data['total'].loc[2024+past_volcanic_record+1:].values)
    ERF_data['total'] = new_vals

    ERF = np.array(ERF_data.loc[1850:2020]['total']) * FORCING_SENSITIVITY
    ERF_fut = np.array(ERF_data.loc[1850:2100]['total'] * FORCING_SENSITIVITY)
    return ERF, ERF_fut


def calc_confidence_interval(data):
    n = len(data)
    x_bar = data.mean()
    st_dev = np.std(data) 
    upper_conf_int = x_bar + 1.960 * st_dev/np.sqrt(n)
    lower_conf_int = x_bar - 1.960 * st_dev/np.sqrt(n)
    return upper_conf_int, lower_conf_int


def main():
    # array for time, in years and seconds
    t = np.array(range(0,171), dtype='int64')
    years = t + 1850
    t_fut = np.array(range(0,251), dtype='int64')
    years_fut = t_fut + 1850
    

    ## file locations
    data_dir = './data'
    filename = 'hadCRUT_data.txt'
    path_to_ssp_forcings = os.path.join(data_dir, 'SSPs/')

    ## load data and calc temperature anomaly
    data_path = os.path.join(data_dir, filename)
    model_data_used = load_data(data_path)
    temp_anom = calc_anomaly(model_data_used, num_years=171)
    
    ## initialise_plot
    fig, ax = plt.subplots(1, figsize=(10,8))
    fig, ax = plot_model(years, temp_anom, label='HadCRUT temperature  anomaly', fig=fig, ax=ax)

    COLORS = []

    ## run model under different forcing scenarios
    for ind, scen_file in enumerate(os.listdir(path_to_ssp_forcings)):

        forcing_scenario_path = os.path.join(path_to_ssp_forcings, scen_file)
        ERF, ERF_fut = load_forcing_data(forcing_scenario_path)

        alpha_val, alpha_stderr = model.get_opt_model(temp_anom=temp_anom, F=ERF)
        projection = model.upper_ocean_temp(t=len(ERF_fut), alpha=alpha_val, F=ERF_fut)
        proj_upper = model.upper_ocean_temp(t=len(ERF_fut), alpha=alpha_val+1.96*0.048, F=ERF_fut)
        proj_lower = model.upper_ocean_temp(t=len(ERF_fut), alpha=alpha_val-1.96*0.048, F=ERF_fut)

        ## plot and save ouputs
        fig, ax = plot_model(years_fut, projection, label='%s' % (scen_file[:-3]), fig=fig, ax=ax)
        fig, ax = plot_model(years_fut, proj_upper, label=None, fig=fig, ax=ax)
        fig, ax = plot_model(years_fut, proj_lower, label=None, fig=fig, ax=ax)
    fig.savefig('outputs/upper_ocean_projection_volcanic.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    main()