import os 
import pandas as pd
import numpy as np
from scripts import model
from scripts.model import KRAK_VALS, KRAKATOA_YEAR
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

## GLOBALS
FORCING_SENSITIVITY = 1

sns.set_context('paper')
sns.set_style('whitegrid')

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


def load_forcing_data(filename):
    ERF_data = pd.read_csv(filename)
    ERF_data = ERF_data.set_index('year')
    plot_volcanic_record(ERF_data)
    
    KRAK_VALS[1883] = ERF_data['volcanic'][KRAKATOA_YEAR]
    KRAK_VALS[1884] = ERF_data['volcanic'][KRAKATOA_YEAR+1]
    KRAK_VALS[1885] = ERF_data['volcanic'][KRAKATOA_YEAR+2]
    KRAK_VALS[1886] = ERF_data['volcanic'][KRAKATOA_YEAR+3]

    past_volcanic_record = len(ERF_data['volcanic'].loc[1850:2024]) 
    new_vals = list(ERF_data['total'].loc[:2024].values)
    new_vals.extend(ERF_data['total'].loc[2024:2023+past_volcanic_record].values + ERF_data['volcanic'].loc[1850:2024].values)
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


def plot_model(years, model, ax=None, fig=None, legend=True, **kwargs):
    if not ax:
        fig, ax = plt.subplots(1)
    plt.plot(years, model, **kwargs)
    plt.hlines(0,1850,2100, linestyle='--', color='k')
    plt.xlim(1850, 2100)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Temperature Anomaly (K)', fontsize=12)
    if legend:
        plt.legend()
    return fig, ax


def plot_volcanic_record(data):
    past_volcanic_record = len(data['volcanic'].loc[1850:2024]) 

    fig, ax = plt.subplots(1, figsize=(10, 6))
    data['volcanic'].loc[1850:2024].plot(ax=ax)
    ax.plot(np.arange(2024,2024+past_volcanic_record), data['volcanic'].loc[1850:2024].values)
    plt.savefig('outputs/volcanic_record_extended.png', bbox_inches='tight')
    plt.close()

def plot_temp_anom(data, data2):
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.plot(np.arange(1850,2021), data, marker='s', label='HadCRUT surface temperature')
    ax.plot(np.arange(1850,2101),data2, label='SSP5 projection')
    plt.hlines(0,1850,2020, linestyle='--', color='k')
    plt.xlim(1850, 2020)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('\Delta Temperature Anomaly (K)', fontsize=12)
    plt.title("HadCRUT global 2 m temperature anomaly (relative to 1961-1990)")
    plt.legend(loc='upper left')
    plt.savefig('outputs/hadCRUT_time.png', bbox_inches='tight')
    plt.close()



def main(krakatwoa=False, save_filename='outputs/upper_ocean_projection_volcanic.png'):
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
    fig, ax = plt.subplots(1, figsize=(10,6))
    fig, ax = plot_model(years, temp_anom, label='HadCRUT temperature  anomaly', fig=fig, ax=ax, marker='s', markersize=2, linewidth=1)
    COLORS = ['#f7564a', '#e6ac1c', '#5963f0']
    

    ## run model under different forcing scenarios
    scenario_files = sorted(os.listdir(path_to_ssp_forcings), reverse=True)
    for ind, scen_file in enumerate(scenario_files):

        forcing_scenario_path = os.path.join(path_to_ssp_forcings, scen_file)
        ERF, ERF_fut = load_forcing_data(forcing_scenario_path)

        alpha_val, alpha_stderr = model.get_opt_model(temp_anom=temp_anom, F=ERF)
        projection = model.upper_ocean_temp(t=len(ERF_fut), alpha=alpha_val, F=ERF_fut, krakatwoa=krakatwoa)
        proj_upper = model.upper_ocean_temp(t=len(ERF_fut), alpha=alpha_val+1.96*0.048, F=ERF_fut, krakatwoa=krakatwoa)
        proj_lower = model.upper_ocean_temp(t=len(ERF_fut), alpha=alpha_val-1.96*0.048, F=ERF_fut, krakatwoa=krakatwoa)
        if not krakatwoa:
            ## IPCC
            low_proj = model.upper_ocean_temp(t=len(ERF_fut), alpha=1.04-0.36, F=ERF_fut, krakatwoa=krakatwoa)
            high_proj = model.upper_ocean_temp(t=len(ERF_fut), alpha=1.04+0.36, F=ERF_fut, krakatwoa=krakatwoa)
            fig, ax = plot_model(years_fut, low_proj, fig=fig, ax=ax, alpha=.2, linestyle='--',  color=COLORS[ind], legend=False)
            fig, ax = plot_model(years_fut, high_proj, fig=fig, ax=ax, alpha=.2, linestyle='--', color=COLORS[ind], legend=False)
            ax.add_patch(mpatches.Rectangle((2105,low_proj.max()),2, (high_proj.max()- low_proj.max()),facecolor=COLORS[ind],
                              clip_on=False,linewidth = 0,  alpha=.7))
            plt.text(2110, 7, r'AR5 $\alpha$ range')
            plt.text(2108, (high_proj.max() + low_proj.max())/2, '%s – RCP %s.%s' % (scen_file[4:8].upper(), scen_file[8:9], scen_file[9:10]), color=COLORS[ind])


        
        ## plot and save ouputs
        fig, ax = plot_model(years_fut, projection, label='%s' % (scen_file[:-16].replace('_', '–').upper()), fig=fig, ax=ax, color=COLORS[ind])
        fig, ax = plot_model(years_fut, proj_upper, label=None, fig=fig, ax=ax, alpha=.4, color=COLORS[ind])
        fig, ax = plot_model(years_fut, proj_lower, label=None, fig=fig, ax=ax, alpha=.4, color=COLORS[ind])
    fig.savefig(save_filename, bbox_inches='tight', dpi=300)
    plt.close()

    ## plot temp anomaly
    plot_temp_anom(temp_anom,projection)
    
    ## TODO: move following somewhere 
    # with SSP5 -> plot value range from IPCC
    if not krakatwoa:
        low_proj = model.upper_ocean_temp(t=len(ERF_fut), alpha=1.04-0.36, F=ERF_fut, krakatwoa=krakatwoa)
        high_proj = model.upper_ocean_temp(t=len(ERF_fut), alpha=1.04+0.36, F=ERF_fut, krakatwoa=krakatwoa)

        ## plot and save ouputs
        fig, ax = plt.subplots(1, figsize=(10,6))
        fig, ax = plot_model(years_fut, low_proj, label='Low alpha', fig=fig, ax=ax, color=COLORS[ind])
        fig, ax = plot_model(years_fut, projection, label='%s' % (scen_file[:-16].replace('_', '–').upper()), fig=fig, ax=ax, color=COLORS[ind])
        fig, ax = plot_model(years_fut, high_proj, label='High alpha', fig=fig, ax=ax, alpha=.4, color=COLORS[ind])
        fig.savefig('outputs/comparison_with_ipcc_alpha_range.png', bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == '__main__':
    main()
    main(krakatwoa=True, save_filename='outputs/upper_ocean_projection_volcanic_krakatwoa.png')