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
COLORS = ['#f7564a', '#e6ac1c', '#5963f0']


sns.set_context('paper')
sns.set_style('white')

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


def load_forcing_data(filename, volcanic=True):
    ERF_data = pd.read_csv(filename)
    ERF_data = ERF_data.set_index('year')

    if volcanic == True:
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
    plt.ylim(-1,7)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Temperature Anomaly (K) (w.r.t. 1961-1990)', fontsize=10)
    ax.grid(True)
    if legend:
        plt.legend(loc='upper left')
    return fig, ax


def plot_volcanic_record(data):
    past_volcanic_record = len(data['volcanic'].loc[1850:2024]) 
    # print('sum volcanic record added = ', data['volcanic'].loc[1850:1850+77].values.sum())
    sns.set_style('white')
    fig, ax = plt.subplots(1, figsize=(7, 5))
    data['volcanic'].loc[1850:2024].plot(ax=ax)
    ax.grid(axis='y')
    ax.plot(np.arange(2024,2024+past_volcanic_record), data['volcanic'].loc[1850:2024].values)
    plt.vlines(2024, -2, 2,color='k', linestyle='--')
    plt.ylim(-2,2)
    plt.xlim(1850,2100)
    plt.xlabel('Year', size=13)
    plt.title("\'New\' Volcanic record", size=14)
    plt.ylabel('Effective Radiative Forcing (ERF)', size=13)
    plt.savefig('outputs/volcanic_record_extended.png', bbox_inches='tight')
    plt.close()

def plot_temp_anom(data, data2):
    fig, ax = plt.subplots(1, figsize=(7, 5))
    ax.plot(np.arange(1850,2021), data, marker='s', label='HadCRUT record')
    ax.plot(np.arange(1850,2101),data2, label='SSP5 projection')
    ax.grid(axis='y')
    plt.hlines(0,1850,2020, linestyle='--', color='k')
    plt.xlim(1850, 2020)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Temperature anomaly (K) (w.r.t. 1961-1990)', fontsize=12)
    plt.title("HadCRUT global 2 m temperature record", size=13)
    plt.legend(loc='upper left')
    plt.savefig('outputs/hadCRUT_time.png', bbox_inches='tight')
    plt.close()

def get_non_volcanic_results(scen_file, forcing_scenario_path, temp_anom, VOLCANIC_RESULTS, krakatwoa=False):
    ERF, ERF_fut = load_forcing_data(forcing_scenario_path, volcanic=False)
    alpha_val, alpha_stderr = model.get_opt_model(temp_anom=temp_anom, F=ERF)
    projection = model.upper_ocean_temp(t=len(ERF_fut), alpha=alpha_val, F=ERF_fut, krakatwoa=krakatwoa)
    proj_upper = model.upper_ocean_temp(t=len(ERF_fut), alpha=alpha_val+1.96*0.048, F=ERF_fut, krakatwoa=krakatwoa)
    proj_lower = model.upper_ocean_temp(t=len(ERF_fut), alpha=alpha_val-1.96*0.048, F=ERF_fut, krakatwoa=krakatwoa)
    VOLCANIC_RESULTS[scen_file[5:8] + 'non_volcanic'] = [proj_lower[-1], proj_upper[-1]]
    return VOLCANIC_RESULTS


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
    fig, ax = plot_model(years, temp_anom, label='HadCRUT', fig=fig, ax=ax, marker='s', markersize=2, linewidth=1)


    ## run model under different forcing scenarios
    scenario_files = sorted(os.listdir(path_to_ssp_forcings), reverse=True)
    for ind, scen_file in enumerate(scenario_files):
        forcing_scenario_path = os.path.join(path_to_ssp_forcings, scen_file)
        ## TODO: clean up
        get_non_volcanic_results(scen_file, forcing_scenario_path, temp_anom, VOLCANIC_RESULTS)
        ## TODO: clean up above
        ERF, ERF_fut = load_forcing_data(forcing_scenario_path)

        alpha_val, alpha_stderr = model.get_opt_model(temp_anom=temp_anom, F=ERF)
        projection = model.upper_ocean_temp(t=len(ERF_fut), alpha=alpha_val, F=ERF_fut, krakatwoa=krakatwoa)
        proj_upper = model.upper_ocean_temp(t=len(ERF_fut), alpha=alpha_val+1.96*0.048, F=ERF_fut, krakatwoa=krakatwoa)
        proj_lower = model.upper_ocean_temp(t=len(ERF_fut), alpha=alpha_val-1.96*0.048, F=ERF_fut, krakatwoa=krakatwoa)
        if not krakatwoa:
            ## IPCC
            # print("expected temperature anomaly for %s " % (scen_file[5:8]), proj_lower[-1], proj_upper[-1])
            VOLCANIC_RESULTS[scen_file[5:8]] = [proj_lower[-1], proj_upper[-1]]
            low_proj = model.upper_ocean_temp(t=len(ERF_fut), alpha=1.04-0.36, F=ERF_fut, krakatwoa=krakatwoa)
            high_proj = model.upper_ocean_temp(t=len(ERF_fut), alpha=1.04+0.36, F=ERF_fut, krakatwoa=krakatwoa)
            fig, ax = plot_model(years_fut, low_proj, fig=fig, ax=ax, alpha=.2, linestyle='--',  color=COLORS[ind], legend=False)
            fig, ax = plot_model(years_fut, high_proj, fig=fig, ax=ax, alpha=.2, linestyle='--', color=COLORS[ind], legend=False)
            ax.add_patch(mpatches.Rectangle((2105,low_proj[-1]),2, (high_proj[-1]- low_proj[-1]),facecolor=COLORS[ind],
                              clip_on=False,linewidth = 0,  alpha=.7))
            plt.text(2110, 7, r'AR5 $\alpha$ range')
            plt.text(2108, (high_proj.max() + low_proj.max())/2, '%s – RCP %s.%s' % (scen_file[4:8].upper(), scen_file[8:9], scen_file[9:10]), color=COLORS[ind])

        else:
            # print("krakatwoa: expected temperature anomaly for %s " % (scen_file[5:8]), proj_lower[-1], proj_upper[-1])
            VOLCANIC_RESULTS[scen_file[5:8] + '_krakatwoa'] = [proj_lower[-1], proj_upper[-1]]
            ax.add_patch(mpatches.Rectangle((2105,proj_lower[-1]),2, (proj_upper[-1]- proj_lower[-1]),facecolor=COLORS[ind],
                              clip_on=False,linewidth = 0,  alpha=.7))

        
        ## plot and save ouputs
        fig, ax = plot_model(years_fut, projection, label='%s' % (scen_file[:-16].replace('_', '–').upper()), fig=fig, ax=ax, color=COLORS[ind])
        fig, ax = plot_model(years_fut, proj_upper, label=None, fig=fig, ax=ax, alpha=.4, color=COLORS[ind])
        fig, ax = plot_model(years_fut, proj_lower, label=None, fig=fig, ax=ax, alpha=.4, color=COLORS[ind])
    fig.savefig(save_filename, bbox_inches='tight', dpi=300)
    plt.close()

    ## plot temp anomaly
    plot_temp_anom(temp_anom,projection)
    

if __name__ == '__main__':
    ## Store results
    VOLCANIC_RESULTS = {}
    main()
    main(krakatwoa=True, save_filename='outputs/upper_ocean_projection_volcanic_krakatwoa.png')
    
    print(VOLCANIC_RESULTS)
    ## comparison plot
    different_scenarios = ['sp1', 'sp4', 'sp5']
    different_types = ['non_volcanic', 'krakatwoa']
    counter = 2
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(7,4))
    for ax, scen in zip(axes, different_scenarios):
        ax.set_title('S' + scen.upper(), size=14)
        ax.grid(axis='x')
        for key in VOLCANIC_RESULTS.keys():
            if scen in key:
                if 'non_volcanic' in key:
                    ax.hlines(1, VOLCANIC_RESULTS[key][0], VOLCANIC_RESULTS[key][1], color=COLORS[counter])
                    ax.vlines(VOLCANIC_RESULTS[key][0], 0.9, 1.1, color=COLORS[counter])
                    ax.vlines(VOLCANIC_RESULTS[key][1], 0.9, 1.1, color=COLORS[counter])
                elif 'krakatwoa' in key:
                    ax.hlines(3, VOLCANIC_RESULTS[key][0], VOLCANIC_RESULTS[key][1], color=COLORS[counter])
                    ax.vlines(VOLCANIC_RESULTS[key][0], 2.9, 3.1, color=COLORS[counter])
                    ax.vlines(VOLCANIC_RESULTS[key][1], 2.9, 3.1, color=COLORS[counter])
                else:
                    ax.hlines(2, VOLCANIC_RESULTS[key][0], VOLCANIC_RESULTS[key][1], color=COLORS[counter])
                    ax.vlines(VOLCANIC_RESULTS[key][0], 1.9, 2.1, color=COLORS[counter])
                    ax.vlines(VOLCANIC_RESULTS[key][1], 1.9, 2.1, color=COLORS[counter])
        plt.yticks(np.arange(1,4,1), ['Non-volcanic', 'Volcanic', 'Krakatwoa'])
        counter -= 1
    # plt.ylim(0, 10)
    # plt.xlim(0,5)
    axes[0].set_ylabel('Experiment', size=12)
    axes[1].set_xlabel('Temperature Anomaly (K) (w.r.t 1961-1990)', size=12)
    plt.suptitle('2100 Temperature anomaly', size=14)
    plt.subplots_adjust(top=.85)
    fig.savefig('outputs/compare_results.png', bbox_inches='tight', dpi=300)
