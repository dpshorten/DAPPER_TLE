import sys
import os
import numpy as np
import dapper.mods as modelling
from dapper.mods.mean_element_propagation import step, live_plots
import pandas as pd
import dapper.da_methods as da
from functools import partial
from dapper.tools.matrices import CovMat
import scipy.linalg as sla
import multiprocessing
import yaml

from python_parameters import base_package_load_path
sys.path.insert(0, base_package_load_path)
from TLE_utilities.tle_loading_and_preprocessing import (propagate_SatelliteTLEData_object,
                                                         get_np_mean_elements_from_satelliteTLEData_object,
                                                         load_tle_data_from_file)
# If set to False, Cartesian coordinates will be used
USE_KEPLERIAN_COORDINATES = True
# Plotting the sliding marginal time series slows things down a lot, so it's useful to be able to turn it off
PLOT_MARGINALS = True

def assimilate_for_one_satellite(satellite_index, dict_shared_parameters, dict_parameters):

    print("assimilating:", dict_shared_parameters["satellite names list"][satellite_index])

    process_pool = multiprocessing.Pool(dict_parameters["dapper n jobs"])

    satelliteTLEData_satellites = load_tle_data_from_file(dict_shared_parameters["data files directory"] +
                                                          dict_shared_parameters["satellite names list"][satellite_index] +
                                                          dict_shared_parameters["tle file suffix"]
                                                          , dict_parameters["start epoch"])

    xx = get_np_mean_elements_from_satelliteTLEData_object(satelliteTLEData_satellites)

    # The observations are then the same as the ground truth
    yy = np.copy(xx)
    yy = yy[1:]

    x0 = xx[0, :]
    Nx = len(x0)

    # Estimate both the model and observation uncertainty as the covariance of the residuals when propagating the TLEs from
    # one epoch to the subsequent epoch. There's probably a better way of doing this, but using this for now.
    pd_df_propagated_mean_elements = propagate_SatelliteTLEData_object(satelliteTLEData_satellites, 1)
    pd_df_residuals = (pd_df_propagated_mean_elements - satelliteTLEData_satellites.pd_df_tle_data).dropna()
    # Remove outliers for covariance estimation
    q1 = pd_df_residuals.quantile(0.1)
    q3 = pd_df_residuals.quantile(0.9)
    iqr = q3 - q1
    pd_df_residuals = pd_df_residuals[~((pd_df_residuals < (q1 - 10 * iqr)) | (pd_df_residuals > (q3 + 10 * iqr))).any(axis=1)]
    residuals_covariance = np.cov(pd_df_residuals.values, rowvar = False)
    # Print out the eigenvalues of this covariance matrix, as this is what is leading to the Dapper error:
    # 'Rank-deficient R not supported.'
    print("eigenvalues of residuals covariance", sla.eigh(residuals_covariance)[0])

    observed_indices = np.arange(Nx)
    # Identity observation model
    Obs = modelling.partial_Id_Obs(Nx, observed_indices)
    Obs['noise'] = modelling.GaussRV(C = CovMat(residuals_covariance, kind ='full'))

    Dyn = {
        'M': Nx,
        'model': partial(step,
                         process_pool = process_pool,
                         satelliteTLEData_object = satelliteTLEData_satellites,
                         use_keplerian_coordinates = USE_KEPLERIAN_COORDINATES),
        'noise': modelling.GaussRV(C = CovMat(residuals_covariance, kind ='full')),
        #'noise': modelling.GaussRV(C=CovMat(0*residuals_covariance, kind='full')),
    }

    X0 = modelling.GaussRV(C = CovMat(residuals_covariance, kind ='full'), mu = x0)
    #X0 = modelling.GaussRV(C=CovMat(0*residuals_covariance, kind='full'), mu=x0)

    tseq = modelling.Chronology(dt=1, dko=1, Ko=xx.shape[0]-2, Tplot=20, BurnIn=10)
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    HMM.liveplotters = live_plots(plot_marginals = PLOT_MARGINALS, use_keplerian_coordinates = USE_KEPLERIAN_COORDINATES)
    xp = da.PartFilt(N = dict_parameters["num particles"], reg = 1, NER = 1, qroot = 1, wroot = 1)
    xp.assimilate(HMM, xx[:], yy[:], liveplots=True)

    pd_df_results = satelliteTLEData_satellites.pd_df_tle_data.iloc[1:]
    pd_df_results.loc[:, dict_shared_parameters["detection column name"]] = -xp.likelihoods
    pd_df_results = pd_df_results[[dict_shared_parameters["detection column name"]]]
    pd_df_results.to_pickle(dict_parameters["detection results save directory"] +
                            dict_shared_parameters["satellite names list"][satellite_index] +
                            dict_parameters["detection results save suffix"])

dict_shared_parameters = yaml.safe_load(open(sys.argv[1], 'r'))
dict_parameters = yaml.safe_load(open(sys.argv[2], 'r'))

#for i in range(len(dict_shared_parameters["satellite names list"])):
for i in [2]:
    assimilate_for_one_satellite(i, dict_shared_parameters, dict_parameters)