import sys
import os
import numpy as np
import dapper.mods as modelling
from dapper.mods.mean_element_propagation import step, live_plots
import pandas as pd
import dapper.da_methods as da
from functools import partial
from dapper.tools.matrices import CovMat
from dapper.da_methods.particle import reweight
import scipy.linalg as sla
import multiprocessing
import yaml
from scipy.stats import multivariate_normal
from TLE_utilities.evaluation import run_a_method_on_satellites

from python_parameters import base_package_load_path
sys.path.insert(0, base_package_load_path)
from TLE_utilities.tle_loading_and_preprocessing import (propagate_SatelliteTLEData_object,
                                                         get_np_mean_elements_from_satelliteTLEData_object,
                                                         load_tle_data_from_file)
# If set to False, Cartesian coordinates will be used
USE_KEPLERIAN_COORDINATES = True
# Plotting the sliding marginal time series slows things down a lot, so it's useful to be able to turn it off
PLOT_MARGINALS = True

INDICES_FOR_ANOMALY_DETECTION = [2, 4]

def assimilate_for_one_satellite(satelliteTLEData_satellites, dict_shared_parameters, dict_parameters, satellite_index):

    print("assimilating:", dict_shared_parameters["satellite names list"][satellite_index])

    process_pool = multiprocessing.Pool(dict_parameters["dapper n jobs"])

    # Estimate both the model and observation uncertainty as the covariance of the residuals when propagating the TLEs from
    # one epoch to the subsequent epoch. There's probably a better way of doing this, but using this for now.
    pd_df_propagated_mean_elements = propagate_SatelliteTLEData_object(satelliteTLEData_satellites, 1)
    pd_df_residuals = (pd_df_propagated_mean_elements - satelliteTLEData_satellites.pd_df_tle_data).dropna()
    # Remove outliers for covariance estimation
    q1 = pd_df_residuals.quantile(0.1)
    q3 = pd_df_residuals.quantile(0.9)
    iqr = q3 - q1
    pd_df_residuals = pd_df_residuals[~((pd_df_residuals < (q1 - 10 * iqr)) | (pd_df_residuals > (q3 + 10 * iqr))).any(axis=1)]
    initial_residuals_covariance = np.cov(pd_df_residuals.values, rowvar = False)
    normalisation_weights = np.sqrt(np.diag(initial_residuals_covariance))
    #normalisation_weights = np.ones(initial_residuals_covariance.shape[0])
    final_residuals_covariance = np.cov(np.divide(pd_df_residuals.values, normalisation_weights), rowvar = False)
    final_residuals_covariance[1, 3] = 0 * np.sqrt(final_residuals_covariance[1,1]) * np.sqrt(final_residuals_covariance[3,3])
    final_residuals_covariance[3, 1] = 0 * np.sqrt(final_residuals_covariance[1,1]) * np.sqrt(final_residuals_covariance[3,3])

    covariance_in_anomaly_dimensions = np.zeros((len(INDICES_FOR_ANOMALY_DETECTION), len(INDICES_FOR_ANOMALY_DETECTION)))
    for index_1 in range(len(INDICES_FOR_ANOMALY_DETECTION)):
        for index_2 in range(len(INDICES_FOR_ANOMALY_DETECTION)):
            covariance_in_anomaly_dimensions[index_1, index_2] = final_residuals_covariance[INDICES_FOR_ANOMALY_DETECTION[index_1],
                                                                                            INDICES_FOR_ANOMALY_DETECTION[index_2]]
    # Print out the eigenvalues of this covariance matrix, as this is what is leading to the Dapper error:
    # 'Rank-deficient R not supported.'
    print("eigenvalues of residuals covariance", sla.eigh(final_residuals_covariance)[0])

    xx = np.divide(
        get_np_mean_elements_from_satelliteTLEData_object(satelliteTLEData_satellites),
        normalisation_weights)
    # The observations are then the same as the ground truth
    yy = np.copy(xx)
    yy = yy[1:]
    x0 = xx[0, :]
    Nx = len(x0)


    observed_indices = np.arange(Nx)
    # Identity observation model
    Obs = modelling.partial_Id_Obs(Nx, observed_indices)
    Obs['noise'] = modelling.GaussRV(C = CovMat(final_residuals_covariance, kind ='full'))

    Dyn = {
        'M': Nx,
        'model': partial(step,
                         process_pool = process_pool,
                         satelliteTLEData_object = satelliteTLEData_satellites,
                         normalisation_weights=normalisation_weights,
                         use_keplerian_coordinates = USE_KEPLERIAN_COORDINATES,
                         ),
        'noise': modelling.GaussRV(C = CovMat(final_residuals_covariance, kind ='full')),
        #'noise': modelling.GaussRV(C=CovMat(0*residuals_covariance, kind='full')),
    }

    X0 = modelling.GaussRV(C = CovMat(final_residuals_covariance, kind ='full'), mu = x0)
    #X0 = modelling.GaussRV(C=CovMat(0*residuals_covariance, kind='full'), mu=x0)

    tseq = modelling.Chronology(dt=1, dko=1, Ko=xx.shape[0]-2, Tplot=20, BurnIn=10)
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

    # get an idea of the range of likelihoods
    np_likelihood_estimate_values = np.zeros(xx.shape[0] - 1)
    for i in range(xx.shape[0] - 1):
        Xi = modelling.GaussRV(C = CovMat(final_residuals_covariance, kind ='full'), mu = xx[i])
        E = Xi.sample(50)
        innovs = (yy[i] - HMM.Obs(E, 1)) @ HMM.Obs.noise.C.sym_sqrt_inv
        #w = reweight(np.ones(E.shape[0]), innovs=innovs)
        w = np.ones(E.shape[0])
        E = HMM.Dyn(E, i, 1)
        np_cov_mat = np.array(HMM.Obs.noise.C.full)
        likelihood = 0
        for j in range(E.shape[0]):
            likelihood += np.log(w[j]) + multivariate_normal.logpdf(yy[i], mean=E[j], cov=np_cov_mat, allow_singular = True)
        np_likelihood_estimate_values[i] = likelihood
    percentiles = np.percentile(np_likelihood_estimate_values, [10, 90])
    anomaly_threshold = percentiles[0] - 0.0 * (percentiles[1] - percentiles[0])
    print(np_likelihood_estimate_values)
    print("anomaly threshold", anomaly_threshold)
    #quit()

    HMM.liveplotters = live_plots(plot_marginals = PLOT_MARGINALS, use_keplerian_coordinates = USE_KEPLERIAN_COORDINATES)
    xp = da.PartFilt(N = dict_parameters["num particles"], reg = 1, NER = 1, qroot = 1, wroot = 1)
    xp.assimilate(HMM, xx[:], yy[:], anomaly_threshold, INDICES_FOR_ANOMALY_DETECTION, covariance_in_anomaly_dimensions, liveplots=True)
    #xp.assimilate(HMM, xx[:], yy[:], anomaly_threshold, INDICES_FOR_ANOMALY_DETECTION, covariance_in_anomaly_dimensions)

    pd_df_results = satelliteTLEData_satellites.pd_df_tle_data.iloc[1:]
    pd_df_results.loc[:, dict_shared_parameters["detection column name"]] = xp.likelihoods
    pd_df_results = pd_df_results[[dict_shared_parameters["detection column name"]]]
    return pd_df_results

dict_shared_parameters = yaml.safe_load(open(sys.argv[1], 'r'))
dict_parameters = yaml.safe_load(open(sys.argv[2], 'r'))

run_a_method_on_satellites(assimilate_for_one_satellite, dict_shared_parameters, dict_parameters)