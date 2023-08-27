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
import pickle

from sklearn.covariance import MinCovDet, EmpiricalCovariance

from python_parameters import base_package_load_path
sys.path.insert(0, base_package_load_path)
sys.path.insert(0, base_package_load_path + "/TLE_utilities")
from tle_loading_and_preprocessing import (propagate_SatelliteTLEData_object,
                                                         get_np_mean_elements_from_satelliteTLEData_object,
                                                         load_tle_data_from_file,
                                                         DICT_ELEMENT_NAMES)
# If set to False, Cartesian coordinates will be used
USE_KEPLERIAN_COORDINATES = True
# Plotting the sliding marginal time series slows things down a lot, so it's useful to be able to turn it off
PLOT_MARGINALS = True

#INDICES_FOR_ANOMALY_DETECTION = [4]

def assimilate_for_one_satellite(satelliteTLEData_satellites,
                                 dict_shared_parameters,
                                 dict_parameters,
                                 satellite_names_list,
                                 satellite_index):

    process_pool = multiprocessing.Pool(dict_parameters["dapper n jobs"])

    # Estimate both the model and observation uncertainty as the covariance of the residuals when propagating the TLEs from
    # one epoch to the subsequent epoch. There's probably a better way of doing this, but using this for now.
    pd_df_propagated_mean_elements = propagate_SatelliteTLEData_object(satelliteTLEData_satellites, 1)
    pd_df_residuals = (pd_df_propagated_mean_elements - satelliteTLEData_satellites.pd_df_tle_data).dropna()

    #initial_residuals_covariance = MinCovDet(assume_centered=True).fit(pd_df_residuals.values).covariance_
    #normalisation_weights = np.sqrt(np.diag(initial_residuals_covariance))
    normalisation_weights = np.ones(6)
    final_residuals_covariance = MinCovDet(assume_centered=True).fit(np.divide(pd_df_residuals.values, normalisation_weights)).covariance_

    propagation_covariance = np.copy(final_residuals_covariance)
    propagation_covariance[3, 3] *= 1
    propagation_covariance[1, 1] *= 1

    #propagation_covariance[1, 3] = -0.9 * np.sqrt(final_residuals_covariance[1, 1]) * np.sqrt(final_residuals_covariance[3, 3])
    #propagation_covariance[3, 1] = -0.9 * np.sqrt(final_residuals_covariance[1,1]) * np.sqrt(final_residuals_covariance[3,3])

    # Print out the eigenvalues of this covariance matrix, as this is what is leading to the Dapper error:
    # 'Rank-deficient R not supported.'
    print("eigenvalues of residuals covariance", sla.eigh(final_residuals_covariance)[0])
    print("eigenvalues of propagation covariance", sla.eigh(propagation_covariance)[0])

    xx = np.divide(
        get_np_mean_elements_from_satelliteTLEData_object(satelliteTLEData_satellites,
                                                          element_set=dict_shared_parameters["element set"]),
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
                         element_set=dict_shared_parameters["element set"],
                         ),
        'noise': modelling.GaussRV(C = CovMat(propagation_covariance, kind ='full')),
        #'noise': modelling.GaussRV(C=CovMat(0*residuals_covariance, kind='full')),
    }

    X0 = modelling.GaussRV(C = CovMat(final_residuals_covariance, kind ='full'), mu = x0)
    #X0 = modelling.GaussRV(C=CovMat(0*residuals_covariance, kind='full'), mu=x0)

    tseq = modelling.Chronology(dt=1, dko=1, Ko=xx.shape[0]-2, Tplot=20, BurnIn=10)
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

    anomaly_threshold = 1e-20
    print("anomaly threshold", anomaly_threshold)

    #xp = da.PartFilt(N = dict_parameters["num particles"], reg = 1, NER = 0.5, qroot = 1, wroot = 1)
    xp = da.OptPF(N=dict_parameters["num particles"], reg=1, NER=0.2, Qs=1, wroot=1)
    if PLOT_MARGINALS:
        HMM.liveplotters = live_plots(plot_marginals = PLOT_MARGINALS,
                                      use_keplerian_coordinates = USE_KEPLERIAN_COORDINATES,
                                      element_names=DICT_ELEMENT_NAMES[dict_shared_parameters["element set"]])
        xp.assimilate(HMM, xx[:], yy[:], anomaly_threshold, liveplots = True)
    else:
        xp.assimilate(HMM, xx[:], yy[:], anomaly_threshold)

    pd_df_results = satelliteTLEData_satellites.pd_df_tle_data.iloc[1:]
    pd_df_results.loc[:, dict_shared_parameters["detection column name"]] = xp.likelihoods
    pd_df_results = pd_df_results[[dict_shared_parameters["detection column name"]]]

    # Saving for the purpose of animation creation
    pickle.dump(xp.particle_positions, open(dict_parameters["filter logs directory"] +
                                            satellite_names_list[satellite_index] +
                                            dict_parameters["particle positions log suffix"], "wb"))
    pickle.dump(satelliteTLEData_satellites, open(dict_parameters["filter logs directory"] +
                                            satellite_names_list[satellite_index] +
                                            dict_parameters["satellites log suffix"], "wb"))
    return pd_df_results

dict_shared_parameters = yaml.safe_load(open(sys.argv[1], 'r'))
dict_parameters = yaml.safe_load(open(sys.argv[2], 'r'))

run_a_method_on_satellites(assimilate_for_one_satellite, dict_shared_parameters, dict_parameters, "../")