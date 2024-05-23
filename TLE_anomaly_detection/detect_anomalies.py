import sys
import os
import numpy as np
import pandas as pd
from functools import partial
import scipy.linalg as sla
import multiprocessing
import yaml
from scipy.stats import multivariate_normal
from sklearn.covariance import MinCovDet, EmpiricalCovariance
import pickle
from functools import partial

this_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_directory + "/../")
import dapper.da_methods as da
import dapper.mods as modelling
from dapper.mods.mean_element_propagation import step, live_plots

from dapper.tools.matrices import CovMat
from dapper.da_methods.particle import reweight

sys.path.append(this_directory + "/../../TLE_utilities/")
from utilities import (propagate_SatelliteTLEData_object,
                       get_np_mean_elements_from_satelliteTLEData_object,
                       load_tle_data_from_file,
                       DICT_ELEMENT_NAMES,
                       run_a_method_on_satellites)



# If set to False, Cartesian coordinates will be used
USE_KEPLERIAN_COORDINATES = True
# Plotting the sliding marginal time series slows things down a lot, so it's useful to be able to turn it off
PLOT_MARGINALS = False

# See DICT_ELEMENT_NAMES in TLE_Utilities/tle_loading_and_preprocessing.py for the specification of which elements
# these elements correspond with.
INDICES_FOR_MARGINAL_ANOMALY_DETECTION = [4]
INDICES_FOR_SHIFT_DECISION = [0, 2, 4, 5]

def assimilate_for_one_satellite(satelliteTLEData_satellites,
                                 dict_shared_parameters,
                                 dict_parameters,
                                 satellite_names_list = [],
                                 satellite_index = 0):

    process_pool = multiprocessing.Pool(dict_parameters["dapper n jobs"])

    # Estimate both the model and observation uncertainty as the covariance of the residuals when propagating the TLEs from
    # one epoch to the subsequent epoch.
    pd_df_propagated_mean_elements = propagate_SatelliteTLEData_object(satelliteTLEData_satellites, 1)
    pd_df_residuals = (pd_df_propagated_mean_elements - satelliteTLEData_satellites.pd_df_tle_data).dropna()

    #q1 = pd_df_residuals.quantile(0.25)
    #q3 = pd_df_residuals.quantile(0.75)
    q1 = pd_df_residuals.quantile(0.1)
    q3 = pd_df_residuals.quantile(0.9)
    iqr = q3 - q1
    pd_df_residuals = pd_df_residuals[~((pd_df_residuals < (q1 - dict_parameters["covariance outlier removal factor"] * iqr))
                                        | (pd_df_residuals > (q3 + dict_parameters["covariance outlier removal factor"] * iqr))).any(axis=1)]
    np_residuals = pd_df_residuals.values
    initial_residuals_covariance = EmpiricalCovariance(assume_centered = True).fit(np_residuals).covariance_
    # Normalise by variance to get an easier covariance matrix to work with
    normalisation_weights = np.sqrt(np.diag(initial_residuals_covariance))
    # TODO: find a more principled way of doing this
    #normalisation_weights[normalisation_weights == 0] = 1e-6
    final_residuals_covariance = EmpiricalCovariance(assume_centered = True).fit(np.divide(pd_df_residuals.values, normalisation_weights)).covariance_

    observation_covariance = np.diag(np.diag(final_residuals_covariance))

    propagation_covariance = np.copy(final_residuals_covariance)
    # See the paper associated with this work for why we are modifying the propagation covariance
    # Because the variables have been scaled, the covariance is already a correlation matrix
    scaling_matrix = np.diag(np.ones(6))
    scaling_matrix[3, 3] = np.sqrt(dict_parameters["anom and perigee variance scaling factor"])
    scaling_matrix[1, 1] = np.sqrt(dict_parameters["anom and perigee variance scaling factor"])
    propagation_covariance[1, 3] = -1
    propagation_covariance[3, 1] = -1
    propagation_covariance = scaling_matrix @ propagation_covariance @ scaling_matrix

    #print("eigenvalues of propagation covariance", sla.eigh(propagation_covariance)[0])

    xx = np.divide(
        get_np_mean_elements_from_satelliteTLEData_object(satelliteTLEData_satellites,
                                                          element_set = dict_shared_parameters["element set"]),
        normalisation_weights)
    # The observations are the same as the ground truth
    yy = np.copy(xx)[1:]
    x0 = xx[0, :]
    Nx = len(x0)

    observed_indices = np.arange(Nx)
    # Identity observation model
    Obs = modelling.partial_Id_Obs(Nx, observed_indices)
    Obs['noise'] = modelling.GaussRV(C = CovMat(observation_covariance, kind = 'full'))

    Dyn = {
        'M': Nx,
        'model': partial(step,
                         process_pool = process_pool,
                         satelliteTLEData_object = satelliteTLEData_satellites,
                         normalisation_weights = normalisation_weights,
                         element_set = dict_shared_parameters["element set"],
                         ),
        'noise': modelling.GaussRV(C = CovMat(propagation_covariance, kind = 'full')),
    }

    X0 = modelling.GaussRV(C = CovMat(observation_covariance, kind = 'full'), mu = x0)

    tseq = modelling.Chronology(dt = 1, dko = 1, Ko = xx.shape[0] - 2, Tplot = 20, BurnIn = 10)
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)

    if dict_parameters["method variant"] == "boot":
        xp = da.PartFilt(N = dict_parameters["num particles"], reg = 1, NER = 0.2, qroot = 1, wroot = 1)
    elif dict_parameters["method variant"] == "OP":
        xp = da.OptPF(N = dict_parameters["num particles"], reg = 1, NER = 0.2, Qs = 1, wroot = 1)
    else:
        print("Invalid method variant")
        quit()

    if PLOT_MARGINALS:
        HMM.liveplotters = live_plots(plot_marginals = PLOT_MARGINALS,
                                      use_keplerian_coordinates = USE_KEPLERIAN_COORDINATES,
                                      element_names = DICT_ELEMENT_NAMES[dict_shared_parameters["element set"]])
        xp.assimilate(HMM,
                      xx[:],
                      yy[:],
                      dict_method_parameters["shift threshold"],
                      INDICES_FOR_MARGINAL_ANOMALY_DETECTION,
                      INDICES_FOR_SHIFT_DECISION,
                      liveplots = True)
    else:
        xp.assimilate(HMM,
                      xx[:],
                      yy[:],
                      dict_parameters["shift threshold"],
                      INDICES_FOR_MARGINAL_ANOMALY_DETECTION,
                      INDICES_FOR_SHIFT_DECISION)

    pd_df_results = satelliteTLEData_satellites.pd_df_tle_data.iloc[1:]
    pd_df_results.loc[:, dict_shared_parameters["detection column name"]] = xp.negative_log_likelihoods
    pd_df_results.loc[:, dict_shared_parameters["secondary detection column name"]] = xp.marginal_negative_log_likelihoods
    pd_df_results = pd_df_results[[dict_shared_parameters["detection column name"],
                                   dict_shared_parameters["secondary detection column name"]]]


    # if dict_shared_parameters["satellite set"] == "simulated_video":
    #     # Saving for the purpose of animation creation
    #     pickle.dump(xp.particle_positions, open(dict_parameters["filter logs directory"] +
    #                                             satellite_names_list[satellite_index] +
    #                                             "_" + dict_parameters["method variant"] +
    #                                             dict_parameters["particle positions log suffix"], "wb"))
    #     pickle.dump(satelliteTLEData_satellites, open(dict_parameters["filter logs directory"] +
    #                                             satellite_names_list[satellite_index] +
    #                                             "_" + dict_parameters["method variant"] +
    #                                             dict_parameters["satellites log suffix"], "wb"))

    # pickle.dump(xp.n_effective, open(dict_parameters["filter logs directory"] +
    #                                  satellite_names_list[satellite_index] +
    #                                  "_" + dict_parameters["method variant"] +
    #                                  dict_parameters["n effective log suffix"], "wb"))
    return pd_df_results

if __name__ == "__main__":

    dict_shared_parameters = yaml.safe_load(open(sys.argv[1], 'r'))

    list_method_parameter_dicts = []
    for parameters_file_name in sys.argv[2:]:
        list_method_parameter_dicts.append(yaml.safe_load(open(parameters_file_name, 'r')))

    for dict_method_parameters in list_method_parameter_dicts:
        run_a_method_on_satellites(assimilate_for_one_satellite, dict_shared_parameters, dict_method_parameters, "../")