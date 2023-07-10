import sys
import os
import numpy as np
import dapper.mods as modelling
from skyfield.api import load as skyfield_load
from dapper.mods.mean_element_propagation import step, live_plots
import pandas as pd
import dapper.da_methods as da
from functools import partial
from dapper.tools.matrices import CovMat
import scipy.linalg as sla
import multiprocessing

from python_parameters import base_package_load_path
sys.path.insert(0, base_package_load_path)
from TLE_utilities.tle_loading_and_preprocessing import (propagate_SatelliteTLEData_object,
                                                         get_np_mean_elements_from_satelliteTLEData_object,
                                                         load_tle_data_from_file)

TLE_FILES_DIRECTORY = "/home/david/Projects/TLE_observation_benchmark_dataset/processed_files/"
TLE_FILE_NAME = "Fengyun-2E.tle"
#TLE_FILE_NAME = "Sentinel-3A.tle"
# If set to False, Cartesian coordinates will be used
USE_KEPLERIAN_COORDINATES = True
# Plotting the sliding marginal time series slows things down a lot, so it's useful to be able to turn it off
PLOT_MARGINALS = False

dict_shared_parameters = yaml.safe_load(open(sys.argv[1], 'r'))
dict_parameters = yaml.safe_load(open(sys.argv[2], 'r'))

process_pool = multiprocessing.Pool(dict_parameters["dapper n jobs"])

satelliteTLEData_satellites = load_tle_data_from_file(TLE_FILES_DIRECTORY + TLE_FILE_NAME, dict_parameters["start epoch"])

xx = get_np_mean_elements_from_satelliteTLEData_object(satelliteTLEData_satellites)

# The observations are then the same as the ground truth
yy = np.copy(xx)
yy = yy[1:]

x0 = xx[0, :]
Nx = len(x0)

# Estimate both the model and observation uncertainty as the covariance of the residuals when propagating the TLEs from
# one epoch to the subsequent epoch. There's probably a better way of doing this, but using this for now.
pd_df_propagated_mean_elements = propagate_SatelliteTLEData_object(satelliteTLEData_satellites, 1)
np_residuals = (pd_df_propagated_mean_elements - satelliteTLEData_satellites.pd_df_tle_data).dropna().values

residuals_covariance = np.cov(np_residuals, rowvar = False)
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
}

X0 = modelling.GaussRV(C = CovMat(residuals_covariance, kind ='full'), mu = x0)

tseq = modelling.Chronology(1, dko = 1, Ko = 1000, Tplot = 20, BurnIn = 80)
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
HMM.liveplotters = live_plots(plot_marginals = PLOT_MARGINALS, use_keplerian_coordinates = USE_KEPLERIAN_COORDINATES)
xp = da.PartFilt(N = dict_parameters["num particles"], reg = 1, NER = 1, qroot = 1, wroot = 1)
xp.assimilate(HMM, xx[:], yy[:], liveplots=True)
xp.stats.average_in_time()
xp.stats.replay()