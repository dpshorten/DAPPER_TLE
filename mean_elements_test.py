import sys
import os
import numpy as np
import dapper.mods as modelling
from skyfield.api import load as skyfield_load
from dapper.mods.mean_element_propagation import step, live_plots, convert_Skyfield_EarthSatellite_to_np_array
import pandas as pd
import dapper.da_methods as da
from functools import partial
from dapper.tools.matrices import CovMat
import scipy.linalg as sla
import multiprocessing

NUM_PROCESSES = 10
TLE_FILES_DIRECTORY = "/home/david/Projects/TLE_observation_benchmark_dataset/processed_files/"
TLE_FILE_NAME = "Fengyun-2D.tle"
START_EPOCH = 100
NUM_EPOCHS_FOR_UNCERTAINTY_ESTIMATION = 500
NUM_PARTICLES = 1000
# If set to False, Cartesian coordinates will be used
USE_KEPLERIAN_COORDINATES = True
# Plotting the sliding marginal time series slows things down a lot, so it's useful to be able to turn it off
PLOT_MARGINALS = True

process_pool = multiprocessing.Pool(NUM_PROCESSES)

# Having a store of the TLE lines allows us to create Skyfield EarthSatellite objects when required. Creating these
# objects directly from the TLE lines is a foolproof way of ensuring that settings such as epoch times are correct.
# Note that we don't want to create the EarthSatellite objects just yet as they are not pickle-able, so this will not be
# compatible with the Python multiprocessing library.
TLE_file = open(TLE_FILES_DIRECTORY + TLE_FILE_NAME, 'r')
list_of_TLE_line_pairs = []
current_TLE_line_pair = []
i = 0
for line in TLE_file:
    # First line in pair
    if i >= 2 * START_EPOCH and (i - START_EPOCH) % 2 == 0:
        current_TLE_line_pair = [line]
    # Second line in pair
    elif i >= 2 * START_EPOCH:
        current_TLE_line_pair.append(line)
        list_of_TLE_line_pairs.append(current_TLE_line_pair)
    i += 1

# Instantiate our 'ground-truth' of mean elements (xx) as the observed TLE values. This gives us something for the accuracy
# metrics to compare against.
xx = []
list_of_Skyfield_EarthSatellites = skyfield_load.tle_file(TLE_FILES_DIRECTORY + TLE_FILE_NAME, reload = False)[START_EPOCH:]
for sat in list_of_Skyfield_EarthSatellites:
    xx.append(convert_Skyfield_EarthSatellite_to_np_array(sat, use_keplerian_coordinates = USE_KEPLERIAN_COORDINATES))
xx = np.array(xx)

# The observations are then the same as the ground truth
yy = np.copy(xx)

x0 = convert_Skyfield_EarthSatellite_to_np_array(list_of_Skyfield_EarthSatellites[0], use_keplerian_coordinates = USE_KEPLERIAN_COORDINATES)
Nx = len(x0)

# Estimate both the model and observation uncertainty as the covariance of the residuals when propagating the TLEs from
# one epoch to the subsequent epoch. There's probably a better way of doing this, but using this for now.
propagations = np.zeros((NUM_EPOCHS_FOR_UNCERTAINTY_ESTIMATION, xx.shape[1]))
for i in range(NUM_EPOCHS_FOR_UNCERTAINTY_ESTIMATION):
    propagations[i, :] = np.squeeze(step(np.expand_dims(xx[i], axis = 0),
                                         i,
                                         1,
                                         process_pool,
                                         list_of_TLE_line_pairs,
                                         USE_KEPLERIAN_COORDINATES),
                                    axis = 0)
residuals = propagations[:NUM_EPOCHS_FOR_UNCERTAINTY_ESTIMATION, :] - xx[1:(NUM_EPOCHS_FOR_UNCERTAINTY_ESTIMATION + 1), :]
residuals_covariance = np.cov(residuals, rowvar = False)
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
                     list_of_TLE_line_pairs = list_of_TLE_line_pairs,
                     use_keplerian_coordinates = USE_KEPLERIAN_COORDINATES),
    'noise': modelling.GaussRV(C = CovMat(residuals_covariance, kind ='full')),
}

X0 = modelling.GaussRV(C = CovMat(residuals_covariance, kind ='full'), mu = x0)

tseq = modelling.Chronology(1, dko = 1, Ko = 1000, Tplot = 20, BurnIn = 80)
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
HMM.liveplotters = live_plots(plot_marginals = PLOT_MARGINALS)
xp = da.PartFilt(N = NUM_PARTICLES, reg = 1, NER = 1, qroot = 1, wroot = 1)
xp.assimilate(HMM, xx[:], yy[:], liveplots=True)
xp.stats.average_in_time()
xp.stats.replay()