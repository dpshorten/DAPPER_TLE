import sys
import os
import numpy as np
import dapper.mods as modelling
from skyfield.api import load as skyfield_load
from dapper.mods.mean_element_propagation import step, live_plots, convert_skyfield_earth_satellite_to_np_array
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
# If set to False, Cartesian coordinates will be used
USE_KEPLERIAN_COORDINATES = True
# Plotting the sliding marginal time series slows things down a lot, so it's useful to be able to turn it off
PLOT_MARGINALS = True


# Having a store of the TLE lines allows us to create Skyfield EarthSatellite objects when required. Creating these
# objects directly from the TLE lines is a foolproof way of ensuring that settings such as epoch times are correct.
# Note that we don't want to create the EarthSatellites just yet as they are not pickle-able, so this will not be
# compatible with the Python multiprocessing library.
TLE_file = open(TLE_FILES_DIRECTORY + TLE_FILE_NAME, 'r')
list_of_TLE_line_pairs = []
current_TLE_line_pair = []
i = 0
for line in TLE_file:
    if i < START_EPOCH:
        pass
    elif (i - START_EPOCH) % 2 == 0:
        current_TLE_line_pair = [line]
    else:
        current_TLE_line_pair.append(line)
        list_of_TLE_line_pairs.append(current_TLE_line_pair)
    i += 1



Tplot = 20
tseq = modelling.Chronology(1, dko=1, Ko=1000, Tplot=Tplot, BurnIn=4*Tplot)

xx = []
list_of_Skyfield_EarthSatellites = skyfield_load.tle_file(TLE_FILES_DIRECTORY + TLE_FILE_NAME, reload=False)[START_EPOCH:]
for sat in list_of_Skyfield_EarthSatellites:
    xx.append(convert_skyfield_earth_satellite_to_np_array(sat, use_keplerian_coordinates=USE_KEPLERIAN_COORDINATES))

xx = np.array(xx)
yy = np.copy(xx)

x0 = convert_skyfield_earth_satellite_to_np_array(list_of_Skyfield_EarthSatellites[0], use_keplerian_coordinates=USE_KEPLERIAN_COORDINATES)
Nx = len(x0)

jj = np.arange(Nx)  # obs_inds
Obs = modelling.partial_Id_Obs(Nx, jj)

process_pool = multiprocessing.Pool(NUM_PROCESSES)

props = np.zeros((xx.shape[0]-1, xx.shape[1]))
for i in range(500):
    props[i, :] = np.squeeze(step(np.expand_dims(xx[i], axis = 0), i, 1,
                                  process_pool,
                                  list_of_TLE_line_pairs,
                                  USE_KEPLERIAN_COORDINATES), axis = 0)
errors = props[:500] - xx[1:501, :]
errors_cov = np.cov(errors, rowvar = False)

Obs['noise'] = modelling.GaussRV(C=CovMat(1e0 * errors_cov, kind = 'full'))

Dyn = {
    'M': Nx,
    'model': partial(step,
                     process_pool = process_pool,
                     list_of_TLE_line_pairs = list_of_TLE_line_pairs,
                     use_keplerian_coordinates = USE_KEPLERIAN_COORDINATES),
    'noise': modelling.GaussRV(C=CovMat(1e0 * errors_cov, kind = 'full')),
}

X0 = modelling.GaussRV(C=CovMat(1e0*errors_cov, kind = 'full'), mu=x0)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
HMM.liveplotters = live_plots(plot_marginals = PLOT_MARGINALS)
xp = da.PartFilt(N=1000, reg=1, NER=1, qroot = 1, wroot = 1.0)
xp.assimilate(HMM, xx[:], yy[:], liveplots=True)
xp.stats.average_in_time()
xp.stats.replay()