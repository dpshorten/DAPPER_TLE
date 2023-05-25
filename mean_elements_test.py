import sys
import os
import numpy as np
import dapper.mods as modelling
from skyfield.api import load as skyfield_load
from dapper.mods.mean_element_propagation import step, LPs, convert_skyfield_earth_satellite_to_np_array
import pandas as pd
import dapper.da_methods as da
from functools import partial
from dapper.tools.matrices import CovMat
import scipy.linalg as sla

TLE_FILES_DIRECTORY = "/home/david/Projects/TLE_observation_benchmark_dataset/processed_files/"
TLE_FILE = "Fengyun-2D.tle"
START_EPOCH = 100
USE_KEPLERIAN_COORDINATES = True

Tplot = 20
tseq = modelling.Chronology(1, dko=1, Ko=1000, Tplot=Tplot, BurnIn=4*Tplot)

list_of_skyfield_earth_satellites = skyfield_load.tle_file(TLE_FILES_DIRECTORY + TLE_FILE, reload=False)[START_EPOCH:]

satellite_epoch_times = []
xx = []
for sat in list_of_skyfield_earth_satellites:
    pd_timestamp_sat_epoch = pd.Timestamp(sat.epoch.utc_datetime())
    satellite_epoch_times.append(pd_timestamp_sat_epoch)
    xx.append(convert_skyfield_earth_satellite_to_np_array(sat, use_keplerian_coordinates=USE_KEPLERIAN_COORDINATES))

xx = np.array(xx)
yy = np.copy(xx)

x0 = convert_skyfield_earth_satellite_to_np_array(list_of_skyfield_earth_satellites[0], use_keplerian_coordinates=USE_KEPLERIAN_COORDINATES)
Nx = len(x0)

jj = np.arange(Nx)  # obs_inds
Obs = modelling.partial_Id_Obs(Nx, jj)

props = np.zeros((xx.shape[0]-1, xx.shape[1]))
for i in range(200):
    props[i, :] = np.squeeze(step(np.expand_dims(xx[i], axis = 0), i, 1,
                                  TLE_FILES_DIRECTORY + TLE_FILE,
                                  START_EPOCH,
                                  USE_KEPLERIAN_COORDINATES), axis = 0)
errors = props[:200] - xx[1:201, :]
for i in range(10):
    print(errors[i, :])
errors_cov = np.cov(errors, rowvar = False)

Obs['noise'] = modelling.GaussRV(C=CovMat(1e0 * errors_cov, kind = 'full'))

Dyn = {
    'M': Nx,
    'model': partial(step,
                     tle_file_path = TLE_FILES_DIRECTORY + TLE_FILE,
                     start_epoch = START_EPOCH,
                     use_keplerian_coordinates = USE_KEPLERIAN_COORDINATES),
    'noise': modelling.GaussRV(C=CovMat(1e0 * errors_cov, kind = 'full')),
}

X0 = modelling.GaussRV(C=CovMat(1e0*errors_cov, kind = 'full'), mu=x0)

HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
HMM.liveplotters = LPs(jj = jj[::10])
xp = da.PartFilt(N=1000, reg=1, NER=1, qroot = 1, wroot = 1.0)
xp.assimilate(HMM, xx[:], yy[:], liveplots=True)
xp.stats.average_in_time()
xp.stats.replay()


# import dapper.tools.viz as viz
# viz.plot_rank_histogram(xp.stats)
# viz.plot_err_components(xp.stats)
# viz.plot_hovmoller(xx)