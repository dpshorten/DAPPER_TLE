import numpy as np
import dapper.mods as modelling
import copy
import dapper.tools.liveplotting as LP
import astropy.constants
import multiprocessing
from functools import partial
import sys

from python_parameters import base_package_load_path
sys.path.insert(0, base_package_load_path)
#sys.path.insert(0, "/home/david/Projects/python-sgp4/sgp4/")
from sgp4.api import Satrec
from sgp4.propagation import sgp4init, sgp4
from sgp4.model import twoline2rv
from sgp4.earth_gravity import wgs84

from TLE_utilities.tle_loading_and_preprocessing import (propagate_np_mean_elements,
                                                         load_tle_data_from_file,
                                                         convert_np_keplerian_coordinates_to_cartesian,
                                                         convert_np_cartesian_coordinates_to_keplerian,)


MINUTES_PER_DAY = 24 * 60
JULIAN_DATE_OF_31_12_1949 = 2433281.5
def propagate_mean_elements(particle_index,
                            np_mean_elements_of_particles,
                            tle_line_pair_initial,
                            tle_line_pair_post,
                            normalisation_weights,
                            np_propagation_corrections = np.empty(0),
                            element_set = "kepler_6"):

    np_particle_mean_elements = np_mean_elements_of_particles[:, particle_index]
    np_particle_mean_elements = np.multiply(np_particle_mean_elements, normalisation_weights)

    #print("prior", np_particle_mean_elements)
    np_propagated_mean_elements = propagate_np_mean_elements(np_particle_mean_elements,
                                                             tle_line_pair_initial,
                                                             tle_line_pair_post,
                                                             element_set=element_set)

    #print(np_propagation_corrections[particle_index, :])
    #quit()
    if np_propagation_corrections.shape[0] > 0:
        np_propagated_mean_elements -= np_propagation_corrections[particle_index, :]

    return np.divide(
        np_propagated_mean_elements,
        normalisation_weights
    )

@modelling.ens_compatible
def step(x, t, dt, process_pool, satelliteTLEData_object, normalisation_weights, correction_model=None, element_set="kepler_6"):

    if correction_model:
        unnormalised_x = np.zeros((x.shape[1], x.shape[0]))
        #print("shape", unnormalised_x.shape)
        for j in range(x.shape[1]):
            unnormalised_x[j, :] = np.multiply(x[:, j], normalisation_weights)
        np_propagation_corrections = correction_model.predict(unnormalised_x)
    else:
        np_propagation_corrections = np.zeros((x.shape[1], x.shape[0]))

    propagated_particles = process_pool.map(
        partial(propagate_mean_elements,
                np_mean_elements_of_particles = x,
                tle_line_pair_initial = satelliteTLEData_object.list_of_tle_line_tuples[t],
                tle_line_pair_post = satelliteTLEData_object.list_of_tle_line_tuples[t+dt],
                normalisation_weights = normalisation_weights,
                np_propagation_corrections = np_propagation_corrections,
                element_set = element_set
                ),
        range(0, x.shape[1])
    )

    return np.transpose(np.array(propagated_particles))


def live_plots(plot_marginals = False, use_keplerian_coordinates = True, params = dict(), element_names = []):
    """
    Sets up the live plotting functionality for Dapper.
    """

    if plot_marginals:
        return [(1, LP.sliding_marginals(obs_inds=tuple(np.arange(10)),
                                         zoomy = 0.75,
                                         ens_props = dict(alpha = 0.1),
                                         labels = element_names,
                                         **params)),]
    else:
        return []