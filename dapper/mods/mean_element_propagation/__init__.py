import numpy as np
import copy
import astropy.constants
import multiprocessing
from functools import partial
import sys
import os
import dapper.tools.liveplotting as LP
import dapper.mods as modelling

this_directory = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.dirname(this_directory + "/../../../../TLE_utilities"))
from TLE_utilities.tle_loading_and_preprocessing import (propagate_np_mean_elements,
                                                         load_tle_data_from_file,
                                                         convert_np_keplerian_coordinates_to_cartesian,
                                                         convert_np_cartesian_coordinates_to_keplerian,)

def propagate_mean_elements(particle_index,
                            np_mean_elements_of_particles,
                            tle_line_pair_initial,
                            tle_line_pair_post,
                            normalisation_weights,
                            element_set = "kepler_6"):

    np_particle_mean_elements = np_mean_elements_of_particles[:, particle_index]
    np_particle_mean_elements = np.multiply(np_particle_mean_elements, normalisation_weights)

    np_propagated_mean_elements = propagate_np_mean_elements(np_particle_mean_elements,
                                                             tle_line_pair_initial,
                                                             tle_line_pair_post,
                                                             element_set=element_set)

    return np.divide(
        np_propagated_mean_elements,
        normalisation_weights
    )

@modelling.ens_compatible
def step(x, t, dt, process_pool, satelliteTLEData_object, normalisation_weights, element_set="kepler_6"):

    propagated_particles = process_pool.map(
        partial(propagate_mean_elements,
                np_mean_elements_of_particles = x,
                tle_line_pair_initial = satelliteTLEData_object.list_of_tle_line_tuples[t],
                tle_line_pair_post = satelliteTLEData_object.list_of_tle_line_tuples[t+dt],
                normalisation_weights = normalisation_weights,
                element_set = element_set
                ),
        range(0, x.shape[1])
    )

    return np.transpose(np.array(propagated_particles))


def live_plots(plot_marginals = False, params = dict(), element_names = []):
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