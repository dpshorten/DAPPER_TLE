from skyfield.api import load as skyfield_load
import skyfield.elementslib as skyfield_ele
from skyfield.sgp4lib import EarthSatellite
import numpy as np
import dapper.mods as modelling
import pandas as pd
import copy
import dapper.tools.liveplotting as LP
import pyorb
import astropy.constants
import multiprocessing
from functools import partial
import sys

sys.path.insert(0, "/home/david/Projects/python-sgp4/sgp4/")
from api import Satrec, WGS72, WGS72OLD, WGS84, jday
from propagation import sgp4init, sgp4
from model import twoline2rv
from earth_gravity import wgs84

MINUTES_PER_DAY = 24 * 60


def convert_keplerian_coordinates_to_cartesian(keplerian_state_vector):

    brouwer_mean_motion = keplerian_state_vector[0]
    eccentricity = keplerian_state_vector[1]
    inclination = keplerian_state_vector[2]
    arg_perigee = keplerian_state_vector[3]
    right_ascension = keplerian_state_vector[4]
    mean_anomaly = keplerian_state_vector[5]

    a = (
             (
                     astropy.constants.GM_earth.value ** (1/3)
             ) /
             (
                     (brouwer_mean_motion / 60) ** (2 / 3)
             )
        )

    orb = pyorb.Orbit(M0 = pyorb.M_earth, degrees = True)
    orb.update(a = a,
               e = eccentricity,
               i = inclination * (180 / np.pi),
               omega = arg_perigee * (180 / np.pi),
               Omega = right_ascension * (180 / np.pi),
               anom = mean_anomaly * (180 / np.pi),
               type = 'mean',
               )
    cartesian_state_vector = np.zeros(6)
    cartesian_state_vector[0] = 1e-4 * orb.x
    cartesian_state_vector[1] = 1e-4 * orb.y
    cartesian_state_vector[2] = 1e-4 * orb.z
    cartesian_state_vector[3] = orb.vx
    cartesian_state_vector[4] = orb.vy
    cartesian_state_vector[5] = orb.vz

    return cartesian_state_vector

def convert_cartesian_coordinates_to_keplerian(cartesian_state_vector):

    orb = pyorb.Orbit(M0=pyorb.M_earth, degrees=True)

    orb.x = 1e4 * cartesian_state_vector[0]
    orb.y = 1e4 * cartesian_state_vector[1]
    orb.z = 1e4 * cartesian_state_vector[2]
    orb.vx = cartesian_state_vector[3]
    orb.vy = cartesian_state_vector[4]
    orb.vz = cartesian_state_vector[5]

    mean_motion = 60 * (((astropy.constants.GM_earth.value ** (1 / 3)) / orb.a[0]) ** (3 / 2))


    orb.e[0],  # ecco: eccentricity
    orb.omega[0] * (np.pi / 180),  # argpo: argument of perigee (radians)
    orb.i[0] * (np.pi / 180),  # inclo: inclination (radians)
    orb.anom[0] * (np.pi / 180),  # mo: mean anomaly (radians)
    mean_motion,  # no_kozai: mean motion (radians/minute)
    orb.Omega[0] * (np.pi / 180),  # nodeo: right ascension of ascending node (radians)

    keplerian_state_vector = np.zeros(6)
    keplerian_state_vector[0] = mean_motion
    keplerian_state_vector[1] = orb.e[0]
    keplerian_state_vector[2] = orb.i[0] * (np.pi / 180)
    keplerian_state_vector[3] = orb.omega[0] * (np.pi / 180)
    keplerian_state_vector[4] = orb.Omega[0] * (np.pi / 180)
    keplerian_state_vector[5] = orb.anom[0] * (np.pi / 180)

    return keplerian_state_vector
def convert_Skyfield_EarthSatellite_to_np_array(skyfield_EarthSatellite,
                                                use_keplerian_coordinates = True):

    np_mean_elements = np.zeros(6)

    # Convert the mean motion from Kozai formulation to Brouwer formulation
    eccsq = skyfield_EarthSatellite.model.ecco * skyfield_EarthSatellite.model.ecco
    omeosq = 1.0 - eccsq
    rteosq = np.sqrt(omeosq)
    cosio2 = np.cos(skyfield_EarthSatellite.model.inclo) ** 2
    ak = np.power(skyfield_EarthSatellite.model.xke / skyfield_EarthSatellite.model.no_kozai, 2 / 3.0)
    d1 = 0.75 * skyfield_EarthSatellite.model.j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq)
    del_ = d1 / (ak * ak)
    adel = ak * (1.0 - del_ * del_ - del_ *
                 (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0))
    del_ = d1 / (adel * adel)
    brouwer_mean_motion = skyfield_EarthSatellite.model.no_kozai / (1.0 + del_)

    np_mean_elements[0] = brouwer_mean_motion
    #np_mean_elements[0] = skyfield_EarthSatellite.model.nm  # eccentricity
    np_mean_elements[1] = skyfield_EarthSatellite.model.em  # eccentricity
    np_mean_elements[2] = skyfield_EarthSatellite.model.im # inclination
    np_mean_elements[3] = skyfield_EarthSatellite.model.om # argument of perigee
    np_mean_elements[4] = skyfield_EarthSatellite.model.Om # right ascension
    np_mean_elements[5] = skyfield_EarthSatellite.model.mm # mean anomaly

    if use_keplerian_coordinates:
        return np_mean_elements
    else:
        return convert_keplerian_coordinates_to_cartesian(np_mean_elements)

def propagate_mean_elements(particle_index,
                            np_mean_elements_of_particles,
                            TLE_line_pair_initial,
                            TLE_line_pair_post,
                            use_keplerian_coordinates = True):

    #ts = skyfield_load.timescale()

    #Skyfield_EarthSatellite_initial = EarthSatellite(TLE_line_pair_initial[0], TLE_line_pair_initial[1], '', ts)
    #Skyfield_EarthSatellite_post = EarthSatellite(TLE_line_pair_post[0], TLE_line_pair_post[1], '', ts)

    np_particle_mean_elements = np_mean_elements_of_particles[:, particle_index]

    if not use_keplerian_coordinates:
        np_particle_mean_elements = convert_cartesian_coordinates_to_keplerian(np_particle_mean_elements)

    satrec_initial = twoline2rv(TLE_line_pair_initial[0], TLE_line_pair_initial[1], wgs84)
    satrec_post = twoline2rv(TLE_line_pair_post[0], TLE_line_pair_post[1], wgs84)

    sgp4init(wgs84, 'i', satrec_initial.satnum,
             satrec_initial.jdsatepoch + satrec_initial.jdsatepochF - 2433281.5,
             #satrec_initial.jdsatepoch + satrec_initial.jdsatepochF,
             satrec_initial.bstar,
             satrec_initial.ndot,
             satrec_initial.nddot,
             np_particle_mean_elements[1],
             np_particle_mean_elements[3],
             np_particle_mean_elements[2],
             np_particle_mean_elements[5],
             np_particle_mean_elements[0],
             np_particle_mean_elements[4],
             satrec_initial)

    # sgp4init(wgs84, 'i', satrec_initial.satnum,
    #          #satrec_initial.jdsatepoch + satrec_initial.jdsatepochF - 2433281.5,
    #          satrec_initial.jdsatepoch + satrec_initial.jdsatepochF,
    #          satrec_initial.bstar,
    #          satrec_initial.ndot,
    #          satrec_initial.nddot,
    #          satrec_initial.ecco,
    #          satrec_initial.argpo,
    #          satrec_initial.inclo,
    #          satrec_initial.mo,
    #          satrec_initial.no_kozai,
    #          satrec_initial.nodeo,
    #          satrec_initial)

    tsince = ((satrec_post.jdsatepoch - satrec_initial.jdsatepoch) * MINUTES_PER_DAY +
              (satrec_post.jdsatepochF - satrec_initial.jdsatepochF) * MINUTES_PER_DAY)

    sgp4(satrec_initial, tsince)
    #sgp4(satrec_initial, 0)

    np_propagated_mean_elements = np.zeros(6)
    np_propagated_mean_elements[0] = satrec_initial.nm  # mean motion
    np_propagated_mean_elements[1] = satrec_initial.em  # eccentricity
    np_propagated_mean_elements[2] = satrec_initial.im  # inclination
    np_propagated_mean_elements[3] = satrec_initial.om  # argument of perigee
    np_propagated_mean_elements[4] = satrec_initial.Om  # right ascension
    np_propagated_mean_elements[5] = satrec_initial.mm  # mean anomaly

    if use_keplerian_coordinates:
        return np_propagated_mean_elements
    else:
        return convert_keplerian_coordinates_to_cartesian(np_propagated_mean_elements)

@modelling.ens_compatible
def step(x, t, dt, process_pool, list_of_TLE_line_pairs, use_keplerian_coordinates = True):

    propagated_particles = process_pool.map(
        partial(propagate_mean_elements,
                np_mean_elements_of_particles = x,
                TLE_line_pair_initial = list_of_TLE_line_pairs[t],
                TLE_line_pair_post = list_of_TLE_line_pairs[t + dt],
                use_keplerian_coordinates = use_keplerian_coordinates
                ),
        range(0, x.shape[1])
    )

    return np.transpose(np.array(propagated_particles))


def live_plots(plot_marginals = False, params = dict(labels='012345', ens_props = dict(alpha = 0.1))):
    """
    Sets up the live plotting functionality for Dapper.
    """

    if plot_marginals:
        return [(1, LP.sliding_marginals(obs_inds=tuple(np.arange(10)), zoomy=0.75, **params)),]
    else:
        return []