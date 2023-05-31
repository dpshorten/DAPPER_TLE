from skyfield.api import load as skyfield_load
import skyfield.elementslib as skyfield_ele
from skyfield.sgp4lib import EarthSatellite
from sgp4.api import Satrec, WGS72, WGS72OLD, WGS84, jday
import numpy as np
import dapper.mods as modelling
import pandas as pd
import copy
import dapper.tools.liveplotting as LP
import pyorb
import astropy.constants
import multiprocessing
from functools import partial

def convert_Skyfield_EarthSatellite_to_np_array(skyfield_EarthSatellite,
                                                use_keplerian_coordinates = True):

    np_mean_elements = np.zeros(6)

    if use_keplerian_coordinates:

        np_mean_elements[0] = skyfield_EarthSatellite.model.nm
        np_mean_elements[1] = skyfield_EarthSatellite.model.em
        np_mean_elements[2] = skyfield_EarthSatellite.model.im
        np_mean_elements[3] = skyfield_EarthSatellite.model.om
        np_mean_elements[4] = skyfield_EarthSatellite.model.Om
        np_mean_elements[5] = skyfield_EarthSatellite.model.mm

    else:

        a = (
                 (
                         astropy.constants.GM_earth.value ** (1/3)
                 ) /
                 (
                         (skyfield_EarthSatellite.model.nm / 60) ** (2 / 3)
                 )
            )

        orb = pyorb.Orbit(M0 = pyorb.M_earth, degrees = True)
        orb.update(a = a,
                   e = skyfield_EarthSatellite.model.em,
                   i =skyfield_EarthSatellite.model.im * (180 / np.pi),
                   omega =skyfield_EarthSatellite.model.om * (180 / np.pi),
                   Omega =skyfield_EarthSatellite.model.Om * (180 / np.pi),
                   anom =skyfield_EarthSatellite.model.mm * (180 / np.pi),
                   type = 'mean',
                   )

        np_mean_elements[0] = 1e-4 * orb.x
        np_mean_elements[1] = 1e-4 * orb.y
        np_mean_elements[2] = 1e-4 * orb.z
        np_mean_elements[3] = orb.vx
        np_mean_elements[4] = orb.vy
        np_mean_elements[5] = orb.vz

    return np_mean_elements



def propagate_mean_elements(particle_index,
                            np_mean_elements_of_particles,
                            TLE_line_pair_initial,
                            TLE_line_pair_post,
                            use_keplerian_coordinates = True):

    ts = skyfield_load.timescale()
    Skyfield_EarthSatellite_initial = EarthSatellite(TLE_line_pair_initial[0], TLE_line_pair_initial[1], '', ts)
    Skyfield_EarthSatellite_post = EarthSatellite(TLE_line_pair_post[0], TLE_line_pair_post[1], '', ts)

    np_particle_mean_elements = np_mean_elements_of_particles[:, particle_index]

    if use_keplerian_coordinates:
        satrec = Satrec()
        satrec.sgp4init(
            WGS84,  # gravity model
            'i',  # 'a' = old AFSPC mode, 'i' = improved mode
            Skyfield_EarthSatellite_initial.model.satnum,  # satnum: Satellite number
            # Skyfield_EarthSatellite_initial.epoch.tai - 2433281.5,  # epoch: days since 1949 December 31 00:00 UT
            Skyfield_EarthSatellite_initial.model.jdsatepoch - 2433281.5 + Skyfield_EarthSatellite_initial.model.jdsatepochF,
            Skyfield_EarthSatellite_initial.model.bstar,  # bstar: drag coefficient (/earth radii)
            Skyfield_EarthSatellite_initial.model.ndot,  # ndot: ballistic coefficient (revs/day)
            Skyfield_EarthSatellite_initial.model.nddot,  # nddot: second derivative of mean motion (revs/day^3)
            np_particle_mean_elements[1],  # ecco: eccentricity
            np_particle_mean_elements[3],  # argpo: argument of perigee (radians)
            np_particle_mean_elements[2],  # inclo: inclination (radians)
            np_particle_mean_elements[5],  # mo: mean anomaly (radians)
            np_particle_mean_elements[0],  # no_kozai: mean motion (radians/minute)
            np_particle_mean_elements[4],  # nodeo: right ascension of ascending node (radians)
        )

    else:

        orb = pyorb.Orbit(M0=pyorb.M_earth, degrees=True)

        orb.x = 1e4 * np_particle_mean_elements[0]
        orb.y = 1e4 * np_particle_mean_elements[1]
        orb.z = 1e4 * np_particle_mean_elements[2]
        orb.vx = np_particle_mean_elements[3]
        orb.vy = np_particle_mean_elements[4]
        orb.vz = np_particle_mean_elements[5]

        mean_motion = 60 * (((astropy.constants.GM_earth.value ** (1 / 3)) / orb.a[0]) ** (3 / 2))

        satrec = Satrec()
        satrec.sgp4init(
            WGS84,  # gravity model
            'i',  # 'a' = old AFSPC mode, 'i' = improved mode
            Skyfield_EarthSatellite_initial.model.satnum,  # satnum: Satellite number
            #Skyfield_EarthSatellite_initial.epoch.tai - 2433281.5,  # epoch: days since 1949 December 31 00:00 UT
            Skyfield_EarthSatellite_initial.model.jdsatepoch - 2433281.5 + Skyfield_EarthSatellite_initial.model.jdsatepochF,
            Skyfield_EarthSatellite_initial.model.bstar,  # bstar: drag coefficient (/earth radii)
            Skyfield_EarthSatellite_initial.model.ndot,  # ndot: ballistic coefficient (revs/day)
            Skyfield_EarthSatellite_initial.model.nddot,  # nddot: second derivative of mean motion (revs/day^3)

            orb.e[0],  # ecco: eccentricity
            orb.omega[0] * (np.pi / 180),  # argpo: argument of perigee (radians)
            orb.i[0] * (np.pi / 180),  # inclo: inclination (radians)
            orb.anom[0] * (np.pi / 180),  # mo: mean anomaly (radians)
            mean_motion,  # no_kozai: mean motion (radians/minute)
            orb.Omega[0] * (np.pi / 180),  # nodeo: right ascension of ascending node (radians)
        )

    Skyfield_EarthSatellite_initial.model = satrec
    Skyfield_EarthSatellite_initial.at(Skyfield_EarthSatellite_post.epoch)

    return convert_Skyfield_EarthSatellite_to_np_array(Skyfield_EarthSatellite_initial, use_keplerian_coordinates)


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


def live_plots(plot_marginals = False, params = dict(labels='012345')):
    """
    Sets up the live plotting functionality for Dapper.
    """

    if plot_marginals:
        return [(1, LP.sliding_marginals(obs_inds=tuple(np.arange(10)), zoomy=0.75, **params)),]
    else:
        return []