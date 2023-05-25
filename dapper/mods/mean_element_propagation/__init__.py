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

MU = 3.986004418e14
SECONDS_PER_DAY = 60 * 60 * 24

def convert_skyfield_earth_satellite_to_np_array(skyfield_earth_satellite,
                                                 use_keplerian_coordinates = True):

    np_mean_elements = np.zeros(6)

    if use_keplerian_coordinates:

        np_mean_elements[0] = skyfield_earth_satellite.model.nm
        np_mean_elements[1] = skyfield_earth_satellite.model.em
        np_mean_elements[2] = skyfield_earth_satellite.model.im
        np_mean_elements[3] = skyfield_earth_satellite.model.om
        np_mean_elements[4] = skyfield_earth_satellite.model.Om
        np_mean_elements[5] = skyfield_earth_satellite.model.mm

    else:

        a = (1 *
             (
                     (astropy.constants.GM_earth.value ** (1/3)) /
                     (
                             (skyfield_earth_satellite.model.nm / 60)**(2/3)
                     )
             )
             )

        orb = pyorb.Orbit(M0=pyorb.M_earth, degrees=True)
        orb.update(a = a,
                   e = skyfield_earth_satellite.model.em,
                   i = skyfield_earth_satellite.model.im * (180/np.pi),
                   omega = skyfield_earth_satellite.model.om * (180/np.pi),
                   Omega = skyfield_earth_satellite.model.Om * (180/np.pi),
                   anom = skyfield_earth_satellite.model.mm * (180/np.pi),
                   type = 'mean',
                   )

        np_mean_elements[0] = 1e-4 * orb.x
        np_mean_elements[1] = 1e-4 * orb.y
        np_mean_elements[2] = 1e-4 * orb.z
        np_mean_elements[3] = orb.vx
        np_mean_elements[4] = orb.vy
        np_mean_elements[5] = orb.vz

    return np_mean_elements


def propagation_initializer(tle_file_path, start_epoch):

    global list_of_skyfield_earth_satellites, first_lines_in_pairs, second_lines_in_pairs

    list_of_skyfield_earth_satellites = skyfield_load.tle_file(tle_file_path, reload=False)[start_epoch:]

    file = open(tle_file_path, 'r')
    for i in range(start_epoch):
        file.readline()
        file.readline()
    first_lines_in_pairs = []
    second_lines_in_pairs = []
    # TODO: loop this to end of file
    for i in range(1000):
        first_lines_in_pairs.append(file.readline())
        second_lines_in_pairs.append(file.readline())


def propagate_mean_elements(i, np_mean_elements, t, use_keplerian_coordinates = True):

    SkyfieldEarthSatellite_initial = list_of_skyfield_earth_satellites[t]
    SkyfieldEarthSatellite_post = list_of_skyfield_earth_satellites[t + 1]

    ts = skyfield_load.timescale()
    dummy_SkyfieldEarthSatellite = EarthSatellite(first_lines_in_pairs[t], second_lines_in_pairs[t], 'ISS (ZARYA)', ts)

    np_mean_elements = np_mean_elements[:, i]

    if use_keplerian_coordinates:
        satrec = Satrec()
        satrec.sgp4init(
            WGS84,  # gravity model
            'i',  # 'a' = old AFSPC mode, 'i' = improved mode
            SkyfieldEarthSatellite_initial.model.satnum,  # satnum: Satellite number
            # SkyfieldEarthSatellite_initial.epoch.tai - 2433281.5,  # epoch: days since 1949 December 31 00:00 UT
            SkyfieldEarthSatellite_initial.model.jdsatepoch - 2433281.5 + SkyfieldEarthSatellite_initial.model.jdsatepochF,
            SkyfieldEarthSatellite_initial.model.bstar,  # bstar: drag coefficient (/earth radii)
            SkyfieldEarthSatellite_initial.model.ndot,  # ndot: ballistic coefficient (revs/day)
            SkyfieldEarthSatellite_initial.model.nddot,  # nddot: second derivative of mean motion (revs/day^3)
            np_mean_elements[1],  # ecco: eccentricity
            np_mean_elements[3],  # argpo: argument of perigee (radians)
            np_mean_elements[2],  # inclo: inclination (radians)
            np_mean_elements[5],  # mo: mean anomaly (radians)
            np_mean_elements[0],  # no_kozai: mean motion (radians/minute)
            np_mean_elements[4],  # nodeo: right ascension of ascending node (radians)

        )

    else:

        orb = pyorb.Orbit(M0=pyorb.M_earth, degrees=True)

        orb.x = 1e4 * np_mean_elements[0]
        orb.y = 1e4 * np_mean_elements[1]
        orb.z = 1e4 * np_mean_elements[2]
        orb.vx = np_mean_elements[3]
        orb.vy = np_mean_elements[4]
        orb.vz = np_mean_elements[5]

        mean_motion = 60 * ((astropy.constants.GM_earth.value ** (1 / 3)) / orb.a[0]) ** (3 / 2)

        satrec = Satrec()
        satrec.sgp4init(
            WGS84,  # gravity model
            'i',  # 'a' = old AFSPC mode, 'i' = improved mode
            SkyfieldEarthSatellite_initial.model.satnum,  # satnum: Satellite number
            #SkyfieldEarthSatellite_initial.epoch.tai - 2433281.5,  # epoch: days since 1949 December 31 00:00 UT
            SkyfieldEarthSatellite_initial.model.jdsatepoch - 2433281.5 + SkyfieldEarthSatellite_initial.model.jdsatepochF,
            SkyfieldEarthSatellite_initial.model.bstar,  # bstar: drag coefficient (/earth radii)
            SkyfieldEarthSatellite_initial.model.ndot,  # ndot: ballistic coefficient (revs/day)
            SkyfieldEarthSatellite_initial.model.nddot,  # nddot: second derivative of mean motion (revs/day^3)

            orb.e[0] * (np.pi / 180),  # ecco: eccentricity
            orb.omega[0] * (np.pi / 180),  # argpo: argument of perigee (radians)
            orb.i[0] * (np.pi / 180),  # inclo: inclination (radians)
            orb.anom[0] * (np.pi / 180),  # mo: mean anomaly (radians)
            mean_motion,  # no_kozai: mean motion (radians/minute)
            orb.Omega[0] * (np.pi / 180),  # nodeo: right ascension of ascending node (radians)
        )

    sat = dummy_SkyfieldEarthSatellite
    sat.model = satrec

    sat.at(SkyfieldEarthSatellite_post.epoch)

    return convert_skyfield_earth_satellite_to_np_array(sat, use_keplerian_coordinates)


@modelling.ens_compatible
def step(x, t, dt, tle_file_path, start_epoch, use_keplerian_coordinates = True):

    pool = multiprocessing.Pool(10, initializer=propagation_initializer, initargs = (tle_file_path, start_epoch))

    out1 = pool.map(
        partial(propagate_mean_elements,
                np_mean_elements = x,
                t = t,
                use_keplerian_coordinates = use_keplerian_coordinates
                ),
        range(0, x.shape[1])
    )
    return np.transpose(np.array(out1))

params = dict(labels='012345')
def LPs(jj=None, params=params): return [
   (1, LP.sliding_marginals(obs_inds=tuple(np.arange(10)), zoomy=0.75, **params)),
]