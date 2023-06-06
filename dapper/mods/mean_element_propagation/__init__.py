from skyfield.api import load as skyfield_load
import skyfield.elementslib as skyfield_ele
from skyfield.sgp4lib import EarthSatellite
from sgp4.api import Satrec, WGS72, WGS72OLD, WGS84, jday
from sgp4.propagation import sgp4init
from sgp4.model import twoline2rv
from sgp4.earth_gravity import wgs84
import numpy as np
import dapper.mods as modelling
import pandas as pd
import copy
import dapper.tools.liveplotting as LP
import pyorb
import astropy.constants
import multiprocessing
from functools import partial

MINUTES_PER_DAY = 24 * 60

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


    if use_keplerian_coordinates:

        np_mean_elements[0] = brouwer_mean_motion
        #np_mean_elements[0] = skyfield_EarthSatellite.model.nm  # eccentricity
        np_mean_elements[1] = skyfield_EarthSatellite.model.em  # eccentricity
        np_mean_elements[2] = skyfield_EarthSatellite.model.im # inclination
        np_mean_elements[3] = skyfield_EarthSatellite.model.om # argument of perigee
        np_mean_elements[4] = skyfield_EarthSatellite.model.Om # right ascension
        np_mean_elements[5] = skyfield_EarthSatellite.model.mm # mean anomaly

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

    #ts = skyfield_load.timescale()

    #Skyfield_EarthSatellite_initial = EarthSatellite(TLE_line_pair_initial[0], TLE_line_pair_initial[1], '', ts)
    #Skyfield_EarthSatellite_post = EarthSatellite(TLE_line_pair_post[0], TLE_line_pair_post[1], '', ts)

    np_particle_mean_elements = np_mean_elements_of_particles[:, particle_index]

    satrec_initial = twoline2rv(TLE_line_pair_initial[0], TLE_line_pair_initial[1], wgs84)
    satrec_post = twoline2rv(TLE_line_pair_post[0], TLE_line_pair_post[1], wgs84)

    # sgp4init(wgs84, 'i', satrec_initial.satnum,
    #                         satrec_initial.jdsatepoch,
    #                         satrec_initial.bstar,
    #                         satrec_initial.ndot,
    #                         satrec_initial.nddot,
    #                         satrec_initial.ecco,
    #                         satrec_initial.argpo,
    #                         satrec_initial.inclo,
    #                         satrec_initial.mo,
    #                         satrec_initial.no_kozai,
    #                         satrec_initial.nodeo,
    #                         satrec_initial)

    no_unkozai =  np_particle_mean_elements[0] # mean motion
    ecco = np_particle_mean_elements[1]  # eccentricity
    inclo = np_particle_mean_elements[2]  # inclination
    #np_propagated_mean_elements[3] = argpm  # argument of perigee
    #np_propagated_mean_elements[4] = nodem  # right ascension
    #np_propagated_mean_elements[5] = mm  # mean anomaly

    deg2rad = np.pi / 180.0
    twopi = 2 * np.pi

    (tumin,
     mu,
     radiusearthkm,
     xke,
     j2,
     j3,
     j4,
     j3oj2) = wgs84

    ss = 78.0 / radiusearthkm + 1.0
    qzms2ttemp = (120.0 - 78.0) / radiusearthkm
    qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp
    x2o3 = 2.0 / 3.0

    eccsq = ecco * ecco
    omeosq = 1.0 - eccsq
    rteosq = np.sqrt(omeosq)
    cosio = np.cos(inclo)
    cosio2 = cosio * cosio

    ak = np.power(xke / no_unkozai, x2o3)
    d1 = 0.75 * satrec_initial.j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq)
    del_ = d1 / (ak * ak)
    adel = ak * (1.0 - del_ * del_ - del_ *
                     (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0))
    del_ = d1 / (adel * adel)
    #no = no / (1.0 + del_)

    ao = np.power(xke / no_unkozai, x2o3)
    sinio = np.sin(inclo)
    po = ao * omeosq
    con42 = 1.0 - 5.0 * cosio2
    con41 = -con42 - cosio2 - cosio2
    ainv = 1.0 / ao
    posq = po * po
    rp = ao * (1.0 - ecco)

    tut1 = (satrec_initial.jdsatepoch + 2433281.5 - 2451545.0) / 36525.0;
    temp = -6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 + \
           (876600.0 * 3600 + 8640184.812866) * tut1 + 67310.54841;  # sec
    temp = (temp * deg2rad / 240.0) % twopi  # 360/86400 = 1/240, to deg, to rad

    if temp < 0.0:
        temp += twopi;

    gsto = temp

    a = pow(no_unkozai * tumin, (-2.0 / 3.0));
    alta = a * (1.0 + ecco) - 1.0;
    altp = a * (1.0 - ecco) - 1.0;

    print(altp)
    quit()

    satrec_initial.t = ((satrec_post.jdsatepoch - satrec_initial.jdsatepoch) * MINUTES_PER_DAY +
              (satrec_post.jdsatepochF - satrec_initial.jdsatepochF) * MINUTES_PER_DAY)


    xmdf = satrec_initial.mo + satrec_initial.mdot * satrec_initial.t
    argpdf = satrec_initial.argpo + satrec_initial.argpdot * satrec_initial.t
    nodedf = satrec_initial.nodeo + satrec_initial.nodedot * satrec_initial.t
    argpm = argpdf
    mm = xmdf
    t2 = satrec_initial.t * satrec_initial.t
    nodem = nodedf + satrec_initial.nodecf * t2
    tempa = 1.0 - satrec_initial.cc1 * satrec_initial.t
    tempe = satrec_initial.bstar * satrec_initial.cc4 * satrec_initial.t
    templ = satrec_initial.t2cof * t2

    if omeosq >= 0.0 or satrec.no_unkozai >= 0.0:

        satrec.isimp = 0;
        if rp < 220.0 / satrec.radiusearthkm + 1.0:
            satrec.isimp = 1;
        sfour = ss;
        qzms24 = qzms2t;
        perige = (rp - 1.0) * satrec.radiusearthkm;

        #  - for perigees below 156 km, s and qoms2t are altered -
        if perige < 156.0:

            sfour = perige - 78.0;
            if perige < 98.0:
                sfour = 20.0;
            #  sgp4fix use multiply for speed instead of pow
            qzms24temp = (120.0 - sfour) / satrec.radiusearthkm;
            qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp;
            sfour = sfour / satrec.radiusearthkm + 1.0;

        pinvsq = 1.0 / posq;

        tsi = 1.0 / (ao - sfour);
        satrec.eta = ao * satrec.ecco * tsi;
        etasq = satrec.eta * satrec.eta;
        eeta = satrec.ecco * satrec.eta;
        psisq = fabs(1.0 - etasq);
        coef = qzms24 * pow(tsi, 4.0);
        coef1 = coef / pow(psisq, 3.5);
        cc2 = coef1 * satrec.no_unkozai * (ao * (1.0 + 1.5 * etasq + eeta *
                                                 (4.0 + etasq)) + 0.375 * satrec.j2 * tsi / psisq * satrec.con41 *
                                           (8.0 + 3.0 * etasq * (8.0 + etasq)));
        satrec.cc1 = satrec.bstar * cc2;
        cc3 = 0.0;
        if satrec.ecco > 1.0e-4:
            cc3 = -2.0 * coef * tsi * satrec.j3oj2 * satrec.no_unkozai * sinio / satrec.ecco;
        satrec.x1mth2 = 1.0 - cosio2;
        satrec.cc4 = 2.0 * satrec.no_unkozai * coef1 * ao * omeosq * \
                     (satrec.eta * (2.0 + 0.5 * etasq) + satrec.ecco *
                      (0.5 + 2.0 * etasq) - satrec.j2 * tsi / (ao * psisq) *
                      (-3.0 * satrec.con41 * (1.0 - 2.0 * eeta + etasq *
                                              (1.5 - 0.5 * eeta)) + 0.75 * satrec.x1mth2 *
                       (2.0 * etasq - eeta * (1.0 + etasq)) * cos(2.0 * satrec.argpo)));
        satrec.cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 *
                                                  (etasq + eeta) + eeta * etasq);
        cosio4 = cosio2 * cosio2;
        temp1 = 1.5 * satrec.j2 * pinvsq * satrec.no_unkozai;
        temp2 = 0.5 * temp1 * satrec.j2 * pinvsq;
        temp3 = -0.46875 * satrec.j4 * pinvsq * pinvsq * satrec.no_unkozai;
        satrec.mdot = satrec.no_unkozai + 0.5 * temp1 * rteosq * satrec.con41 + 0.0625 * \
                      temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4);
        satrec.argpdot = (-0.5 * temp1 * con42 + 0.0625 * temp2 *
                          (7.0 - 114.0 * cosio2 + 395.0 * cosio4) +
                          temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4));
        xhdot1 = -temp1 * cosio;
        satrec.nodedot = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * cosio2) +
                                   2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio;
        xpidot = satrec.argpdot + satrec.nodedot;
        satrec.omgcof = satrec.bstar * cc3 * cos(satrec.argpo);
        satrec.xmcof = 0.0;
        if satrec.ecco > 1.0e-4:
            satrec.xmcof = -x2o3 * coef * satrec.bstar / eeta;
        satrec.nodecf = 3.5 * omeosq * xhdot1 * satrec.cc1;
        satrec.t2cof = 1.5 * satrec.cc1;
        #  sgp4fix for divide by zero with xinco = 180 deg
        if fabs(cosio + 1.0) > 1.5e-12:
            satrec.xlcof = -0.25 * satrec.j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio);
        else:
            satrec.xlcof = -0.25 * satrec.j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4;
        satrec.aycof = -0.5 * satrec.j3oj2 * sinio;
        #  sgp4fix use multiply for speed instead of pow
        delmotemp = 1.0 + satrec.eta * cos(satrec.mo);
        satrec.delmo = delmotemp * delmotemp * delmotemp;
        satrec.sinmao = sin(satrec.mo);
        satrec.x7thm1 = 7.0 * cosio2 - 1.0;

    #if satrec_initial.isimp != 1:
    if True:

        delomg = satrec_initial.omgcof * satrec_initial.t
        #  sgp4fix use mutliply for speed instead of pow
        delmtemp = 1.0 + satrec_initial.eta * np.cos(xmdf)
        delm = satrec_initial.xmcof * \
               (delmtemp * delmtemp * delmtemp -
                satrec_initial.delmo)
        temp = delomg + delm
        mm = xmdf + temp
        argpm = argpdf - temp
        t3 = t2 * satrec_initial.t
        t4 = t3 * satrec_initial.t
        tempa = tempa - satrec_initial.d2 * t2 - satrec_initial.d3 * t3 - \
                satrec_initial.d4 * t4
        tempe = tempe + satrec_initial.bstar * satrec_initial.cc5 * (np.sin(mm) -
                                                     satrec_initial.sinmao)
        templ = templ + satrec_initial.t3cof * t3 + t4 * (satrec_initial.t4cof +
                                                  satrec_initial.t * satrec_initial.t5cof)

    nm = np_particle_mean_elements[0]
    em = np_particle_mean_elements[1]
    inclm = np_particle_mean_elements[2]

    am = pow((satrec_initial.xke / nm), 2 / 3.0) * tempa * tempa;
    nm = satrec_initial.xke / pow(am, 1.5);
    em = em - tempe;

    #  fix tolerance for error recognition
    #  sgp4fix am is fixed from the previous nm check
    if em >= 1.0 or em < -0.001:  # || (am < 0.95)

        satrec_initial.error_message = ('mean eccentricity {0:f} not within'
                                ' range 0.0 <= e < 1.0'.format(em))
        satrec_initial.error = 1;
        #  sgp4fix to return if there is an error in eccentricity
        return false, false;

    #  sgp4fix fix tolerance to avoid a divide by zero
    if em < 1.0e-6:
        em = 1.0e-6;
    mm = mm + np_particle_mean_elements[0] * templ;
    xlm = mm + argpm + nodem;
    emsq = em * em;
    temp = 1.0 - emsq;

    #nodem = nodem % 2 * np.pi if nodem >= 0.0 else -(-nodem % 2 * np.pi)
    argpm = argpm % 2 * np.pi
    xlm = xlm % 2 * np.pi
    mm = (xlm - argpm - nodem) % 2 * np.pi

    np_propagated_mean_elements = np.zeros(6)
    np_propagated_mean_elements[0] = nm
    np_propagated_mean_elements[1] = em  # eccentricity
    np_propagated_mean_elements[2] = inclm # inclination
    np_propagated_mean_elements[3] = argpm  # argument of perigee
    np_propagated_mean_elements[4] = nodem  # right ascension
    np_propagated_mean_elements[5] = mm  # mean anomaly

    return np_propagated_mean_elements

    # if use_keplerian_coordinates:
    #     satrec = Satrec()
    #     satrec.sgp4init(
    #         WGS84,  # gravity model
    #         'i',  # 'a' = old AFSPC mode, 'i' = improved mode
    #         Skyfield_EarthSatellite_initial.model.satnum,  # satnum: Satellite number
    #         # Skyfield_EarthSatellite_initial.epoch.tai - 2433281.5,  # epoch: days since 1949 December 31 00:00 UT
    #         Skyfield_EarthSatellite_initial.model.jdsatepoch - 2433281.5 + Skyfield_EarthSatellite_initial.model.jdsatepochF,
    #         Skyfield_EarthSatellite_initial.model.bstar,  # bstar: drag coefficient (/earth radii)
    #         Skyfield_EarthSatellite_initial.model.ndot,  # ndot: ballistic coefficient (revs/day)
    #         Skyfield_EarthSatellite_initial.model.nddot,  # nddot: second derivative of mean motion (revs/day^3)
    #         np_particle_mean_elements[1],  # ecco: eccentricity
    #         np_particle_mean_elements[3],  # argpo: argument of perigee (radians)
    #         np_particle_mean_elements[2],  # inclo: inclination (radians)
    #         np_particle_mean_elements[5],  # mo: mean anomaly (radians)
    #         np_particle_mean_elements[0],  # no_kozai: mean motion (radians/minute)
    #         np_particle_mean_elements[4],  # nodeo: right ascension of ascending node (radians)
    #     )
    #
    # else:
    #
    #     orb = pyorb.Orbit(M0=pyorb.M_earth, degrees=True)
    #
    #     orb.x = 1e4 * np_particle_mean_elements[0]
    #     orb.y = 1e4 * np_particle_mean_elements[1]
    #     orb.z = 1e4 * np_particle_mean_elements[2]
    #     orb.vx = np_particle_mean_elements[3]
    #     orb.vy = np_particle_mean_elements[4]
    #     orb.vz = np_particle_mean_elements[5]
    #
    #     mean_motion = 60 * (((astropy.constants.GM_earth.value ** (1 / 3)) / orb.a[0]) ** (3 / 2))
    #
    #     satrec = Satrec()
    #     satrec.sgp4init(
    #         WGS84,  # gravity model
    #         'i',  # 'a' = old AFSPC mode, 'i' = improved mode
    #         Skyfield_EarthSatellite_initial.model.satnum,  # satnum: Satellite number
    #         #Skyfield_EarthSatellite_initial.epoch.tai - 2433281.5,  # epoch: days since 1949 December 31 00:00 UT
    #         Skyfield_EarthSatellite_initial.model.jdsatepoch - 2433281.5 + Skyfield_EarthSatellite_initial.model.jdsatepochF,
    #         Skyfield_EarthSatellite_initial.model.bstar,  # bstar: drag coefficient (/earth radii)
    #         Skyfield_EarthSatellite_initial.model.ndot,  # ndot: ballistic coefficient (revs/day)
    #         Skyfield_EarthSatellite_initial.model.nddot,  # nddot: second derivative of mean motion (revs/day^3)
    #
    #         orb.e[0],  # ecco: eccentricity
    #         orb.omega[0] * (np.pi / 180),  # argpo: argument of perigee (radians)
    #         orb.i[0] * (np.pi / 180),  # inclo: inclination (radians)
    #         orb.anom[0] * (np.pi / 180),  # mo: mean anomaly (radians)
    #         mean_motion,  # no_kozai: mean motion (radians/minute)
    #         orb.Omega[0] * (np.pi / 180),  # nodeo: right ascension of ascending node (radians)
    #     )
    #
    # Skyfield_EarthSatellite_initial.model = satrec
    # Skyfield_EarthSatellite_initial.at(Skyfield_EarthSatellite_post.epoch)

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