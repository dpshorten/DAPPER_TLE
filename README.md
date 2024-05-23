# Particle filter based manoeuvre detection methods for TLE data

Author: David Shorten <dpshorten@gmail.com>

## Background

This package was written as part of the SmartSat CRC project P2.11: Trusted AI Frameworks for Change and Anomaly Detection 
in Observed ISR. This project was mostly conducted from early 2022 to early 2024 at the Univerity of Adelaide.

The goal of the project was to develop techniques for detecting manoeuvres from the Two-Line-Element data made available 
by the USA's 18th Space Defense Squadron.

This particular package implements a particle filter based approach to manoeuvre detection. The particle filters are used
to track the mean Keplerian elements of the satellites from the noisy observations in the TLE data. From the inferred 
mean elements at a given epoch, and the associated uncertainty, a prediction can be made for the mean elements at the 
subsequent epoch, along with an uncertainty. If the observed mean elements at the subsequent epoch are sufficiently
unlikely given the predicted mean elements and uncertainty, a manoeuvre is inferred.

## Running

To install dependencies, run:

pip install -r requirements.txt

Running code in this package requires the following packages to be situated at the same directory level:
- `TLE_utilities`
- `python-sgp4` (the version edited as part of this SmartSat project)
- `TLE_observation_benchmark_dataset`

The experiments on the benchmark and simulated data can be run using the shell scripts:
- benchmark_set_script.sh
- simulated_set_script.sh

These can be found in the `scripts` directory.

## Package Structure

This package is a fork of the `DAPPER` package, which is a particle filter library 
(<github.com/nansencenter/DAPPER>). New, higher-level driving code for manoeuvre detection can be found in the 
directory 'TLE_anomaly_detection/'. Other new code can be found in `dapper/mods/mean_element_propagation/`. 
Significant edits have been made to `dapper/da_methods/particle.py`. 