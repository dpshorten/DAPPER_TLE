import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

import sys
from python_parameters import base_package_load_path
sys.path.insert(0, base_package_load_path)
sys.path.insert(0, base_package_load_path + "/TLE_utilities")
from tle_loading_and_preprocessing import convert_np_keplerian_coordinates_to_cartesian, propagate_np_mean_elements

PATH_LENGTH = 18

def update_plot(num,
                particle_positions,
                particle_traces,
                particle_heads,
                satellite_positions,
                satellite_trace,
                satellite_head,
                ):
    path_length = min(num, PATH_LENGTH)
    satellite_trace.set_data(satellite_positions[(num - path_length):num, :2].T)
    satellite_trace.set_3d_properties(satellite_positions[(num - path_length):num, 2].T)
    satellite_head.set_data(satellite_positions[(num - 1), :2].T)
    satellite_head.set_3d_properties(satellite_positions[(num - 1), 2].T)
    for particle_trace, particle_position, particle_head in \
            zip(particle_traces, particle_positions, particle_heads):
        # NOTE: there is no .set_data() for 3 dim data...
        particle_trace.set_data(particle_position[(num - path_length):num, :2].T)
        particle_trace.set_3d_properties(particle_position[(num - path_length):num, 2])
        particle_head.set_data(particle_position[(num-1), :2].T)
        particle_head.set_3d_properties(particle_position[(num-1), 2])
    return particle_traces

# Data: 40 random walks as (num_steps, 3) arrays
EPOCHS = 10
INFLATION_FACTOR = 100
num_steps = EPOCHS * INFLATION_FACTOR
particle_positions = pickle.load(open("filter_logs/Gosat-2_particle_positions.pkl", "rb"))
satelliteTLEData_satellites = pickle.load(open("filter_logs/Gosat-2_satellites.pkl", "rb"))

print(particle_positions[0, :3, :])
perturbations = np.zeros((particle_positions.shape[1], 6))
for j in range(perturbations.shape[0]):
    perturbations[j, 0] = 1e-1 * np.random.random()
    perturbations[j, 2] = 1e-1 * np.random.random()

particle_cartesian_positions = np.zeros((INFLATION_FACTOR * EPOCHS, particle_positions.shape[1], 3))
satellite_cartesian_positions = np.zeros((INFLATION_FACTOR * EPOCHS, 3))
#for i in range(0, positions.shape[0] - 2):
for i in range(0, EPOCHS):
    satellite_cartesian_positions[INFLATION_FACTOR * i, :] = convert_np_keplerian_coordinates_to_cartesian(
        satelliteTLEData_satellites.pd_df_tle_data.values[i, :])[:3]
    for k in range(INFLATION_FACTOR - 1):
        next_satellite_pos = propagate_np_mean_elements(satelliteTLEData_satellites.pd_df_tle_data.values[i + 1, :],
                                                    satelliteTLEData_satellites.list_of_tle_line_tuples[i + 1],
                                                    satelliteTLEData_satellites.list_of_tle_line_tuples[i + 2],
                                                    proportion=(k + 1) / INFLATION_FACTOR)
        satellite_cartesian_positions[INFLATION_FACTOR * i + k + 1, :] = (
            convert_np_keplerian_coordinates_to_cartesian(next_satellite_pos)[:3])
    for j in range(0, particle_positions.shape[1]):
        particle_cartesian_positions[INFLATION_FACTOR * i, j, :] = convert_np_keplerian_coordinates_to_cartesian(
            particle_positions[i, j, :] + perturbations[j])[:3]
        for k in range(INFLATION_FACTOR - 1):
            next_pos = propagate_np_mean_elements(particle_positions[i, j, :],
                                                  satelliteTLEData_satellites.list_of_tle_line_tuples[i + 1],
                                                  satelliteTLEData_satellites.list_of_tle_line_tuples[i + 2],
                                                  proportion = (k + 1)/INFLATION_FACTOR)
            particle_cartesian_positions[INFLATION_FACTOR * i + k + 1, j, :] = (
                convert_np_keplerian_coordinates_to_cartesian(next_pos + perturbations[j])[:3])

reshaped_cartesian_particle_positions = [particle_cartesian_positions[:, index, :] for index in range(20)]
#reshaped_cartesian_satellite_positions = [satellite_cartesian_positions[index, :] for index in range(20)]
print(reshaped_cartesian_particle_positions[0])

# Attaching 3D axis to the figure
plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax._axis3don = False

# Create lines initially without data
particle_traces = [ax.plot([], [], [], alpha = 0.5)[0] for _ in reshaped_cartesian_particle_positions]
particle_heads = [ax.plot([], [], [], '.')[0] for _ in reshaped_cartesian_particle_positions]

satellite_trace = ax.plot([], [], [], alpha = 0.75, color = 'red')[0]
satellite_head = ax.plot([], [], [], '.', markersize = 20, color = 'red')[0]

# Setting the axes properties
ax.set(xlim3d=(-1e8, 1e8), xlabel='X')
ax.set(ylim3d=(-1e8, 1e8), ylabel='Y')
ax.set(zlim3d=(-1e8, 1e8), zlabel='Z')

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_plot, num_steps, fargs=(reshaped_cartesian_particle_positions,
                                        particle_traces,
                                        particle_heads,
                                        satellite_cartesian_positions,
                                        satellite_trace,
                                        satellite_head,
                                        ), interval=100)

plt.show()
