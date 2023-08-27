import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import pandas as pd
from matplotlib import cm, colors


import sys
from python_parameters import base_package_load_path
sys.path.insert(0, base_package_load_path)
sys.path.insert(0, base_package_load_path + "/TLE_utilities")
from tle_loading_and_preprocessing import convert_np_keplerian_coordinates_to_cartesian, propagate_np_mean_elements

PATH_LENGTH = 10
PATH_LENGTH_SAT = 20

SATELLITE_PATH_COLORS = ["red", "cyan", "chartreuse"]

def update_plot(num,
                particle_positions,
                particle_traces,
                particle_heads,
                satellite_positions,
                satellite_trace,
                satellite_head,
                x, y, z):

    satellite_color_index = max(int(round((num - 9)/ 20)) % 3, 0)
    path_length = min(num, PATH_LENGTH)
    path_length_sat = min(num, PATH_LENGTH_SAT)
    satellite_trace.set_data(satellite_positions[(num - path_length_sat):num, :2].T)
    satellite_trace.set_3d_properties(satellite_positions[(num - path_length_sat):num, 2].T)
    plt.setp(satellite_trace, color=SATELLITE_PATH_COLORS[satellite_color_index])
    satellite_head.set_data(satellite_positions[(num - 1), :2].T)
    satellite_head.set_3d_properties(satellite_positions[(num - 1), 2].T)
    plt.setp(satellite_head, color=SATELLITE_PATH_COLORS[satellite_color_index])
    for particle_trace, particle_position, particle_head in \
            zip(particle_traces, particle_positions, particle_heads):
        # NOTE: there is no .set_data() for 3 dim data...
        particle_trace.set_data(particle_position[(num - path_length):num, :2].T)
        particle_trace.set_3d_properties(particle_position[(num - path_length):num, 2])
        particle_head.set_data(particle_position[(num-1), :2].T)
        particle_head.set_3d_properties(particle_position[(num-1), 2])
    return particle_traces

# Data: 40 random walks as (num_steps, 3) arrays
EPOCHS = 1000
INFLATION_FACTOR = 1
num_steps = EPOCHS * INFLATION_FACTOR

particle_positions = pickle.load(open("filter_logs/Skysat-C19_c_1_particle_positions.pkl", "rb"))
#pd_df_satellite_positions = pickle.load(open("filter_logs/Skysat-C19_c_1_satellites.pkl", "rb"))

pd_df_satellite_positions = pd.read_pickle("foo.pkl")

# print(satelliteTLEData_satellites.pd_df_tle_data.iloc[:20])
# print(satelliteTLEData_satellites.pd_df_tle_data.iloc[-20:])
# quit()

print(particle_positions[0, :3, :])
perturbations = np.zeros((particle_positions.shape[1], 6))
# for j in range(perturbations.shape[0]):
#     perturbations[j, 0] = 1e-1 * np.random.random()
#     perturbations[j, 2] = 1e-1 * np.random.random()

particle_cartesian_positions = np.zeros((INFLATION_FACTOR * EPOCHS, particle_positions.shape[1], 3))
satellite_cartesian_positions = np.zeros((INFLATION_FACTOR * EPOCHS, 3))
#for i in range(0, positions.shape[0] - 2):
for i in range(0, EPOCHS):
    satellite_cartesian_positions[INFLATION_FACTOR * i, :] = convert_np_keplerian_coordinates_to_cartesian(
        pd_df_satellite_positions.values[i + 1, :])[:3]

    #print(satelliteTLEData_satellites.pd_df_tle_data.values[i, :])
    #print(satellite_cartesian_positions[INFLATION_FACTOR * i, :])

    for k in range(INFLATION_FACTOR - 1):
        next_satellite_pos = propagate_np_mean_elements(pd_df_satellite_positions.pd_df_tle_data.values[i + 1, :],
                                                        pd_df_satellite_positions.list_of_tle_line_tuples[i + 1],
                                                        pd_df_satellite_positions.list_of_tle_line_tuples[i + 2],
                                                        proportion=(k + 1) / INFLATION_FACTOR)
        satellite_cartesian_positions[INFLATION_FACTOR * i + k + 1, :] = (
            convert_np_keplerian_coordinates_to_cartesian(next_satellite_pos)[:3])
    for j in range(0, particle_positions.shape[1]):
        particle_cartesian_positions[INFLATION_FACTOR * i, j, :] = convert_np_keplerian_coordinates_to_cartesian(
            particle_positions[i, j, :] + perturbations[j])[:3]
        for k in range(INFLATION_FACTOR - 1):
            next_pos = propagate_np_mean_elements(particle_positions[i, j, :],
                                                  pd_df_satellite_positions.list_of_tle_line_tuples[i + 1],
                                                  pd_df_satellite_positions.list_of_tle_line_tuples[i + 2],
                                                  proportion = (k + 1)/INFLATION_FACTOR)
            particle_cartesian_positions[INFLATION_FACTOR * i + k + 1, j, :] = (
                convert_np_keplerian_coordinates_to_cartesian(next_pos + perturbations[j])[:3])

reshaped_cartesian_particle_positions = [particle_cartesian_positions[:, index, :] for index in range(20)]
#reshaped_cartesian_satellite_positions = [satellite_cartesian_positions[index, :] for index in range(20)]
#print(reshaped_cartesian_particle_positions[0])

# Attaching 3D axis to the figure
plt.style.use('dark_background')
fig = plt.figure(figsize = (7, 7))
ax = fig.add_subplot(projection="3d", computed_zorder = True)
ax._axis3don = False

#planet = ax.plot([0], [0], [0], '.', markersize = 30, markeredgewidth = 0, color = 'green', alpha = 0.5)[0]

# Create lines initially without data
particle_traces = [ax.plot([], [], [], alpha = 0.5, linewidth = 1)[0] for _ in reshaped_cartesian_particle_positions]
particle_heads = [ax.plot([], [], [], '.', markersize = 5)[0] for _ in reshaped_cartesian_particle_positions]

satellite_trace = ax.plot([], [], [], alpha = 0.5, linewidth = 1, color = 'red')[0]
satellite_head = ax.plot([], [], [], '.', markersize = 10, color = 'red')[0]

# Setting the axes properties
box_side_half_length = 5e6
ax.set(xlim3d=(-box_side_half_length, box_side_half_length), xlabel='X')
ax.set(ylim3d=(-box_side_half_length, box_side_half_length), ylabel='Y')
ax.set(zlim3d=(-box_side_half_length, box_side_half_length), zlabel='Z')

u, v = np.mgrid[0:np.pi:50j, 0:2 * np.pi:50j]
strength = u
norm = colors.Normalize(vmin=np.min(strength),
                        vmax=np.max(strength), clip=False)
radius = 7e5
x = radius * np.sin(u) * np.cos(v)
y = radius * np.sin(u) * np.sin(v)
z = radius * np.cos(u)

ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.viridis,
                    linewidth=0, antialiased=True,
                    facecolors=cm.viridis(norm(strength)), alpha=0.2)

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_plot, num_steps, fargs=(reshaped_cartesian_particle_positions,
                                        particle_traces,
                                        particle_heads,
                                        satellite_cartesian_positions,
                                        satellite_trace,
                                        satellite_head,
                                        x, y, z,
                                        ), interval=75)
#plt.show()
writer = animation.FFMpegFileWriter(fps = 12)
ani.save("foo.mp4",  writer=writer, dpi = 200)
