import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

import sys
from python_parameters import base_package_load_path
sys.path.insert(0, base_package_load_path)
sys.path.insert(0, base_package_load_path + "/TLE_utilities")
from tle_loading_and_preprocessing import convert_np_keplerian_coordinates_to_cartesian, propagate_np_mean_elements

PATH_LENGTH = 20
def random_walk(num_steps, max_step=0.05):
    """Return a 3D random walk as (num_steps, 3) array."""
    start_pos = np.random.random(3)
    steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
    walk = start_pos + np.cumsum(steps, axis=0)
    return walk

def update_lines(num, walks, lines, heads):
    for line, walk, head in zip(lines, walks, heads):
        # NOTE: there is no .set_data() for 3 dim data...
        path_length = min(num, PATH_LENGTH)
        line.set_data(walk[(num - path_length):num, :2].T)
        line.set_3d_properties(walk[(num - path_length):num, 2])
        head.set_data(walk[(num-1), :2].T)
        head.set_3d_properties(walk[(num-1), 2])
    return lines

# Data: 40 random walks as (num_steps, 3) arrays
EPOCHS = 10
INFLATION_FACTOR = 25
num_steps = EPOCHS * INFLATION_FACTOR
positions = pickle.load(open("filter_logs/Fengyun-2D_particle_positions.pkl", "rb"))
satelliteTLEData_satellites = pickle.load(open("filter_logs/Fengyun-2D_satellites.pkl", "rb"))

print(positions[0, :3, :])

cartesian_positions = np.zeros((INFLATION_FACTOR * EPOCHS, positions.shape[1], 3))
#for i in range(0, positions.shape[0] - 2):
for i in range(0, EPOCHS):
    for j in range(0, positions.shape[1]):
        cartesian_positions[INFLATION_FACTOR * i, j, :] = convert_np_keplerian_coordinates_to_cartesian(
            positions[i, j, :])[:3]
        for k in range(INFLATION_FACTOR - 1):
            next_pos = propagate_np_mean_elements(positions[i, j, :],
                                              satelliteTLEData_satellites.list_of_tle_line_tuples[i + 1],
                                              satelliteTLEData_satellites.list_of_tle_line_tuples[i + 2],
                                              proportion = (k + 1)/INFLATION_FACTOR)
            cartesian_positions[INFLATION_FACTOR * i + k + 1, j, :] = convert_np_keplerian_coordinates_to_cartesian(next_pos)[:3]

walks = [cartesian_positions[:, index, :] for index in range(20)]
print(walks[0])

# Attaching 3D axis to the figure
plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax._axis3don = False

# Create lines initially without data
lines = [ax.plot([], [], [], alpha = 0.5)[0] for _ in walks]
heads = [ax.plot([], [], [], '.')[0] for _ in walks]

# Setting the axes properties
ax.set(xlim3d=(-1e8, 1e8), xlabel='X')
ax.set(ylim3d=(-1e8, 1e8), ylabel='Y')
ax.set(zlim3d=(-1e8, 1e8), zlabel='Z')

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(walks, lines, heads), interval=100)

plt.show()
