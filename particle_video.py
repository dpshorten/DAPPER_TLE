import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PATH_LENGTH = 5
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
num_steps = 30
walks = [random_walk(num_steps) for index in range(40)]

# Attaching 3D axis to the figure
plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax._axis3don = False

# Create lines initially without data
lines = [ax.plot([], [], [], alpha = 0.5)[0] for _ in walks]
heads = [ax.plot([], [], [], '.')[0] for _ in walks]

# Setting the axes properties
ax.set(xlim3d=(0, 1), xlabel='X')
ax.set(ylim3d=(0, 1), ylabel='Y')
ax.set(zlim3d=(0, 1), zlabel='Z')

# Creating the Animation object
ani = animation.FuncAnimation(
    fig, update_lines, num_steps, fargs=(walks, lines, heads), interval=100)

plt.show()
