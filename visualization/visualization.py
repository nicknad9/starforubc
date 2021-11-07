import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# PLAN:
# 1. implement multiprocessing with matplotlib (easy)
# 2. reimplement to use vispy and opengl (really tough)

# ~~~ VARIABLES ~~~

# name of file that data will be stored in
csv_to_visualize = "./dummy_data.csv"
# the number of particles per step size
number_particles = 2
# range of x, y, and z axes
xyz_range = 25
# temperature thresholds
cold_thresh = 0
hot_thresh = 10000

# ~~~ FUNCTIONS ~~~

# divides csv file into a group for each step size
def generate_groups(csv_file, num_particles):
    dataframe = pd.read_csv(csv_file)
    num_groups = int(dataframe.shape[0] / num_particles)
    groups = []
    for group_number in range(num_groups):
        start = group_number * num_particles
        end = (group_number + 1) * num_particles
        groups.append(dataframe.iloc[start:end])
    return groups

# choose the opacity based on density
def chooseAlpha(density):
    if density >= 0.8:
        return 1
    elif density <= 0.1:
        return 0
    else:
        return density / 5

# creates a new frame for the animation
def update_animation(frame, *fargs):
    fig = fargs[0]
    a_range = fargs[1]
    ax = fig.add_subplot(xlim=(-a_range, a_range), ylim=(-a_range, a_range))
    # ax.grid(False)
    # ax.set_axis_off()
    for index, row in fargs[2][frame].iterrows():
        ax.scatter(row[0], row[1], c=row[3], cmap="magma" ,alpha=chooseAlpha(row[2]))

# main abstraction for visualizing
def visualize(csv_file, num_particles, axes_range):
    step_dataframes = generate_groups(csv_file, num_particles)
    fig = plt.figure()
    ani = anim.FuncAnimation(fig, update_animation, interval=500, fargs=(fig, axes_range, step_dataframes), frames=len(step_dataframes))
    ani.save('simulation.gif', writer='imagemagick')


visualize("./data.csv", 14400, 15)
