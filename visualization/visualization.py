import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# ~~~ FUNCTIONS ~~~

def generate_groups(csv_file, num_particles):
    dataframe = pd.read_csv(csv_file)
    num_groups = int(dataframe.shape[0] / num_particles)
    groups = []
    for group_number in range(num_groups):
        start = group_number * num_particles
        end = (group_number + 1) * num_particles
        groups.append(dataframe.iloc[start:end])
    return groups

def update_animation(frame, *fargs):
    fig = fargs[0]
    # specifies range of x, y and z values => WILL CHANGE
    ax = fig.add_subplot(xlim=(0, 25), ylim=(0, 25), zlim=(0, 25), projection='3d')
    ax.grid(False)
    ax.set_axis_off()
    for index, row in fargs[1][frame].iterrows():
        ax.scatter(row[0], row[1], row[2])
    
def visualize(csv_file, num_particles):
    step_dataframes = generate_groups(csv_file, num_particles)
    fig = plt.figure()
    ani = anim.FuncAnimation(fig, update_animation, interval=500, fargs=(fig, step_dataframes), frames=len(step_dataframes))
    ani.save('dummy_data.gif', writer='imagemagick')


# ~~~ SPECIFICATION ~~~

# name of file that data will be stored in
csv_to_visualize = "./dummy_data.csv"
# the number of particles per step size
number_particles = 2

visualize("./dummy_data.csv", number_particles)