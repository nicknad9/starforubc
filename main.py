import numpy as np
import pandas as pd
from Hydrocode import MHD
from visualization import visualization as vs

def SaveFile(density, temperature, X, Y):
    df = pd.DataFrame(columns=[])

    df['X'] = X.flatten()
    df['Y'] = Y.flatten()
    df['rho'] = rho.flatten()
    df['temp'] = temperature.flatten()
    df.to_csv('data.csv', mode='a', index=False, header=False)

L = np.arange(-15, 15, 1) + 0.5
X, Y = np.meshgrid(L, L)

grav = 0.1 * np.array([X, Y]) / np.power((X*X + Y*Y), 3)
rho = np.exp(-1 * (X*X + Y*Y) / 200)
vel = np.array([np.random.random((30, 30)),
                np.random.random((30, 30))])
internal = rho * 0.01
energy = internal + 0.5 * rho * np.sum(vel * vel)
pressure = (2.0 / 3.0) * internal
timestep = 0.000000000001
numsteps = 30

df = pd.DataFrame(columns=[])

df['X'] = X.flatten()
df['Y'] = Y.flatten()
df['rho'] = rho.flatten()
df['temp'] = (pressure / rho).flatten()
df.to_csv('data.csv', mode='a', index=False, header=False)

# Velocity: 2D Vector with [Vx (LxLxL), Vy(LxLxL), Vz (LxLxL)]
# Rho: 2-dimensional Array of Scalars
# Grav: 2D Vector with [Gx (LxLxL), Gy (LxLxL), Gz (LxLxL)]
# Energy: 2-dimensional Array of Scalars
# Pressure: 2-dimensional Array of Scalars

grav_x, grav_y = grav[0], grav[1]
padded_grav_x = np.pad(grav_x, ((1,1), (1,1)), mode='constant')
padded_grav_y = np.pad(grav_y, ((1,1), (1,1)), mode='constant')
grav_pad = np.array([padded_grav_x, padded_grav_y])

for step in range(0, numsteps):
    rho, vel, energy, pressure = MHD.Update(rho, vel, energy, pressure, grav_pad, timestep)
    temperature = pressure / rho
    SaveFile(rho, temperature, X, Y)

vs.visualize("data.csv", 13500, 30)
