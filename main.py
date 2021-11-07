import numpy as np
import pandas as pd
from Hydrocode import MHD
from visualization import visualization as vs
from Hydrocode import gravity

def SaveFile(density, temperature, X, Y):
    df = pd.DataFrame(columns=[])

    df['X'] = X.flatten()
    df['Y'] = Y.flatten()
    df['rho'] = rho.flatten()
    df['temp'] = temperature.flatten()
    df.to_csv('data.csv', mode='a', index=False, header=False)

L = np.arange(-15, 15, 0.25) + 0.125
X, Y = np.meshgrid(L, L)

rho = np.exp(-1 * (X*X + Y*Y) / 200)
grav = np.gradient(gravity.get_gravitational_potential(rho))
vel = np.array([np.ones((120, 120)) * 0.1,
                np.ones((120, 120)) * 0.1])
internal = rho
energy = internal + 0.5 * rho * np.sum(vel * vel)
pressure = (2.0 / 3.0) * internal
timestep = 0.00001
numsteps = 30

df = pd.DataFrame(columns=[])

df['X'] = X.flatten()
df['Y'] = Y.flatten()
df['rho'] = rho.flatten()
df['temp'] = (pressure / rho).flatten()
df.to_csv('data.csv', index=False, header=False)

# Velocity: 2D Vector with [Vx (LxLxL), Vy(LxLxL), Vz (LxLxL)]
# Rho: 2-dimensional Array of Scalars
# Grav: 2D Vector with [Gx (LxLxL), Gy (LxLxL), Gz (LxLxL)]
# Energy: 2-dimensional Array of Scalars
# Pressure: 2-dimensional Array of Scalars

grav_x, grav_y = grav[0], grav[1]
padded_grav_x = np.pad(grav_x, ((1,1), (1,1)), mode='constant')
padded_grav_y = np.pad(grav_y, ((1,1), (1,1)), mode='constant')
grav_pad = np.array([padded_grav_x, padded_grav_y])

for step in range(1, numsteps+1):
    rho, vel, energy, pressure = MHD.Update(rho, vel, energy, pressure, grav, timestep, 0.25)
    temperature = pressure / rho
    grav = np.gradient(gravity.get_gravitational_potential(rho))
    if (step%5 == 0):
        SaveFile(rho, temperature, X, Y)

vs.visualize("data.csv", 13500, 30)
