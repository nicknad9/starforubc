import numpy as np
import pandas as pd
from Hydrocode import MHD
from visualization import visualization as vs

def SaveFile(density, temperature, X, Y, Z):
    df = pd.DataFrame(columns=[])

    df['X'] = X.flatten()
    df['Y'] = Y.flatten()
    df['Z'] = Z.flatten()
    df['rho'] = rho.flatten()
    df['temp'] = temperature.flatten()
    df.to_csv('data.csv', mode='a', index=False, header=False)

L = np.arange(-15, 15, 1) + 0.5
X, Y, Z = np.meshgrid(L, L, L)

grav = 10 * [X, Y, Z] / np.power((X*X + Y*Y + Z*Z), 3)
rho = np.random.random((31, 31, 31)) * 5
vel = np.array([np.random.random((31, 31, 31)),
                np.random.random((31, 31, 31)),
                np.random.random((31, 31, 31))])
internal = np.random.random((31, 31, 31)) * 10
energy = internal + 0.5 * rho * np.sum(vel * vel)
pressure = (2.0 / 3.0) * internal
timestep = 1
finaltime = 30
SaveFile(rho, temperature, X, Y, Z)

# Velocity: 3D Vector with [Vx (LxLxL), Vy(LxLxL), Vz (LxLxL)]
# Rho: 3-dimensional Array of Scalars
# Grav: 3D Vector with [Gx (LxLxL), Gy (LxLxL), Gz (LxLxL)]
# Energy: 3-dimensional Array of Scalars
# Pressure: 3-dimensional Array of Scalars

for step in range(0, finaltime):
    rho, vel, energy, pressure = MHD.Update(rho, vel, energy, pressure, grav, timestep)
    temperature = pressure / rho
    SaveFile(rho, temperature, X, Y, Z)

vs.visualize("data.csv", 29791, 31)
