import numpy as np
#import gravity

L = 10 #number of mesh points
dim = 3 #dimensionality of space

vel = np.zeros(shape=(dim,L,L,L))


def Divergence(vec, spacing):
    gradx = np.array(np.gradient(vec[0], spacing, edge_order=2, axis=1))
    grady = np.array(np.gradient(vec[1], spacing, edge_order=2, axis=2))
    gradz = np.array(np.gradient(vec[2], spacing, edge_order=2, axis=3))
    return gradx+grady+gradz

def VectorGradient(vec, spacing):
    xdevs = np.array(np.gradient(vec, spacing, edge_order=2, axis=1))
    ydevs = np.array(np.gradient(vec, spacing, edge_order=2, axis=2))
    zdevs = np.array(np.gradient(vec, spacing, edge_order=2, axis=3))
    return np.array([xdevs, ydevs, zdevs])

def StepDensity(rho, vel):
    return -1 * Divergence(rho * vel)

def StepVelocity(rho, vel, pressure, grav):
    return (rho * grav - rho * np.sum(vel * VectorGradient(vel)) - np.gradient(pressure, poweroftwo)) / rho

def StepEnergy(rho, vel, grav, pressure, energy):
    return rho * grav * vel - Divergence((energy + pressure) * vel)

# Velocity: 3D Vector with [Vx (LxLxL), Vy(LxLxL), Vz (LxLxL)]
# Rho: 3-dimensional Array of Scalars
# Grav: 3D Vector with [Gx (LxLxL), Gy (LxLxL), Gz (LxLxL)]
# Energy: 3-dimensional Array of Scalars
# Pressure: 3-dimensional Array of Scalars

def Update(rho, vel, energy, pressure, grav, dt):
    #Padding
    vel_x, vel_y, vel_z = vel[0],vel[1],vel[2]
    padded_vel_x = np.pad(vel_x, ((1,1), (1,1), (1, 1)), mode='constant')
    padded_vel_y = np.pad(vel_y, ((1,1), (1,1), (1, 1)), mode='constant')
    padded_vel_z = np.pad(vel_z, ((1,1), (1,1), (1, 1)), mode='constant')
    vel_pad = np.array([padded_vel_x,padded_vel_y,padded_vel_z])

    rho_pad = np.pad(dp, ((1,1), (1,1), (1, 1)), mode='constant')
    energy_pad = np.pad(dp, ((1,1), (1,1), (1, 1)), mode='constant')
    pressure_pad = np.pad(dp, ((1,1), (1,1), (1, 1)), mode='constant')

    #Calculating
    dE = StepEnergy(rho_pad, vel_pad, grav_pad, pressure_pad, energy_pad)
    dvel = StepVelocity(rho_pad, vel_pad, pressure_pad, grav_pad)
    drho = StepDensity(rho_pad, vel_pad)

    #Updating and Splicing
    vel_pad_updated = (vel_pad + dvel * dt)
    vel_pad_updated_x, vel_pad_updated_y, vel_pad_updated_z = vel_pad_updated[0], vel_pad_updated[1], vel_pad_updated[2]
    vel = np.array([padded_vel_x[1:-1, 1:-1, 1:-1], padded_vel_y[1:-1, 1:-1, 1:-1], padded_vel_z[1:-1, 1:-1, 1:-1]])
    energy = (energy + dE * dt)[1:-1, 1:-1, 1:-1]
    rho = (rho + drho)[1:-1, 1:-1, 1:-1]
    internal = (1.0 / 2.0) * rho * np.sum(vel * vel) - energy
    pressure = (2.0 / 3.0) * (internal)

    return rho, vel, energy, pressure
