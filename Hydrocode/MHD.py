import numpy as np
#import gravity

L = 10 #number of mesh points
dim = 3 #dimensionality of space

vel = np.zeros(shape=(dim,L,L,L))

def Divergence(vec, spacing=1):
    gradx = np.array(np.gradient(vec[0], spacing, edge_order=2, axis=0))
    grady = np.array(np.gradient(vec[1], spacing, edge_order=2, axis=1))
    return gradx+grady

def VectorGradient(vec, spacing=1):
    xdevs = np.array(np.gradient(vec, spacing, edge_order=2, axis=1))
    ydevs = np.array(np.gradient(vec, spacing, edge_order=2, axis=2))
    return np.array([xdevs, ydevs])

def StepDensity(rho, vel, spacing):
    return -1 * Divergence(rho * vel, spacing)

def StepVelocity(rho, vel, pressure, grav, spacing):
    return (rho * grav - rho * np.sum(vel * VectorGradient(vel, spacing), axis=0) - np.gradient(pressure, spacing)) / rho

def StepEnergy(rho, vel, grav, pressure, energy, spacing):
    return rho * np.sum(grav * vel, axis=0) - Divergence((energy + pressure) * vel, spacing)

# Velocity: 3D Vector with [Vx (LxLxL), Vy(LxLxL), Vz (LxLxL)]
# Rho: 3-dimensional Array of Scalars
# Grav: 3D Vector with [Gx (LxLxL), Gy (LxLxL), Gz (LxLxL)]
# Energy: 3-dimensional Array of Scalars
# Pressure: 3-dimensional Array of Scalars

def Update(rho, vel, energy, pressure, grav, dt, spacing):
    #Padding
    vel_x, vel_y = vel[0], vel[1]
    padded_vel_x = np.pad(vel_x, ((1,1), (1,1)), mode='constant')
    padded_vel_y = np.pad(vel_y, ((1,1), (1,1)), mode='constant')
    vel_pad = np.array([padded_vel_x, padded_vel_y])

    grav_x, grav_y = grav[0], grav[1]
    padded_grav_x = np.pad(grav_x, ((1,1), (1,1)), mode='constant')
    padded_grav_y = np.pad(grav_y, ((1,1), (1,1)), mode='constant')
    grav_pad = np.array([padded_grav_x, padded_grav_y])

    rho_pad = np.pad(rho, ((1,1), (1,1)), mode='constant')
    energy_pad = np.pad(energy, ((1,1), (1,1)), mode='constant')
    pressure_pad = np.pad(pressure, ((1,1), (1,1)), mode='constant')

    #Calculating
    dE = StepEnergy(rho_pad, vel_pad, grav_pad, pressure_pad, energy_pad, spacing)
    dvel = StepVelocity(rho_pad, vel_pad, pressure_pad, grav_pad, spacing)
    drho = StepDensity(rho_pad, vel_pad, spacing)

    #Updating and Splicing
    vel_pad_updated = (vel_pad + dvel * dt)
    vel_pad_updated_x, vel_pad_updated_y = vel_pad_updated[0], vel_pad_updated[1]
    vel = np.array([vel_pad_updated_x[1:-1, 1:-1],
                    vel_pad_updated_y[1:-1, 1:-1]])
    energy = (energy_pad + dE * dt)[1:-1, 1:-1]
    rho = (rho_pad + drho)[1:-1, 1:-1]
    internal = (1.0 / 2.0) * rho * np.sum(vel * vel) - energy
    pressure = (2.0 / 3.0) * (internal)

    return rho, vel, energy, pressure
