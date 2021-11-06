import numpy as np
import scipy as sp
#import gravity

L = 10 #number of mesh points
dim = 3 #dimensionality of space

vel = np.zeroes(shape=(dim,L,L,L))


def Divergence(vel):
    gradx = np.gradient(vel[0], axis=1)
    grady = np.gradient(vel[1], axis=2)
    gradz = np.gradient(vel[2], axis=3)
    return gradx+grady+gradz

def Gradient(vec):
    if vec.ndim == 4:
        xdevs = np.gradient(vel, axis=0)
        ydevs = np.gradient(vel, axis=1)
        zdevs = np.gradient(vel, axis=2)
    elif vec.ndim == 3:
        xdevs = np.gradient(vel, axis=0)
        ydevs = np.gradient(vel, axis=1)
        zdevs = np.gradient(vel, axis=2)
    else:
        xdevs = 0
        ydevs = 0
        zdevs = 0
    return [xdevs, ydevs, zdevs]

def StepDensity(rho, vel):
    return -1 * Divergence(rho * vel)

def StepVelocity(rho, vel, pressure, grav):
    return (rho * grav - rho * np.sum(vel * Gradient(vel)) - Gradient(pressure, poweroftwo)) / rho

def StepEnergy(rho, vel, grav, pressure, energy):
    return rho * grav * vel - Divergence((energy + pressure) * vel)
