import math
import numpy as np
import pylab as py
import scipy.sparse as sp                 # import sparse matrix library
from scipy.sparse.linalg import spsolve

def Diff_mat_1D(Nx):

    # First derivative
    D_1d = sp.diags([-1, 1], [-1, 1], shape = (Nx,Nx)) # A division by (2*dx) is required later.
    D_1d = sp.lil_matrix(D_1d)
    D_1d[0,[0,1,2]] = [-3, 4, -1]               # this is 2nd order forward difference (2*dx division is required)
    D_1d[Nx-1,[Nx-3, Nx-2, Nx-1]] = [1, -4, 3]  # this is 2nd order backward difference (2*dx division is required)

    # Second derivative
    D2_1d =  sp.diags([1, -2, 1], [-1,0,1], shape = (Nx, Nx)) # division by dx^2 required
    D2_1d = sp.lil_matrix(D2_1d)
    D2_1d[0,[0,1,2,3]] = [2, -5, 4, -1]                    # this is 2nd order forward difference. division by dx^2 required.
    D2_1d[Nx-1,[Nx-4, Nx-3, Nx-2, Nx-1]] = [-1, 4, -5, 2]  # this is 2nd order backward difference. division by dx^2 required.

    return D_1d, D2_1d




def Diff_mat_2D(Nx,Ny):
    # 1D differentiation matrices
    Dx_1d, D2x_1d = Diff_mat_1D(Nx)
    Dy_1d, D2y_1d = Diff_mat_1D(Ny)


    # Sparse identity matrices
    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)



    # 2D matrix operators from 1D operators using kronecker product
    # First partial derivatives
    Dx_2d = sp.kron(Iy,Dx_1d)
    Dy_2d = sp.kron(Dy_1d,Ix)

    # Second partial derivatives
    D2x_2d = sp.kron(Iy,D2x_1d)
    D2y_2d = sp.kron(D2y_1d,Ix)



    # Return compressed Sparse Row format of the sparse matrices
    return Dx_2d.tocsr(), Dy_2d.tocsr(), D2x_2d.tocsr(), D2y_2d.tocsr()

def get_gravitational_potential(rho):
    # Dirichlet/Neumann boundary conditions at outerwalls (boundary condition type is defined through boundary operators)
    uL = 0
    uR = 0
    uT = 0
    uB = 0

    # Define independent variables
    Nx, Ny = rho.shape              # No. of grid points along x,y direction
    x = np.linspace(-5,5,Nx)        # x variables in 1D
    y = np.linspace(-5,5,Ny)        # y variable in 1D

    dx = x[1] - x[0]                # grid spacing along x direction
    dy = y[1] - y[0]                # grid spacing along y direction

    X,Y = np.meshgrid(x,y)          # 2D meshgrid

    # 1D indexing
    Xu = X.ravel()                  # Unravel 2D meshgrid to 1D array
    Yu = Y.ravel()

    # Source function (right hand side vector)
    rho = rho.flatten()

    ind_unravel_L = np.squeeze(np.where(Xu==x[0]))          # Left boundary
    ind_unravel_R = np.squeeze(np.where(Xu==x[Nx-1]))       # Right boundary
    ind_unravel_B = np.squeeze(np.where(Yu==y[0]))          # Bottom boundary
    ind_unravel_T = np.squeeze(np.where(Yu==y[Ny-1]))       # Top boundary

    ind_boundary_unravel = np.squeeze(np.where((Xu==x[0]) | (Xu==x[Nx-1]) | (Yu==y[0]) | (Yu==y[Ny-1])))  # outer boundaries 1D unravel indices
    ind_boundary = np.where((X==x[0]) | (X==x[Nx-1]) | (Y==y[0]) | (Y==y[Ny-1]))    # outer boundary

    Dx_2d, Dy_2d, D2x_2d, D2y_2d = Diff_mat_2D(Nx,Ny)

    I_sp = sp.eye(Nx*Ny).tocsr()
    L_sys = D2x_2d/dx**2 + D2y_2d/dy**2     # system matrix without boundary conditions

    # Boundary operators
    BD = I_sp       # Dirichlet boundary operator

    L_sys[ind_unravel_T,:] = BD[ind_unravel_T,:]    # Boundaries at the top layer
    L_sys[ind_unravel_B,:] = BD[ind_unravel_B,:]    # Boundaries at the bottom layer
    L_sys[ind_unravel_L,:] = BD[ind_unravel_L,:]    # Boundaries at the left layer
    L_sys[ind_unravel_R,:] = BD[ind_unravel_R,:]    # Boundaries at the right edges

    # Construction of right hand vector (function of x and y)
    b = rho
    # Insert boundary values at the boundary points
    b[ind_unravel_L] = uL
    b[ind_unravel_R] = uR
    b[ind_unravel_T] = uT
    b[ind_unravel_B] = uB
    phi = spsolve(L_sys,b).reshape(Ny,Nx)
    return phi
