import numpy as np
cimport numpy as np

cdef extern from "c_gridder_xcode.cpp":
    void _grid "grid"(float complex *data, float *kr, float *ki,
                            float *traj, float *dens,
                            int ndim, int n_points, int *N,
                            float krad, int grid_mod, float *kernel, int nthreads)
    void _igrid "igrid"(float complex *data, float complex *k_data,
                            float *traj, float *dens,
                            int ndim, int n_points, int *N,
                            float krad, int grid_mod, float *kernel, int nthreads) 

def array_prep(A, dtype, linear=True):
    A = np.array(A, dtype=dtype, copy=False, order="C")
    
    if not A.flags['C_CONTIGUOUS']:
        A = np.ascontiguousarray(A)
    
    A = A.astype(dtype, order='C', copy=False)
    
    if linear:
        A = A.ravel()

    return A 

def c_grid(data, grid_params, traj, dens, kernel, nthreads=0): 
    
    n_points = len(dens)

    cdef float complex[::1] data_view = array_prep(data, np.complex64)
    cdef float [::1] traj_view = array_prep(traj, np.float32)
    cdef float [::1] dens_view = array_prep(dens, np.float32)
    cdef float [::1] kernel_view = array_prep(kernel, np.float32)
    cdef int[::1] N_view = array_prep(grid_params['imsize_os'], np.intc)

    kr = np.zeros(grid_params['imsize_os'], np.float32)
    cdef float[::1] kr_view = array_prep(kr, np.float32)

    ki = np.zeros(grid_params['imsize_os'], np.float32)
    cdef float[::1] ki_view = array_prep(ki, np.float32)

    _grid(&data_view[0], &kr_view[0], &ki_view[0], &traj_view[0], &dens_view[0], 
             grid_params["grid_dims"], n_points, &N_view[0], 
             grid_params['krad'], grid_params['grid_mod'], &kernel_view[0], nthreads)

    return kr + 1j*ki

def c_igrid(kspace, grid_params, traj, dens, kernel, nthreads=0):
    
    n_points = len(dens)

    cdef float complex[::1] kspace_view = array_prep(kspace, np.complex64)
    cdef float [::1] traj_view = array_prep(traj, np.float32)
    cdef float [::1] dens_view = array_prep(dens, np.float32)
    cdef float [::1] kernel_view = array_prep(kernel, np.float32)
    cdef int[::1] N_view = array_prep(grid_params['imsize_os'], np.intc)

    data = np.zeros(n_points, np.complex64)
    cdef float complex[::1] data_view = array_prep(data, np.complex64)

    _igrid(&data_view[0], &kspace_view[0], &traj_view[0], &dens_view[0], 
             grid_params["grid_dims"], n_points, &N_view[0], 
             grid_params['krad'], grid_params['grid_mod'], &kernel_view[0], nthreads)

    return data