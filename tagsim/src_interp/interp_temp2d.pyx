import cython

import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double floor(double num)
    double ceil(double num)


@cython.boundscheck(False)
@cython.wraparound(False)
def interpolate_temp2D(np.ndarray[float, ndim=3] A,
                        np.ndarray[double, ndim=1] t_in):

    cdef double dt, ddt
    cdef int t0, t1
    cdef int i, ij, ik
    cdef float[:, :, :] out = np.zeros_like(A)

    for i in range(t_in.size):
        t0 = <int>floor(t_in[i]) 
        t1 = <int>ceil(t_in[i])

        dt = t_in[i]-t0
        ddt = (<double>1.0-dt)

        # If the we go past the end of the time array, just use the last frame exactly as is
        if t1 >= A.shape[0]:
            t0 = A.shape[0]-1
            t1 = A.shape[0]-1
            ddt = 1.0
            dt = 0.0

        for ij in range(A.shape[1]):
            for ik in range(A.shape[2]):
                out[i, ij, ik] = ddt * A[t0, ij, ik] + dt * A[t1, ij, ik]
        
    return np.asarray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
def interpolate_temp1D(np.ndarray[float, ndim=2] A,
                        np.ndarray[double, ndim=1] t_in):

    cdef double dt, ddt
    cdef int t0, t1
    cdef int i, ij
    cdef float[:, :] out = np.zeros_like(A)

    for i in range(t_in.size):
        t0 = <int>floor(t_in[i]) 
        t1 = <int>ceil(t_in[i])

        dt = t_in[i]-t0
        ddt = (<double>1.0-dt)

        # If the we go past the end of the time array, just use the last frame exactly as is
        if t1 >= A.shape[0]:
            t0 = A.shape[0]-1
            t1 = A.shape[0]-1
            ddt = 1.0
            dt = 0.0

        for ij in range(A.shape[1]):
            out[i, ij] = ddt * A[t0, ij] + dt * A[t1, ij]
        
    return np.asarray(out)

@cython.boundscheck(False)
@cython.wraparound(False)
def interpolate_temp(np.ndarray[float, ndim=1] A,
                     np.ndarray[double, ndim=1] t_in):

    cdef double dt, ddt
    cdef int t0, t1
    cdef int i
    cdef float[:] out = np.zeros_like(A)
    cdef int N2 
    N2 = A.size//2

    for i in range(t_in.size):
        t0 = <int>floor(t_in[i])
        t1 = <int>ceil(t_in[i])

        dt = t_in[i]-t0
        ddt = (<double>1.0-dt)

        # If the we go past the end of the time array, just use the last frame exactly as is
        if t1 >= N2:
            t0 = N2-1
            t1 = N2-1
            ddt = 1.0
            dt = 0.0

        out[i] = ddt * A[t0] + dt * A[t1]
        out[i+N2] = ddt * A[t0+N2] + dt * A[t1+N2]
        
    return np.asarray(out)