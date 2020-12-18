import cython

import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double floor(double num)
    double ceil(double num)


@cython.boundscheck(False)
@cython.wraparound(False)
def interpolate2Dpoints(np.ndarray[double, ndim=2] A,
                        np.ndarray[double, ndim=1] r0,
                        np.ndarray[double, ndim=1] r1):

    cdef double dx, dy, ddx, ddy
    cdef double temp
    cdef int x0, x1, y0, y1
    cdef int i
    cdef double[:] out = np.zeros(r0.size)

    for i in range(r0.size):
        x0 = <int>floor(r0[i])
        x1 = <int>ceil(r0[i])

        y0 = <int>floor(r1[i])
        y1 = <int>ceil(r1[i])

        dx = r0[i]-x0
        dy = r1[i]-y0
    
        ddx = (<double>1.0-dx)
        ddy = (<double>1.0-dy)

    
        temp = ddx*ddy*A[y0,x0] + \
              ddx*dy*A[y1,x0] + \
              dx*ddy*A[y0,x1] + \
              dx*dy*A[y1,x1]

        out[i] = temp
        
    return np.asarray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
def interpolate2Dpoints_f(np.ndarray[float, ndim=2] A,
                        np.ndarray[float, ndim=1] r0,
                        np.ndarray[float, ndim=1] r1):

    cdef float dx, dy, ddx, ddy
    cdef float temp
    cdef int x0, x1, y0, y1
    cdef int i
    cdef float[:] out = np.zeros(r0.size, np.float32)

    for i in range(r0.size):
        x0 = <int>floor(r0[i])
        x1 = <int>ceil(r0[i])

        y0 = <int>floor(r1[i])
        y1 = <int>ceil(r1[i])

        dx = r0[i]-x0
        dy = r1[i]-y0
    
        ddx = (<float>1.0-dx)
        ddy = (<float>1.0-dy)

    
        temp = ddx*ddy*A[y0,x0] + \
              ddx*dy*A[y1,x0] + \
              dx*ddy*A[y0,x1] + \
              dx*dy*A[y1,x1]

        out[i] = temp
        
    return np.asarray(out)


@cython.boundscheck(False)
@cython.wraparound(False)
def interpolate2Dpoints_fc(np.ndarray[float, ndim=2] A,
                        np.ndarray[float, ndim=1] r0,
                        np.ndarray[float, ndim=1] r1):

    cdef float dx, dy, ddx, ddy
    cdef float A00, A10, A01, A11
    cdef float temp
    cdef int x0, x1, y0, y1
    cdef int i
    cdef float[:] out = np.zeros(r0.size, np.float32)

    for i in range(r0.size):
        x0 = <int>floor(r0[i])
        x1 = <int>ceil(r0[i])

        y0 = <int>floor(r1[i])
        y1 = <int>ceil(r1[i])

        dx = r0[i]-x0
        dy = r1[i]-y0
    
        ddx = (<float>1.0-dx)
        ddy = (<float>1.0-dy)

        if (y0 >= 0) and (y0 < A.shape[0]) and (x0 >= 0) and (x0 < A.shape[1]):
            A00 = A[y0,x0]
        else:
            A00 = 0.0

        if (y1 >= 0) and (y1 < A.shape[0]) and (x0 >= 0) and (x0 < A.shape[1]):
            A10 = A[y1,x0]
        else:
            A10 = 0.0

        if (y0 >= 0) and (y0 < A.shape[0]) and (x1 >= 0) and (x1 < A.shape[1]):
            A01 = A[y0,x1]
        else:
            A01 = 0.0

        if (y1 >= 0) and (y1 < A.shape[0]) and (x1 >= 0) and (x1 < A.shape[1]):
            A11 = A[y1,x1]
        else:
            A11 = 0.0


        temp = ddx*ddy*A00 + \
              ddx*dy*A10 + \
              dx*ddy*A01 + \
              dx*dy*A11

        out[i] = temp
        
    return np.asarray(out)