// Used in cupy based on example from here: https://github.com/cupy/cupy/tree/master/examples/gemm
// TODO: Add 2D functions

#include <cupy/complex.cuh>

extern "C" __global__
void grid2d(complex<float> *data, float *kr, float *ki, float *traj, float *dens,
            long n_points, long *N, float krad, long grid_mod, float *kernel)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < n_points; 
        i += blockDim.x * gridDim.x) 
    {
    if (i < n_points) {           
            float x = (traj[i * 3 + 0] + 0.5) * N[1];
            float y = (traj[i * 3 + 1] + 0.5) * N[0];

            int xmin = floor(x - krad);
            int ymin = floor(y - krad);

            int xmax = ceil(x + krad);
            int ymax = ceil(y + krad);

            // Kernel lookup variables (TODO: give clearer names)
            float kii, kdi;
            int ri;


            for (int iy = ymin; iy <= ymax; iy++)
            {
                if ((iy >= 0) && (iy < N[0]))
                {
                    float dy = abs(y - iy);
                    kii = dy / krad * grid_mod;
                    ri = floor(kii);
                    kdi = kii - ri;
                    float kvy = (kernel[ri] * (1 - kdi) + kernel[ri + 1] * kdi);
                    
                    for (int ix = xmin; ix <= xmax; ix++)
                    {
                        if ((ix >= 0) && (ix < N[1]))
                        {
                            float dx = abs(x - ix);
                            kii = dx / krad * grid_mod;
                            ri = floor(kii);
                            kdi = kii - ri;
                            float kvx = (kernel[ri] * (1 - kdi) + kernel[ri + 1] * kdi);

                            float dr = data[i].real() * dens[i] * kvx * kvy;
                            float di = data[i].imag() * dens[i] * kvx * kvy;

                            atomicAdd(&kr[iy * N[1] + ix], dr);
                            atomicAdd(&ki[iy * N[1] + ix], di);
                        }
                    }
                }
            }

    }
    }
}


extern "C" __global__
void grid3d(complex<float> *data, float *kr, float *ki, float *traj, float *dens,
            long n_points, long *N, float krad, long grid_mod, float *kernel)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < n_points; 
        i += blockDim.x * gridDim.x) 
    {
    if (i < n_points) {           
            float x = (traj[i * 3 + 0] + 0.5) * N[2];
            float y = (traj[i * 3 + 1] + 0.5) * N[1];
            float z = (traj[i * 3 + 2] + 0.5) * N[0];

            int xmin = floor(x - krad);
            int ymin = floor(y - krad);
            int zmin = floor(z - krad);

            int xmax = ceil(x + krad);
            int ymax = ceil(y + krad);
            int zmax = ceil(z + krad);

            // Kernel lookup variables (TODO: give clearer names)
            float kii, kdi;
            int ri;

            for (int iz = zmin; iz <= zmax; iz++)
            {
                if ((iz >= 0) && (iz < N[0]))
                {
                    float dz = abs(z - iz);
                    kii = dz / krad * grid_mod;
                    ri = floor(kii);
                    kdi = kii - ri;
                    float kvz = (kernel[ri] * (1 - kdi) + kernel[ri + 1] * kdi);

                    for (int iy = ymin; iy <= ymax; iy++)
                    {
                        if ((iy >= 0) && (iy < N[1]))
                        {
                            float dy = abs(y - iy);
                            kii = dy / krad * grid_mod;
                            ri = floor(kii);
                            kdi = kii - ri;
                            float kvy = (kernel[ri] * (1 - kdi) + kernel[ri + 1] * kdi);
                            
                            for (int ix = xmin; ix <= xmax; ix++)
                            {
                                if ((ix >= 0) && (ix < N[2]))
                                {
                                    float dx = abs(x - ix);
                                    kii = dx / krad * grid_mod;
                                    ri = floor(kii);
                                    kdi = kii - ri;
                                    float kvx = (kernel[ri] * (1 - kdi) + kernel[ri + 1] * kdi);

                                    float dr = data[i].real() * dens[i] * kvx * kvy * kvz;
                                    float di = data[i].imag() * dens[i] * kvx * kvy * kvz;

                                    atomicAdd(&kr[iz * N[2] * N[1] + iy * N[2] + ix], dr);
                                    atomicAdd(&ki[iz * N[2] * N[1] + iy * N[2] + ix], di);
                                }
                            }
                        }
                    }
                }
            }
    }
    }
}


extern "C" __global__
void igrid2d(complex<float> *data, complex<float> *kspace, float *traj, float *dens,
            long n_points, long *N, float krad, long grid_mod, float *kernel)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < n_points; 
        i += blockDim.x * gridDim.x) 
    {
    if (i < n_points) {           
            float x = (traj[i * 3 + 0] + 0.5) * N[1];
            float y = (traj[i * 3 + 1] + 0.5) * N[0];

            int xmin = floor(x - krad);
            int ymin = floor(y - krad);

            int xmax = ceil(x + krad);
            int ymax = ceil(y + krad);

            // Kernel lookup variables (TODO: give clearer names)
            float kii, kdi;
            int ri;


            for (int iy = ymin; iy <= ymax; iy++)
            {
                if ((iy >= 0) && (iy < N[0]))
                {
                    float dy = abs(y - iy);
                    kii = dy / krad * grid_mod;
                    ri = floor(kii);
                    kdi = kii - ri;
                    float kvy = (kernel[ri] * (1 - kdi) + kernel[ri + 1] * kdi);
                    
                    for (int ix = xmin; ix <= xmax; ix++)
                    {
                        if ((ix >= 0) && (ix < N[1]))
                        {
                            float dx = abs(x - ix);
                            kii = dx / krad * grid_mod;
                            ri = floor(kii);
                            kdi = kii - ri;
                            float kvx = (kernel[ri] * (1 - kdi) + kernel[ri + 1] * kdi);

                            data[i] += kspace[iy * N[1] + ix] * kvx * kvy;
                        }
                    }
                }
            }
    }
    }
}


extern "C" __global__
void igrid3d(complex<float> *data, complex<float> *kspace, float *traj, float *dens,
            long n_points, long *N, float krad, long grid_mod, float *kernel)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < n_points; 
        i += blockDim.x * gridDim.x) 
    {
    if (i < n_points) {           
            float x = (traj[i * 3 + 0] + 0.5) * N[2];
            float y = (traj[i * 3 + 1] + 0.5) * N[1];
            float z = (traj[i * 3 + 2] + 0.5) * N[0];

            int xmin = floor(x - krad);
            int ymin = floor(y - krad);
            int zmin = floor(z - krad);

            int xmax = ceil(x + krad);
            int ymax = ceil(y + krad);
            int zmax = ceil(z + krad);

            // Kernel lookup variables (TODO: give clearer names)
            float kii, kdi;
            int ri;

            for (int iz = zmin; iz <= zmax; iz++)
            {
                if ((iz >= 0) && (iz < N[0]))
                {
                    float dz = abs(z - iz);
                    kii = dz / krad * grid_mod;
                    ri = floor(kii);
                    kdi = kii - ri;
                    float kvz = (kernel[ri] * (1 - kdi) + kernel[ri + 1] * kdi);

                    for (int iy = ymin; iy <= ymax; iy++)
                    {
                        if ((iy >= 0) && (iy < N[1]))
                        {
                            float dy = abs(y - iy);
                            kii = dy / krad * grid_mod;
                            ri = floor(kii);
                            kdi = kii - ri;
                            float kvy = (kernel[ri] * (1 - kdi) + kernel[ri + 1] * kdi);
                            
                            for (int ix = xmin; ix <= xmax; ix++)
                            {
                                if ((ix >= 0) && (ix < N[2]))
                                {
                                    float dx = abs(x - ix);
                                    kii = dx / krad * grid_mod;
                                    ri = floor(kii);
                                    kdi = kii - ri;
                                    float kvx = (kernel[ri] * (1 - kdi) + kernel[ri + 1] * kdi);

                                    data[i] += kspace[iz * N[2] * N[1] + iy * N[2] + ix] * kvx * kvy * kvz;
                                }
                            }
                        }
                    }
                }
            }
    }
    }
}

extern "C" __global__
void deapp3(complex<float> *A, long *N, float *D0, float *D1, float *D2) 
{

for (int k = blockIdx.z * blockDim.z + threadIdx.z; 
         k < N[0]; 
         k += blockDim.z * gridDim.z) 
    {
for (int j = blockIdx.y * blockDim.y + threadIdx.y; 
         j < N[1]; 
         j += blockDim.y * gridDim.y) 
    {    
for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < N[2]; 
         i += blockDim.x * gridDim.x) 
    {
    
    A[k*N[2]*N[1] + j*N[2] + i] /= (D0[k] * D1[j] * D2[i]);
    
    }}}

}


extern "C" __global__
void deapp2(complex<float> *A, long *N, float *D0, float *D1) 
{

    for (int j = blockIdx.y * blockDim.y + threadIdx.y; 
            j < N[0]; 
            j += blockDim.y * gridDim.y) 
    {    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
            i < N[1]; 
            i += blockDim.x * gridDim.x) 
    {
    
    A[j*N[1] + i] /= (D0[j] * D1[i]);
    
    }}

}