#include <iostream>
#include <complex>
#include <cmath>
// #include <omp.h>
using namespace std;

float get_kval(float dr, float krad, int grid_mod, float *kernel)
{
    // float i = dr / krad * grid_mod;
    // int ri = floor(i + 0.5f);
    // return kernel[ri];

    float i = dr / krad * grid_mod;
    int ri = floor(i);
    float di = i - ri;

    return (kernel[ri] * (1 - di) + kernel[ri + 1] * di);
}


void grid2D(complex<float> *data, float *kr, float *ki, float *traj, float *dens,
            int n_points, int *N, float krad, int grid_mod, float *kernel)
{
    #pragma omp parallel for
    for (int i = 0; i < n_points; i++)
    {

        float x = (traj[i * 3 + 0] + 0.5) * N[1];
        float y = (traj[i * 3 + 1] + 0.5) * N[0];

        int xmin = floor(x - krad);
        int ymin = floor(y - krad);

        int xmax = ceil(x + krad);
        int ymax = ceil(y + krad);

        for (int iy = ymin; iy <= ymax; iy++)
        {
            if ((iy >= 0) && (iy < N[0]))
            {
                float dy = abs(y - iy);
                float kvy = get_kval(dy, krad, grid_mod, kernel);

                for (int ix = xmin; ix <= xmax; ix++)
                {
                    if ((ix >= 0) && (ix < N[1]))
                    {
                        float dx = abs(x - ix);
                        float kvx = get_kval(dx, krad, grid_mod, kernel);

                        float dr = data[i].real() * dens[i] * kvx * kvy;
                        float di = data[i].imag() * dens[i] * kvx * kvy;

                        #pragma omp atomic
                        kr[iy * N[1] + ix] += dr;

                        #pragma omp atomic
                        ki[iy * N[1] + ix] += di;
                    }
                }
            }
        }
    }
    return;
}

void grid3D(complex<float> *data, float *kr, float *ki, float *traj, float *dens,
            int n_points, int *N, float krad, int grid_mod, float *kernel)
{
    #pragma omp parallel for
    for (int i = 0; i < n_points; i++)
    {

        float x = (traj[i * 3 + 0] + 0.5) * N[2];
        float y = (traj[i * 3 + 1] + 0.5) * N[1];
        float z = (traj[i * 3 + 2] + 0.5) * N[0];

        int xmin = floor(x - krad);
        int ymin = floor(y - krad);
        int zmin = floor(z - krad);

        int xmax = ceil(x + krad);
        int ymax = ceil(y + krad);
        int zmax = ceil(z + krad);

        for (int iz = zmin; iz <= zmax; iz++)
        {
            if ((iz >= 0) && (iz < N[0]))
            {
                float dz = abs(z - iz);
                float kvz = get_kval(dz, krad, grid_mod, kernel);
                
                for (int iy = ymin; iy <= ymax; iy++)
                {
                    if ((iy >= 0) && (iy < N[1]))
                    {
                        float dy = abs(y - iy);
                        float kvy = get_kval(dy, krad, grid_mod, kernel);

                        for (int ix = xmin; ix <= xmax; ix++)
                        {
                            if ((ix >= 0) && (ix < N[2]))
                            {
                                float dx = abs(x - ix);
                                float kvx = get_kval(dx, krad, grid_mod, kernel);

                                float dr = data[i].real() * dens[i] * kvx * kvy * kvz;
                                float di = data[i].imag() * dens[i] * kvx * kvy * kvz;

                                #pragma omp atomic
                                kr[iz * N[2] * N[1] + iy * N[2] + ix] += dr;

                                #pragma omp atomic
                                ki[iz * N[2] * N[1] + iy * N[2] + ix] += di;
                            }
                        }
                    }
                }
            }
        }


    }
    return;
}

void igrid2D(complex<float> *data, complex<float> *kspace, float *traj, float *dens,
             int n_points, int *N, float krad, int grid_mod, float *kernel)
{
    #pragma omp parallel for
    for (int i = 0; i < n_points; i++)
    {
        float x = (traj[i * 3 + 0] + 0.5) * N[1];
        float y = (traj[i * 3 + 1] + 0.5) * N[0];
        int xmin = floor(x - krad);
        int ymin = floor(y - krad);
        int xmax = ceil(x + krad);
        int ymax = ceil(y + krad);

        for (int iy = ymin; iy <= ymax; iy++)
        {
            if ((iy >= 0) && (iy < N[0]))
            {
                float dy = abs(y - iy);
                float kvy = get_kval(dy, krad, grid_mod, kernel);
                for (int ix = xmin; ix <= xmax; ix++)
                {
                    if ((ix >= 0) && (ix < N[1]))
                    {
                        float dx = abs(x - ix);
                        float kvx = get_kval(dx, krad, grid_mod, kernel);
                        data[i] += kspace[iy * N[1] + ix] * kvx * kvy;
                    }
                }
            }
        }
    }
    return;
}

void igrid3D(complex<float> *data, complex<float> *kspace, float *traj, float *dens,
             int n_points, int *N, float krad, int grid_mod, float *kernel)
{
    #pragma omp parallel for
    for (int i = 0; i < n_points; i++)
    {
        float x = (traj[i * 3 + 0] + 0.5) * N[2];
        float y = (traj[i * 3 + 1] + 0.5) * N[1];
        float z = (traj[i * 3 + 2] + 0.5) * N[0];

        int xmin = floor(x - krad);
        int ymin = floor(y - krad);
        int zmin = floor(z - krad);

        int xmax = ceil(x + krad);
        int ymax = ceil(y + krad);
        int zmax = ceil(z + krad);

        for (int iz = zmin; iz <= zmax; iz++)
        {
            if ((iz >= 0) && (iz < N[0]))
            {
                float dz = abs(z - iz);
                float kvz = get_kval(dz, krad, grid_mod, kernel);
                
                for (int iy = ymin; iy <= ymax; iy++)
                {
                    if ((iy >= 0) && (iy < N[1]))
                    {
                        float dy = abs(y - iy);
                        float kvy = kvz * get_kval(dy, krad, grid_mod, kernel);
                        
                        for (int ix = xmin; ix <= xmax; ix++)
                        {
                            if ((ix >= 0) && (ix < N[2]))
                            {
                                float dx = abs(x - ix);
                                float kvx = kvy * get_kval(dx, krad, grid_mod, kernel);
                                data[i] += kspace[iz * N[2] * N[1] + iy * N[2] + ix] * kvx;
                            }
                        }
                    }
                }
            }
        }
    }
    return;
}


void grid(complex<float> *data, float *kr, float *ki, float *traj, float *dens,
        int ndim, int n_points, int *N, float krad, int grid_mod, float *kernel, int nthreads)
{
    if (nthreads > 0) {
        // omp_set_num_threads(nthreads);
    }
    
    if (ndim == 2) {
        grid2D(data, kr, ki, traj, dens, n_points, N, krad, grid_mod, kernel);
    } else if (ndim == 3) {
        grid3D(data, kr, ki, traj, dens, n_points, N, krad, grid_mod, kernel);
    }

}

void igrid(complex<float> *data, complex<float> *kspace, float *traj, float *dens,
        int ndim, int n_points, int *N, float krad, int grid_mod, float *kernel, int nthreads)
{   
    if (nthreads > 0) {
        // omp_set_num_threads(nthreads);
    }

    if (ndim == 2) {
        igrid2D(data, kspace, traj, dens, n_points, N, krad, grid_mod, kernel);
    } else if (ndim == 3) {
        igrid3D(data, kspace, traj, dens, n_points, N, krad, grid_mod, kernel);
    }
}