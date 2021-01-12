import numpy as np
from .c_grid import c_grid, c_igrid
from .grid_kernel import GridKernel
from .utils import roundup4, zeropad, crop, check_traj_dens
from time import time
import os

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class Gridder:
    """This is the main gridding class
    
    It can mostly just be called with Gridder(imsize=(Nz, Ny, Nz)) or imsize=(Ny,Nx)
    """

    def __init__(self, **kwargs):
        self.traj = 0
        self.dens = 0

        self.use_gpu = bool(kwargs.get("use_gpu", True))

        self.grid_params = {}
        self.grid_params["kernel_type"] = kwargs.get("kernel_type", "kb")
        self.grid_params["krad"] = kwargs.get("krad", 1.5)
        self.grid_params["grid_mod"] = kwargs.get("grid_mod", 32)
        self.grid_params["over_samp"] = kwargs.get("over_samp", 1.5)
        self.grid_params["grid_dims"] = kwargs.get("grid_dims", 2)

        self.grid_params["imsize"] = list(kwargs.get("imsize", (256, 256)))
        for i in range(1, self.grid_params["grid_dims"] + 1):
            self.grid_params["imsize"][-i] = roundup4(self.grid_params["imsize"][-i])

        # Basically get rid of manual control of grid_dims
        if self.grid_params["grid_dims"] != len(self.grid_params["imsize"]):
            self.grid_params["grid_dims"] = len(self.grid_params["imsize"])
            # print('imsize a different dimension than grid_dims, going with grid_dims = %d' %  self.grid_params["grid_dims"])

        self.grid_params["imsize_os"] = list(self.grid_params["imsize"])

        for i in range(1, self.grid_params["grid_dims"] + 1):
            self.grid_params["imsize_os"][-i] = roundup4(
                self.grid_params["imsize_os"][-i] * self.grid_params["over_samp"]
            )

        self.kernel = GridKernel(self.grid_params)

        # TODO: there should be user control as well, in case cupy is installed without GPU
        #           there is functionality to do this built into cupy, but then cupy needs to be installed
        #           which I am not sure is a given for all users I might be dealing with
        if CUDA_AVAILABLE and self.use_gpu:
            self.prep_cuda()
    
    def prep_cuda(self):
        """ Reads in compute kernels for CUDA and copies som eof the smaller fixed items
            onto the GPU.
        """

        # # Load the cuda .ptx file, this isnt really any faster than the RawKernel implementation
        # # but maybe less compiling involved?
        # cubin_filename = os.path.join(os.path.dirname(__file__), "cuda", "cu_gridder_standalone.ptx")
        # cuda_module = cp.RawModule(path = cubin_filename)
        # self.igrid3d_kern = cuda_module.get_function("igrid3d")
        # self.grid2d_kern = cuda_module.get_function("grid2d")
        # self.grid3d_kern = cuda_module.get_function("grid3d")
        # self.deapp2_kern = cuda_module.get_function("deapp2")
        # self.deapp3_kern = cuda_module.get_function("deapp3")


        # Load in all the necessary kernels with cupy.RawKernel()
        code_filename = os.path.join(os.path.dirname(__file__), "cuda", "cu_gridder.cu")
        with open(code_filename, "r") as f:
            code = f.read()
        self.igrid2d_kern = cp.RawKernel(code, "igrid2d")
        self.igrid3d_kern = cp.RawKernel(code, "igrid3d")
        self.grid2d_kern = cp.RawKernel(code, "grid2d")
        self.grid3d_kern = cp.RawKernel(code, "grid3d")
        self.deapp2_kern = cp.RawKernel(code, "deapp2")
        self.deapp3_kern = cp.RawKernel(code, "deapp3")

        # Transfer gridding kernel and image dimensions to GPU
        self.kernel_g = cp.asarray(self.kernel.ky.astype(np.float32))
        self.N_g = cp.asarray(self.grid_params["imsize_os"])

        # Transfer 1D deapp windows onto the GPU
        self.D0_g = cp.asarray(self.kernel.Dy[0])
        self.D1_g = cp.asarray(self.kernel.Dy[1])
        if self.grid_params["grid_dims"] == 3:
            self.D2_g = cp.asarray(self.kernel.Dy[2])

        # print('Prepped CUDA')

    def set_traj_dens(self, traj, dens):
        """Sets object trajectory and densty arrays, and ends to GPU if CUDA is on
        
        Args:
            traj (float32 ndarray): trajectory, will eventually get reshaped to (Nx3)
            dens (float32 ndarray): density, will eventually get reshaped to (Nx1)

        Inputs go through a checker that should get any formatting problems sorted out,
        but its not guaranteed
        """
        self.data_shape = dens.shape
        self.traj, self.dens = check_traj_dens(traj, dens)
        if CUDA_AVAILABLE and self.use_gpu:
            self.traj_g = cp.asarray(self.traj)
            self.dens_g = cp.asarray(self.dens)

    def cu_im2k(self, im, traj=None, dens=None, transfer_cpu=True, imspace = False):
        # Set trajectory and density, ideally this is done previously, though it doesn't really matter
        if traj is not None or dens is not None:
            self.data_shape = dens.shape
            traj, dens = check_traj_dens(traj, dens)
            traj_g = cp.asarray(traj)
            dens_g = cp.asarray(dens)
        else:
            traj_g = self.traj_g
            dens_g = self.dens_g

        # Transfer image to the gpu and zeropad by the oversampling factor
        im_g = cp.asarray(im.astype(np.complex64))

        if imspace:
            im_g = cp.fft.ifftshift(cp.fft.ifftn(cp.fft.fftshift(im_g)))

        im_g = zeropad(im_g, self.grid_params["imsize_os"], use_gpu=True)

        # Apply GPU deappodization
        if self.grid_params["grid_dims"] == 3:
            self.deapp3_kern(
                (64, 64, 64), (8, 8, 8), (im_g, self.N_g, self.D0_g, self.D1_g, self.D2_g)
            )
        elif self.grid_params["grid_dims"] == 2:
            self.deapp2_kern(
                (64, 64), (8, 8), (im_g, self.N_g, self.D0_g, self.D1_g)
            )

        # Perform FFT with cupy (cuFFT)
        im_g = cp.fft.ifftshift(cp.fft.fftn(cp.fft.fftshift(im_g)))

        if self.grid_params["grid_dims"] == 3:
            kern_grid = self.igrid3d_kern
        elif self.grid_params["grid_dims"] == 2:
            kern_grid = self.igrid2d_kern

        # Run through the CUDA inverse gridding kernel
        kdata_g = cu_igrid(
            im_g, self.grid_params, traj_g, dens_g, self.kernel_g, self.N_g, kern_grid, self.data_shape
        )

        # Transfer back to the host if desired
        if transfer_cpu:
            out = kdata_g.get()
        else:
            out = kdata_g

        return out

    def cu_k2im(self, data, traj=None, dens=None, transfer_cpu=True, imspace = False):
        if traj is not None or dens is not None:
            traj, dens = check_traj_dens(traj, dens)
            traj_g = cp.asarray(traj)
            dens_g = cp.asarray(dens)
        else:
            traj_g = self.traj_g
            dens_g = self.dens_g

        # t0 = time()
        data_g = cp.asarray(data.astype(np.complex64))
        
        if self.grid_params["grid_dims"] == 3:
            kern_grid = self.grid3d_kern
        elif self.grid_params["grid_dims"] == 2:
            kern_grid = self.grid2d_kern

        im_g = cu_grid(
            data_g, self.grid_params, traj_g, dens_g, self.kernel_g, self.N_g, kern_grid
        )

        # t1 = time()
        im_g = cp.fft.ifftshift(cp.fft.ifftn(cp.fft.fftshift(im_g)))
        # t2 = time()

        if self.grid_params["grid_dims"] == 3:
            self.deapp3_kern(
                (64, 64, 64), (8, 8, 8), (im_g, self.N_g, self.D0_g, self.D1_g, self.D2_g)
            )
        elif self.grid_params["grid_dims"] == 2:
            self.deapp2_kern(
                (64, 64), (8, 8), (im_g, self.N_g, self.D0_g, self.D1_g)
            )

        # t3 = time()
        im_g = crop(im_g, self.grid_params["imsize"], use_gpu=True)
        # t4 = time()

        if imspace:
            im_g = cp.fft.ifftshift(cp.fft.fftn(cp.fft.fftshift(im_g)))
        
        if transfer_cpu:
            out = im_g.get()
        else:
            out = im_g
        # t5 = time()

        # return out, (t0, t1, t2, t3, t4, t5)
        return out

    def im2k(self, im, traj=None, dens=None, nthreads=0, imspace = False):
        if traj is None:
            traj = self.traj
        if dens is None:
            dens = self.dens


        traj, dens = check_traj_dens(traj, dens)

        if imspace:
            im = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(im)))

        im = zeropad(im, self.grid_params["imsize_os"])

        self.kernel.apply_deapp(im)

        im = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(im)))

        kdata = c_igrid(im, self.grid_params, traj, dens, self.kernel.ky, nthreads)
        # kdata = cu_igrid(im, self.grid_params, traj, dens, self.kernel.ky)


        return kdata 

    def k2im(self, data, traj=None, dens=None, nthreads=0, imspace = False):
        if traj is None:
            traj = self.traj
        if dens is None:
            dens = self.dens

        traj, dens = check_traj_dens(traj, dens)

        im = c_grid(data, self.grid_params, traj, dens, self.kernel.ky, nthreads)
        im = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(im)))
        self.kernel.apply_deapp(im)

        im = crop(im, self.grid_params["imsize"])

        if imspace:
            im = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(im)))

        return im




def cu_igrid(kspace_g, grid_params, traj_g, dens_g, kernel_g, N_g, cu_kernel, data_shape = None):

    n_points = len(dens_g)
    if data_shape is not None:
        data_g = cp.zeros(data_shape, np.complex64)
    else:
        data_g = cp.zeros(dens_g.shape, np.complex64)

    cu_kernel(
        (4096,),
        (128,),
        (
            data_g,
            kspace_g,
            traj_g,
            dens_g,
            n_points,
            N_g,
            np.float32(grid_params["krad"]),
            grid_params["grid_mod"],
            kernel_g,
        ),
    )

    return data_g


def cu_grid(data_g, grid_params, traj_g, dens_g, kernel_g, N_g, cu_kernel):

    n_points = len(dens_g)
    kr = cp.zeros(grid_params["imsize_os"], np.float32)
    ki = cp.zeros(grid_params["imsize_os"], np.float32)

    cu_kernel(
        (4096,),
        (128,),
        (
            data_g,
            kr,
            ki,
            traj_g,
            dens_g,
            n_points,
            N_g,
            np.float32(grid_params["krad"]),
            grid_params["grid_mod"],
            kernel_g,
        ),
    )

    return kr + 1j * ki

