import numpy as np


class GridKernel:
    """Holds all code and functions for gridding kernel creation and manipulation

    Attributes:
        kx (float ndarray): kernel x axis (radius).
        ky (float ndarray): kernel values
    """

    def __init__(self, grid_params):
        self.krad = grid_params["krad"]
        self.grid_mod = grid_params["grid_mod"]
        self.grid_params = grid_params

        self.calc_kernel(grid_params)
        self.fourier_demod(grid_params)

    def calc_kernel(self, grid_params):
        """Helper function call the real calc kernel functions

        Args:
            grid_params (dict): Common dict of grad parameters
        """
        if grid_params["kernel_type"] == "kb":
            self.calc_kernel_kb(grid_params)
        elif grid_params["kernel_type"] == "tri":
            self.calc_kernel_tri(grid_params)
        elif grid_params["kernel_type"] == "ones":
            self.calc_kernel_ones(grid_params)
        elif grid_params["kernel_type"] == "gauss":
            self.calc_kernel_gauss(grid_params)

    def calc_kernel_kb(self, grid_params):
        """Calculates Kaiser-Bessel kernel, as in the Beatty paper

        Args:
            grid_params (dict): Common dict of grad parameters

        Sets the self.kx and self.ky attributes
        """
        kw0 = 2.0 * grid_params["krad"] / grid_params["over_samp"]
        kr = grid_params["krad"]
        beta = np.pi * np.sqrt((kw0 * (grid_params["over_samp"] - 0.5)) ** 2 - 0.8)
        x = np.linspace(0, kr, grid_params["grid_mod"])

        x_bess = np.sqrt(1 - (x / kr) ** 2)
        y = np.i0(beta * x_bess)
        y = y / y[0]

        x = np.concatenate((x, x + x[-1] + x[1]))
        y = np.concatenate((y, np.zeros(grid_params["grid_mod"])))

        self.kx = x
        self.ky = y

    def calc_kernel_tri(self, grid_params):
        """Calculates triangle kernel

        Args:
            grid_params (dict): Common dict of grad parameters

        Sets the self.kx and self.ky attributes
        """
        kr = grid_params["krad"]
        x = np.linspace(0, kr, grid_params["grid_mod"])

        y = 1.0 - x / kr

        x = np.concatenate((x, x + x[-1] + x[1]))
        y = np.concatenate((y, np.zeros(grid_params["grid_mod"])))

        self.kx = x
        self.ky = y

    def calc_kernel_ones(self, grid_params):
        """Calculates ones kernel

        Args:
            grid_params (dict): Common dict of grad parameters

        Sets the self.kx and self.ky attributes
        """
        kr = grid_params["krad"]
        x = np.linspace(0, kr, grid_params["grid_mod"])

        y = np.ones(x.size)

        x = np.concatenate((x, x + x[-1] + x[1]))
        y = np.concatenate((y, np.zeros(grid_params["grid_mod"])))

        self.kx = x
        self.ky = y

    def calc_kernel_gauss(self, grid_params):
        """Calculates Gaussian kernel

        Args:
            grid_params (dict): Common dict of grad parameters

        Sets the self.kx and self.ky attributes
        """
        kr = grid_params["krad"]
        x = np.linspace(0, kr, grid_params["grid_mod"])

        sig = grid_params["krad"] / 3
        y = 1.0 / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x / sig) ** 2.0)

        x = np.concatenate((x, x + x[-1] + x[1]))
        y = np.concatenate((y, np.zeros(grid_params["grid_mod"])))

        self.kx = x
        self.ky = y

    def fourier_demod(self, grid_params):
        """Takes the self.ky and the image size to perform an FT and
        get the deappodization window.

        Args:
            grid_params (dict): Common dict of grad parameters

        Sets the self.Dx and self.Dy attributes
        """
        self.Dx = []
        self.Dy = []

        for i in range(len(grid_params["imsize_os"])):
            xres = grid_params["imsize_os"][i]
            Dx = np.arange(xres)
            Dx = Dx - xres // 2.0
            Dy = np.zeros(Dx.size, np.complex64)

            for i in range(1, self.kx.size):
                Dy += self.ky[i] * 2 * np.exp(2 * 1j * np.pi * Dx / xres * self.kx[i])

            Dy = Dy.real
            Dy = Dy + self.ky[0]
            Dy = Dy / self.kx.size

            self.Dx.append(Dx)
            self.Dy.append(Dy)

    def apply_deapp(self, A):
        """Performs in place deappodization of the input image

        Args:
            grid_params (ndarray): Image to deapp

        Operation is in-place.  Currently there are no checks for size issues
        """

        if self.grid_params['grid_dims'] == 2:
            A /= (self.Dy[1][np.newaxis, :] * self.Dy[0][:, np.newaxis])
        elif self.grid_params['grid_dims'] == 3:
            A /= (self.Dy[2][np.newaxis, np.newaxis, :] * self.Dy[1][np.newaxis, :, np.newaxis] * self.Dy[0][:, np.newaxis, np.newaxis])


