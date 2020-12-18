import hdf5storage
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import sys, os

from .pygrid_internal import pygrid as pygrid
from .pygrid_internal import c_grid as c_grid
from .pygrid_internal import utils as pygrid_utils

class SimObject:
    def __init__(self):
        self.Nt = 24  # Number of timeframes in the dataset
        self.period = 1000  # periodic phantom interval in [ms]
        self.dt = self.period / self.Nt
        self.tt = np.arange(self.Nt) * self.dt
        self.fov = np.array([360, 360, 8.0]) * 1.0e-3

    def gen_from_generator(self, r, s, t1, t2, Nz=4):
        zz = np.linspace(-0.5, 0.5, Nz)

        self.r = r
        self.sig0 = s
        self.T1 = t1
        self.T2 = t2

        self.Nt = self.r.shape[0]  # Number of timeframes in the dataset
        self.period = 1000  # periodic phantom interval in [ms]
        self.dt = self.period / self.Nt
        self.tt = np.arange(self.Nt) * self.dt

    def gen_standard_cardiac(self, Nz=8, NN=512, t1=800, t2=80, inflow=0):

        zz = np.linspace(-0.5, 0.5, Nz)

        data_loc = os.path.dirname(__file__) + "/cardiac_data/outdata_v7.npz"
        data = np.load(data_loc)

        x_myo = data["xa"].T
        y_myo = data["ya"].T
        z_myo = data["za"].T

        x_myo /= np.abs(x_myo).max() * 9
        y_myo /= np.abs(y_myo).max() * 9
        z_myo = (z_myo - 8) / 16

        blood_rad = []
        for ind_t in range(x_myo.shape[0]):
            blood_rad.append(np.abs(x_myo[ind_t, np.abs(y_myo[ind_t]) < 0.005]).min())
        blood_rad = np.array(blood_rad)

        r_blood = make_blood_pool(blood_rad, Nz=Nz)

        myo_pos = np.array([0.06, -0.14])
        myo_rad = np.array([x_myo.max() - x_myo.min(), y_myo.max() - y_myo.min()]) / 2.0

        x_myo += myo_pos[0]
        y_myo += myo_pos[1]

        r_blood[:, :, 0] += myo_pos[0]
        r_blood[:, :, 1] += myo_pos[1]
        s_blood = np.ones(r_blood.shape[1]) / Nz / 4

        r_myo = np.stack((x_myo, y_myo, z_myo), 2)
        s_myo = np.ones(r_myo.shape[1]) / r_myo.shape[1] / 200 * Nz * NN * NN
        # s_myo = np.ones(r_myo.shape[1]) * 0.2

        data_loc = os.path.dirname(__file__) + "/cardiac_data/SAX_cavity_dark.png"
        img = Image.open(data_loc).convert("L")
        img = img.crop((10, 20, 170, 180))
        img = img.resize((NN, NN))
        img = np.asarray(img).astype("float64")
        img /= img.max()

        y, x = np.meshgrid(
            np.linspace(-0.5, 0.5, NN, False),
            np.linspace(-0.5, 0.5, NN, False),
            indexing="ij",
        )

        x = x.ravel()
        x = np.tile(x[np.newaxis, np.newaxis, :], [self.tt.shape[0], Nz, 1])

        y = y.ravel()
        y = np.tile(y[np.newaxis, np.newaxis, :], [self.tt.shape[0], Nz, 1])

        z = np.tile(zz[np.newaxis, :, np.newaxis], [y.shape[0], 1, y.shape[2]])

        x = np.reshape(x, (self.tt.shape[0], -1))
        y = np.reshape(y, (self.tt.shape[0], -1))
        z = np.reshape(z, (self.tt.shape[0], -1))

        rad = np.sqrt(
            ((x[0] - myo_pos[0]) / myo_rad[0]) ** 2.0
            + ((y[0] - myo_pos[1]) / myo_rad[1]) ** 2.0
        )
        mask = np.ones(x.shape[1], np.bool)
        mask[(rad < 1.0)] = False

        s = np.tile(img[np.newaxis, :], [Nz, 1]).ravel()
        r = np.stack((x, y, z), 2)

        s = s[mask]
        r = r[:, mask, :]

        if inflow > 0:
            r_all = np.concatenate((r, r_myo, r_blood), 1)
            s_all = np.concatenate((s, s_myo, s_blood))
        else:
            r_all = np.concatenate((r, r_myo), 1)
            s_all = np.concatenate((s, s_myo))

        self.r = r_all
        self.sig0 = s_all
        self.T1 = np.ones_like(self.sig0) * t1
        self.T2 = np.ones_like(self.sig0) * t2
        self.r_myo = r_myo

        if inflow > 0:
            self.T1[-s_blood.size:] = inflow

    def shift_positions(self, dt):
        r_new = np.zeros_like(self.r)
        for i in range(self.r.shape[0]):
            t0 = self.tt[i]
            t1 = t0 + dt
            r_new[i] = self.get_pos_time(t1)
        self.r = r_new

    def get_pos_time(self, p_time):
        p_time = p_time % self.period
        for i in range(self.tt.size):
            if self.tt[i] > p_time:
                lo = i - 1
                hi = i

                lo_mod = 1 - (p_time - self.tt[lo]) / self.dt
                hi_mod = (p_time - self.tt[lo]) / self.dt
                break
            elif i == (self.tt.size - 1):
                lo = i
                hi = 0
                lo_mod = 1 - (p_time - self.tt[lo]) / self.dt
                hi_mod = (p_time - self.tt[lo]) / self.dt

        pos = self.r[lo] * lo_mod + self.r[hi] * hi_mod
        return pos

    def get_pos_time_r(self, p_time, r_in):
        p_time = p_time % self.period
        for i in range(self.tt.size):
            if self.tt[i] > p_time:
                lo = i - 1
                hi = i

                lo_mod = 1 - (p_time - self.tt[lo]) / self.dt
                hi_mod = (p_time - self.tt[lo]) / self.dt
                break
            elif i == (self.tt.size - 1):
                lo = i
                hi = 0
                lo_mod = 1 - (p_time - self.tt[lo]) / self.dt
                hi_mod = (p_time - self.tt[lo]) / self.dt

        pos = r_in[lo] * lo_mod + r_in[hi] * hi_mod
        return pos


    def grid_im_from_M(self, pos, M, N_im=256, w=64, oversamp=4.0, krad=1.5, nthreads = 0, use_gpu = False, dens = None):        
        gridder = pygrid.Gridder(
            imsize=(N_im, N_im), grid_dims=2, over_samp=oversamp, krad=krad, use_gpu=use_gpu
        )

        kx_all = pos[:, 0].astype(np.float32)
        ky_all = pos[:, 1].astype(np.float32)
        if dens is None:
            dens = np.ones_like(kx_all)
        else:
            dens = dens.astype(np.float32)

        traj = np.stack((kx_all, ky_all, np.zeros_like(ky_all)), 1).astype(np.float32)
        
        MM = M[:, 0] + 1j * M[:, 1]

        out = None
        if use_gpu:
            out = gridder.cu_k2im(MM.astype(np.complex64), traj, dens, imspace=True)
        else:
            out = gridder.k2im(MM.astype(np.complex64), traj, dens, imspace=True)

        return out


    def get_im_from_M(self, pos, M, N_im=512, w=64):
        xx = pos[:, 0]
        yy = pos[:, 1]

        im = np.zeros((N_im, N_im), np.complex)

        rx = np.round(N_im * (xx + 0.5)).astype("int")
        ry = np.round(N_im * (yy + 0.5)).astype("int")

        for i in range(M.shape[0]):
            if (
                ry[i] >= 0
                and ry[i] < im.shape[0]
                and rx[i] >= 0
                and rx[i] < im.shape[1]
            ):
                im[ry[i], rx[i]] += M[i, 0] + 1j * M[i, 1]

        k = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(im))) / np.sqrt(N_im / w)
        k = k[
            k.shape[0] // 2 - w : k.shape[0] // 2 + w,
            k.shape[1] // 2 - w : k.shape[1] // 2 + w,
        ]
        im2 = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(k)))

        return im2


def make_blood_pool(blood_rad, density=100, Nz=4):
    Nt = blood_rad.size
    zz = np.linspace(-0.5, 0.5, Nz)

    all_x = []
    all_y = []
    for iit in range(Nt):
        lim = 1.1 * blood_rad[iit]
        ddr = np.linspace(-lim, lim, density)
        x, y = np.meshgrid(ddr, ddr)

        x = x.ravel()
        y = y.ravel()

        rad = np.sqrt(x * x + y * y)
        x = x[rad < blood_rad[iit]]
        y = y[rad < blood_rad[iit]]

        x = np.tile(x[np.newaxis, :], [Nz, 1])
        y = np.tile(y[np.newaxis, :], [Nz, 1])

        all_x.append(x)
        all_y.append(y)

    x = np.array(all_x)
    y = np.array(all_y)
    z = np.tile(zz[np.newaxis, :, np.newaxis], [y.shape[0], 1, y.shape[2]])

    x = np.reshape(x, (Nt, -1))
    y = np.reshape(y, (Nt, -1))
    z = np.reshape(z, (Nt, -1))

    r = np.stack((x, y, z), 2)
    return r


if __name__ == "__main__":
    print('Nothing in __main__ right now')
