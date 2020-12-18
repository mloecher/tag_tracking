import numpy as np

from scipy.interpolate import griddata
from scipy import interpolate
from scipy.signal.windows import kaiser

from .base_im_generation import grab_random_model
from .SimObject import SimObject
from .SimPSD import SimInstant
from .interp2d import interpolate2Dpoints, interpolate2Dpoints_f, interpolate2Dpoints_fc
from .interp_temp2d import interpolate_temp1D

import sys
import os

from .pygrid_internal import pygrid as pygrid
from .pygrid_internal import c_grid as c_grid
from .pygrid_internal import utils as pygrid_utils

def gen_tag_data_v2(ke=0.15, seed=-1, Nt=25, Nt0 = 25, inflow_range=None, fixed_input=None,
                    beta_lims = [0, 40], blur_chance=0.0, noise_scale = 0.0, basepath = "./image_db/",
                    mode = 'gridtag', use_gpu=False, N_im = 256, do_density = True,
                    random_pts = False, new_sim = False):

    # We can gived fixed inputs here rather than calling the random generator
    if fixed_input is None:
        r, s, t1, t2, final_mask, r_a, r_b, theta, t_sim0, img0, inflow_mask, xmod, ymod, descaler = grab_random_model(
            seed=seed, inflow_range=inflow_range, Nt = Nt0, basepath = im_library
        )
    else:
        r, s, t1, t2, final_mask, r_a, r_b, theta, t_sim0, img0, inflow_mask, xmod, ymod, descaler = fixed_input

    # Generate the list of time points [ms]
    acq_loc = np.arange(0, Nt) * 1000 / Nt + 10.0

    # Make the simulator object and PSD objects
    sim_object = SimObject()
    sim_object.gen_from_generator(r, s, t1, t2)

    simulator = SimInstant(sim_object, use_gpu=use_gpu)
    if (mode == 'gridtag'):
        if new_sim:
            scale_tagrf = np.random.uniform(0.8, 1.2)
            simulator.sample_tagging_smn_PSD(ke=ke, acq_loc=acq_loc, scale_tagrf=scale_tagrf)
        else:
            simulator.sample_tagging1331_v2_PSD(ke=ke, acq_loc=acq_loc)
    elif (mode == 'DENSE'):
        simulator.sample_DENSE_PSD(ke=ke, kd = 0.0, acq_loc=acq_loc)
    
    # Run the PSD simulation, and the extra phase cycles for DENSE if needed
    acqs0 = simulator.run()
    if (mode == 'DENSE'):
        extra_theta = np.linspace(0, 2*np.pi, 4)[1:-1]
        extra_acq = []
        for theta_i in extra_theta:
            simulator = SimInstant(sim_object, use_gpu=use_gpu)
            simulator.sample_DENSE_PSD(rf_dir = [np.cos(theta_i), np.sin(theta_i), 0], ke=ke, kd = 0.0, acq_loc=acq_loc)
            extra_acq.append(simulator.run())


    ###### The following code is all to generate the initial tracking points
    # It is hard coded right now for all tag intersections, but we could do:
    #      - The point subsampling here i.e. only pick 10 or 100 points for speed
    #      - Select random points
    if random_pts:
        rand_scale = 0.9
        rand_N = 5000
        xpts = rand_scale * np.random.rand(rand_N) - rand_scale/2.0
        ypts = rand_scale * np.random.rand(rand_N) - rand_scale/2.0
    else:
        scaler = 1e-3 / ke / sim_object.fov[0] / np.sqrt(2)
        Ntag = np.ceil(0.5 / scaler) + 1

        rr = np.arange(-Ntag, Ntag).astype(np.float32)
        xpts0, ypts0 = np.meshgrid(rr, rr)
        xpts0 = xpts0.ravel()
        ypts0 = ypts0.ravel()

        xpts = (xpts0 + ypts0) * (1e-3 / ke / sim_object.fov[0] / np.sqrt(2))
        ypts = (xpts0 - ypts0) * (1e-3 / ke / sim_object.fov[0] / np.sqrt(2))

        # 0.45 instead of 0.5 for less edge points
        ind = (np.abs(xpts) <= 0.45) & (np.abs(ypts) <= 0.45)
        xpts = xpts[ind]
        ypts = ypts[ind]

    xpts_s = (np.array(xpts) + 0.5) * final_mask.shape[0]
    ypts_s = (np.array(ypts) + 0.5) * final_mask.shape[1]


    ##### Now we sample the cartesian maps for various variables at the tracking points
    # Primarily the motion path variables, and the masks
    tag_ra = interpolate2Dpoints_fc(
        r_a.astype(np.float32),
        xpts_s.astype(np.float32),
        ypts_s.astype(np.float32)
    )
    tag_rb = interpolate2Dpoints_fc(
        r_b.astype(np.float32),
        xpts_s.astype(np.float32),
        ypts_s.astype(np.float32)
    )
    tag_theta = interpolate2Dpoints_fc(
        theta.astype(np.float32),
        xpts_s.astype(np.float32),
        ypts_s.astype(np.float32)
    )

    tag_xmod = []
    tag_ymod = []
    for i in range(Nt):
        temp_x = interpolate2Dpoints_fc(
            xmod[i].astype(np.float32),
            xpts_s.astype(np.float32),
            ypts_s.astype(np.float32)
        )
        temp_y = interpolate2Dpoints_fc(
            ymod[i].astype(np.float32),
            xpts_s.astype(np.float32),
            ypts_s.astype(np.float32)
        )
        tag_xmod.append(temp_x)
        tag_ymod.append(temp_y)

    tag_xmod = np.array(tag_xmod)
    tag_ymod = np.array(tag_ymod)

    tag_mask = interpolate2Dpoints_fc(
        final_mask.astype(np.float32),
        xpts_s.astype(np.float32),
        ypts_s.astype(np.float32)
    )

    if inflow_mask is not None:
        tag_inflow = interpolate2Dpoints_fc(
            inflow_mask.astype(np.float32),
            xpts_s.astype(np.float32),
            ypts_s.astype(np.float32)
        )
    else:
        tag_inflow = None

    tidx_acq = acq_loc / 1000 * t_sim0.size  # 0:Nt style indexing for acq_loc
    tag_xmod_acq = interpolate_temp1D(tag_xmod, tidx_acq)
    tag_ymod_acq = interpolate_temp1D(tag_ymod, tidx_acq)

    # t_sim will be the non linear motion times (t_sim0), sampled at the correct acq_loc
    # The final array will  be in the range 0-2*pi, and will be used for the x y terms next
    t_sim = acq_loc / 1000 * t_sim0.size
    xx = np.arange(t_sim0.size+1)
    f = interpolate.interp1d(xx, np.append(t_sim0, t_sim0[0]+(2*np.pi)))
    t_sim = f(t_sim)

    # This generates the motion paths for the tracked points, following the same code used to generate the 
    # sim object motion paths
    t_sim_i = np.tile(t_sim[:, np.newaxis], [1, tag_ra.size])
    tag_ra_up = np.tile(tag_ra[np.newaxis, :], [t_sim.shape[0], 1])
    tag_rb_up = np.tile(tag_rb[np.newaxis, :], [t_sim.shape[0], 1])

    ell_x = tag_ra_up * (np.cos(t_sim_i) - 1.0)
    ell_y = tag_rb_up * np.sin(t_sim_i)
    dx = np.cos(tag_theta) * ell_x - np.sin(tag_theta) * ell_y + tag_xmod_acq
    dy = np.sin(tag_theta) * ell_x + np.cos(tag_theta) * ell_y + tag_ymod_acq

    # So these are our true motion paths for training
    xpts_motion = np.tile(xpts[np.newaxis, :], [t_sim.shape[0], 1]) + dx * descaler
    ypts_motion = np.tile(ypts[np.newaxis, :], [t_sim.shape[0], 1]) + dy * descaler


    ##### Now we generate the images
    all_im = np.zeros((Nt, N_im, N_im))
    all_imc = np.zeros((Nt, N_im, N_im), np.complex64)
    all_im_pc = np.zeros((Nt, N_im, N_im), np.complex64)

    dens_mod = 1.0
    if do_density:
        dd = get_dens(acqs0[0][0], use_gpu = use_gpu)
        dens_mod = np.median(dd)

    noise_range = [4, 16]
    noise_scale = np.random.rand() * (noise_range[1] - noise_range[0]) + noise_range[0]

    kaiser_range = [2,6]
    kaiser_beta = np.random.rand() * (kaiser_range[1] - kaiser_range[0]) + kaiser_range[0]

    for ii in range(Nt):
        # Generate the images without any noise or artifacts
        if do_density:
            dd = get_dens(acqs0[ii][0], use_gpu = use_gpu)
            dd = dens_mod / (dd + dens_mod * .01)
        else:
            dd = np.ones(acqs0[0][0].shape[0], np.float32)

        im0 = sim_object.grid_im_from_M(acqs0[ii][0], acqs0[ii][1], w=N_im // 2, use_gpu = use_gpu, dens = dd)
        im0 = proc_im(im0, N_im, noise_scale, kaiser_beta)

        if (mode == 'DENSE'):
            extra_im = []
            for acq in extra_acq:
                im_temp = sim_object.grid_im_from_M(acq[ii][0], acq[ii][1], w = 128, use_gpu = use_gpu)
                extra_im.append(im_temp)

        # Generates a phase cycled image for DENSE
        if (mode == 'DENSE'):
            im_pc = im0.copy()
            for i in range(len(extra_im)):
                im_pc += np.conj(np.exp(1j * extra_theta[i])) * extra_im[i]
            all_im_pc[ii] = im_pc

        all_imc[ii] = im0
        all_im[ii] = np.abs(im0)

    return {
        "ims": all_im,
        "pts": (xpts_motion, ypts_motion),
        "tag_mask": tag_mask,
        "tag_inflow": tag_inflow,
        "all_imc": all_imc,
        "all_im_pc": all_im_pc,
    }


# Here we add noise and random blurring
# TODO: make this a sub function with some more control over things, we can then also apply it to the DENSE images
# Also: for DENSE should the phase cycled image have artifacts?  Possibly not, though then we are learning denoising too . . .
def proc_im(im, N_im = 256, noise_scale = 50, kaiser_beta = 4):
    

    k0 = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(im)))
    k0 += noise_scale * (np.random.standard_normal(k0.shape) + 1j * np.random.standard_normal(k0.shape))
    
    # if np.random.rand() < blur_chance:
    #     win_beta = np.random.rand() * (beta_lims[1] - beta_lims[0]) + beta_lims[0]
    #     window = kaiser(N_im, win_beta, sym=False)
    #     k0 *= np.outer(window, window)

    window = kaiser(N_im, kaiser_beta, sym=False)
    k0 *= np.outer(window, window)

    im = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(k0)))
    
    return im



def get_dens(pos, N_im=256, oversamp=2.0, krad=1.5, nthreads = 0, use_gpu = False, scaler = 0.8):
    gridder = pygrid.Gridder(
            imsize=(N_im, N_im), grid_dims=2, over_samp=oversamp, krad=krad, use_gpu=use_gpu
            )

    kx_all = pos[:, 0].astype(np.float32)
    ky_all = pos[:, 1].astype(np.float32)
    dens = np.ones_like(kx_all)

    traj = np.stack((kx_all, ky_all, np.zeros_like(ky_all)), 1).astype(np.float32) * scaler

    MM = np.ones_like(kx_all).astype(np.complex64)

    out = None
    if use_gpu:
        out = gridder.cu_k2im(MM.astype(np.complex64), traj, dens, imspace=True)
    else:
        out = gridder.k2im(MM.astype(np.complex64), traj, dens, imspace=True)

    dd = None
    if use_gpu:
        dd = gridder.cu_im2k(out, traj, dens, imspace=True)
    else:
        dd = gridder.im2k(out, traj, dens, imspace=True)

    # dd = np.abs(dd)
    # dd0 = dd.copy()

    # out = None
    # if use_gpu:
    #     out = gridder.cu_k2im(MM.astype(np.complex64), traj, dd, imspace=True)
    # else:
    #     out = gridder.k2im(MM.astype(np.complex64), traj, dd, imspace=True)

    # dd = None
    # if use_gpu:
    #     dd = gridder.cu_im2k(out, traj, dens, imspace=True)
    # else:
    #     dd = gridder.im2k(out, traj, dens, imspace=True)

    # dd = dd0 / np.abs(dd)
    
    return np.abs(dd)