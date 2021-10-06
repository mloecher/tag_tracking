import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import splev, splrep
from scipy.signal import triang, windows
from scipy import ndimage, interpolate

from skimage.filters import gaussian
from skimage.transform import rescale
from skimage.morphology import binary_dilation, disk

from .base_im_generation import gen_motion_params, get_temporal_waveform, get_temporal_waveform2, make_random_mask, get_random_image, map_1Dpoly
from .sim_fullmotion import get_dens, proc_im
from .SimObject import SimObject
from .SimPSD import SimInstant
from .interp2d import interpolate2Dpoints_fc
from .interp_temp2d import interpolate_temp1D


def get_random_heart(NN = 768,
                    Nt = 25,
                    N_im = 256,
                    basepath="./image_db/",
                    rseed=None,
                    seed = -1,
                    img_thresh=0.20,
                    t1_lims=(900, 1200),
                    t2_lims=(40, 70),
                    SNR_range = (10, 30),
                    use_gpu = True,
                    ke = 0.12,
                    mode = 'gridtag',
                    do_density=True,
                    new_sim = True,
                    random_pts = False,
                    TE = .03,
                    crop_N = None):

    final_mask = make_random_mask(NN=NN, rseed=rseed)
    img = get_random_image(basepath, NN=NN, seed=seed)

    res_heart = gen_heart(NN = NN, N_im=N_im)
    r_heart = res_heart['r']

    r0_NN = np.rint((r_heart[0]+0.5) * NN).astype(np.int)
    mask_myo = np.zeros((NN,NN))
    mask_myo[r0_NN[:,0], r0_NN[:,1]] = 1.0

    mask_myo_init = mask_myo.copy()

    mask_blood0 = ndimage.binary_fill_holes(mask_myo)
    # mask_blood0 = ndimage.morphology.binary_erosion(mask_blood0, disk(2))

    img_mask = img > img_thresh
    final_mask = final_mask & img_mask & ~mask_blood0

    s = img
    s = s[final_mask]

    t = res_heart['t']
    r_a = res_heart['r_a'][final_mask]
    r_b = res_heart['r_b'][final_mask]
    theta = res_heart['theta'][final_mask]
    xmod = res_heart['xmod'][:, final_mask]
    ymod = res_heart['ymod'][:, final_mask]

    ell_x = r_a[None, :] * (np.cos(t) - 1.0)[:,None] 
    ell_y = r_b[None, :] * np.sin(t)[:,None] 

    dx = np.cos(theta)[None,:] * ell_x  - np.sin(theta)[None,:] * ell_y + xmod
    dy = np.sin(theta)[None,:] * ell_x  + np.cos(theta)[None,:] * ell_y + ymod

    # dx *= 0.0
    # dy *= 0.0

    mesh_range = np.arange(NN)/NN - 0.5
    xx, yy = np.meshgrid(mesh_range, mesh_range, indexing = 'ij')

    x = xx[final_mask]
    y = yy[final_mask]

    x = x[None, :] + dx
    y = y[None, :] + dy

    z = np.zeros_like(x)

    r = np.stack([x, y, z], 2)

    t1 = map_1Dpoly(s, t1_lims)
    t2 = map_1Dpoly(s, t2_lims)

    s_heart = np.ones(r_heart.shape[1]) * np.random.uniform(0.5, 0.7)
    t1_heart = np.ones(r_heart.shape[1]) * np.random.uniform(t1_lims[0], t1_lims[1])
    t2_heart = np.ones(r_heart.shape[1]) * np.random.uniform(t2_lims[0], t2_lims[1])

    NNa = NN//2

    all_static_mask = np.ones([Nt, NN, NN])
    for it in range(Nt):

        r_NN = np.rint((r_heart[it]+0.5) * NNa).astype(np.int)
        mask_myo = np.zeros((NNa,NNa))
        mask_myo[r_NN[:,0], r_NN[:,1]] = 1.0

        mask_blood = ndimage.binary_fill_holes(mask_myo)
        mask_blood = rescale(mask_blood, 2, 1)   
        mask_myo = rescale(mask_myo, 2, 1)

        mask_cavity = mask_blood - mask_myo
        mask_cavity = ndimage.morphology.binary_dilation(mask_cavity)
        
        all_static_mask[it] -= mask_cavity

    mask_blood0d = ndimage.morphology.binary_dilation(mask_blood0, disk(5), iterations = 3)



    all_point_mask = []
    for it in range(Nt):
        m_temp = all_static_mask[it][mask_blood0d > 0.5]
        m_temp = ~(m_temp > 0.5)
        all_point_mask.append(m_temp)

    x_blood = xx[mask_blood0d > 0.5]
    y_blood = yy[mask_blood0d > 0.5]
    z_blood = np.zeros_like(x_blood)

    r0_blood = np.stack([x_blood, y_blood, z_blood], 1)
        

    r_blood = np.tile(r0_blood, [Nt,1,1])

    s_blood = np.ones(r_blood.shape[1]) * np.random.uniform(0.20, 0.40)
    t1_blood = np.ones(r_blood.shape[1]) * np.random.uniform(30, 60)
    t2_blood = np.ones(r_blood.shape[1]) * np.random.uniform(10, 20)

    r_all = np.concatenate([r_blood, r, r_heart], 1)
    s_all = np.concatenate([s_blood, s, s_heart])
    t1_all = np.concatenate([t1_blood, t1, t1_heart])
    t2_all = np.concatenate([t2_blood, t2, t2_heart])

    acq_loc = np.arange(0, Nt) * 1000 / Nt + 10.0

    sim_object = SimObject()
    sim_object.gen_from_generator(r_all, s_all, t1_all, t2_all)

    simulator = SimInstant(sim_object, use_gpu=use_gpu)
    if (mode == 'gridtag'):
        if new_sim:
            scale_tagrf = np.random.uniform(0.8, 1.2)
            simulator.sample_tagging_smn_PSD(ke=ke, acq_loc=acq_loc, scale_tagrf=scale_tagrf)
        else:
            simulator.sample_tagging1331_v2_PSD(ke=ke, acq_loc=acq_loc)
    elif (mode == 'DENSE'):
        simulator.sample_DENSE_PSD(ke=ke, kd = 0.0, acq_loc=acq_loc, TE = TE)

    acqs0 = simulator.run()

    # For DENSE specifically run the phase cycling acquisition
    if (mode == 'DENSE'):
        extra_theta = np.linspace(0, 2*np.pi, 4)[1:-1]
        extra_acq = []
        for theta_i in extra_theta:
            simulator = SimInstant(sim_object, use_gpu=use_gpu)
            simulator.sample_DENSE_PSD(rf_dir = [np.cos(theta_i), np.sin(theta_i), 0], ke=ke, kd = 0.0, acq_loc=acq_loc, TE=TE)
            extra_acq.append(simulator.run())

    for it in range(Nt):
    
        point_mask = np.ones(s_all.size)
        point_mask[:all_point_mask[it].size] = all_point_mask[it]
        
        acqs0[it][0] = acqs0[it][0][point_mask > 0.5, :]
        acqs0[it][1] = acqs0[it][1][point_mask > 0.5, :]

        if (mode == 'DENSE'):
            for acq in extra_acq:
                acq[it][0] = acq[it][0][point_mask > 0.5, :]
                acq[it][1] = acq[it][1][point_mask > 0.5, :]


    ###### The following code is all to generate the initial tracking points
    # It is hard coded right not for all tag intersections, but we could do:
    #      - The point subsampling here i.e. only pick 10 or 100 points for speed
    #      - Select random points
    if random_pts:

        rand_scale = 0.75
        rand_N = 10000
        xpts = rand_scale * np.random.rand(rand_N) - rand_scale/2.0
        ypts = rand_scale * np.random.rand(rand_N) - rand_scale/2.0

        temp_xs = (np.array(xpts) + 0.5) * final_mask.shape[0]
        temp_ys = (np.array(ypts) + 0.5) * final_mask.shape[1]
    
        tag_mask_temp = interpolate2Dpoints_fc(
            mask_myo_init.astype(np.float32),
            temp_ys.astype(np.float32),
            temp_xs.astype(np.float32)
        )



    else:
    
        scaler = 1e-3 / ke / sim_object.fov[0] / np.sqrt(2)
        Ntag = np.ceil(0.5 / scaler) + 1

        rr = np.arange(-Ntag, Ntag).astype(np.float32)
        xpts0, ypts0 = np.meshgrid(rr, rr, indexing = 'ij')
        xpts0 = xpts0.ravel()
        ypts0 = ypts0.ravel()

        xpts = (xpts0 + ypts0) * (1e-3 / ke / sim_object.fov[0] / np.sqrt(2))
        ypts = (xpts0 - ypts0) * (1e-3 / ke / sim_object.fov[0] / np.sqrt(2))

        ind = (np.abs(xpts) <= 0.45) & (np.abs(ypts) <= 0.45)
        xpts = xpts[ind]
        ypts = ypts[ind]

    xpts_s = (np.array(xpts) + 0.5) * final_mask.shape[0]
    ypts_s = (np.array(ypts) + 0.5) * final_mask.shape[1]


    ##### Now we sample the cartesian maps for various variables at the tracking points
    # Primarily the motion path variables, and the masks
    tag_ra = interpolate2Dpoints_fc(
        res_heart['r_a'].astype(np.float32),
        ypts_s.astype(np.float32),
        xpts_s.astype(np.float32)
    )
    tag_rb = interpolate2Dpoints_fc(
        res_heart['r_b'].astype(np.float32),
        ypts_s.astype(np.float32),
        xpts_s.astype(np.float32)
    )
    tag_theta = interpolate2Dpoints_fc(
        res_heart['theta'].astype(np.float32),
        ypts_s.astype(np.float32),
        xpts_s.astype(np.float32)
    )

    tag_xmod = []
    tag_ymod = []
    for i in range(Nt):
        temp_x = interpolate2Dpoints_fc(
            res_heart['xmod'][i].astype(np.float32),
            ypts_s.astype(np.float32),
            xpts_s.astype(np.float32)
        )
        temp_y = interpolate2Dpoints_fc(
            res_heart['ymod'][i].astype(np.float32),
            ypts_s.astype(np.float32),
            xpts_s.astype(np.float32)
        )
        tag_xmod.append(temp_x)
        tag_ymod.append(temp_y)

    tag_xmod = np.array(tag_xmod)
    tag_ymod = np.array(tag_ymod)

    tag_mask = interpolate2Dpoints_fc(
        mask_myo_init.astype(np.float32),
        ypts_s.astype(np.float32),
        xpts_s.astype(np.float32)
    )

    tidx_acq = acq_loc / 1000 * t.size  # 0:Nt style indexing for acq_loc
    tag_xmod_acq = interpolate_temp1D(tag_xmod, tidx_acq)
    tag_ymod_acq = interpolate_temp1D(tag_ymod, tidx_acq)
    
    # t_sim will be the non linear motion times (t_sim0), sampled at the correct acq_loc
    # The final array will  be in the range 0-2*pi, and will be used for the x y terms next
    t_sim = acq_loc / 1000 * t.size
    xx = np.arange(t.size+1)
    f = interpolate.interp1d(xx, np.append(t, t[0]+(2*np.pi)))
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
    xpts_motion = np.tile(xpts[np.newaxis, :], [t_sim.shape[0], 1]) + dx
    ypts_motion = np.tile(ypts[np.newaxis, :], [t_sim.shape[0], 1]) + dy


    ##### Now we generate the images
    all_im = np.zeros((Nt, N_im, N_im))
    all_imc = np.zeros((Nt, N_im, N_im), np.complex64)
    all_im_pc = np.zeros((Nt, N_im, N_im), np.complex64)

    dens_mod = 1.0
    if do_density:
        dd = get_dens(acqs0[0][0], use_gpu = use_gpu)
        dens_mod = np.median(dd)

    noise_scale = 0.3*256*256/N_im/np.random.uniform(SNR_range[0], SNR_range[1])

    kaiser_range = [2,6]
    kaiser_beta = np.random.rand() * (kaiser_range[1] - kaiser_range[0]) + kaiser_range[0]

    for ii in range(Nt):
        # Generate the images without any noise or artifacts
        if do_density:
            dd = get_dens(acqs0[ii][0], use_gpu = use_gpu)
            dd = dens_mod / (dd + dens_mod * .1)
        else:
            dd = np.ones(acqs0[0][0].shape[0], np.float32)

        im0 = sim_object.grid_im_from_M(acqs0[ii][0], acqs0[ii][1], N_im = N_im, use_gpu = use_gpu, dens = dd)
        im0 = proc_im(im0, N_im, noise_scale, kaiser_beta)

        if (mode == 'DENSE'):
            extra_im = []
            for acq in extra_acq:
                im_temp = sim_object.grid_im_from_M(acq[ii][0], acq[ii][1], N_im = N_im, use_gpu = use_gpu, dens = dd)
                im_temp = proc_im(im_temp, N_im, noise_scale, kaiser_beta)
                extra_im.append(im_temp)

        # Generates a phase cycled image for DENSE
        if (mode == 'DENSE'):
            im_pc = im0.copy()
            for i in range(len(extra_im)):
                im_pc += np.conj(np.exp(1j * extra_theta[i])) * extra_im[i]
            all_im_pc[ii] = im_pc

        all_imc[ii] = im0
        all_im[ii] = np.abs(im0)

    if crop_N is not None:
        start_pos = N_im//2 - crop_N//2
        all_im = all_im[:,start_pos:start_pos+crop_N, start_pos:start_pos+crop_N]
        all_imc = all_imc[:,start_pos:start_pos+crop_N, start_pos:start_pos+crop_N]
        all_im_pc = all_im_pc[:,start_pos:start_pos+crop_N, start_pos:start_pos+crop_N]

    return {
        "ims": all_im,
        "pts": (xpts_motion, ypts_motion),
        "tag_mask": tag_mask,
        "all_imc": all_imc,
        "all_im_pc": all_im_pc,
        "acqs": acqs0,
        "t": res_heart['t'],
        "t": t_sim,
        "r_a": res_heart['r_a'],
        "r_b": res_heart['r_b'],
        "theta": res_heart['theta'],
        "xmod": res_heart['xmod'],
        "ymod": res_heart['ymod'],
        "full_mask": res_heart['full_mask'],
        "r0_lv": res_heart['r0_lv'],
    }




# This is more of a gaussian dilation type thing now
def blur_outer(im, mask, blur=4.0):
    edt, inds = ndimage.distance_transform_edt(mask < 0.5, return_indices=True)

    edt_g = np.exp(-(edt/blur)**2.0)
    im2 = edt_g * im[inds[0], inds[1]]
    
    # im2 = ndimage.filters.gaussian_filter(im, blur) * 1.5

    im2[mask>0.5] = im[mask>0.5]
    return im2


def gen_heart(Np = 10, N_up=100, LV_smooth = .02, NN = 512, Nt = 25, N_im = 256, motion_blur = 1.0):
    
    # Generate points on a circle, with random variations to theta and radius, i.e. LB
    theta0 = (np.random.random(Np) + np.arange(Np)) * 2.0 * np.pi / Np
    rad0 = np.ones_like(theta0) + np.random.random(Np) * 0.5
    rad0 *= 0.5

    theta0 = np.hstack([theta0, theta0[0] + 2 * np.pi])
    rad0 = np.hstack([rad0, rad0[0]])

    x0 = rad0 * np.cos(theta0)
    y0 = rad0 * np.sin(theta0)
    
    # Interpolate up to N_up points, with some smoothing
    theta0_up = np.linspace(0, 2*np.pi, N_up) + theta0[0]

    spl = splrep(theta0, x0, per=True, s = LV_smooth)
    x0_up = splev(theta0_up, spl)

    spl = splrep(theta0, y0, per=True, s = LV_smooth)
    y0_up = splev(theta0_up, spl)
    
    # Select points on the LV curve to be RV insertion points
    c0_ind = 0
    c1_ind = int(N_up / (2.5 + np.random.rand()))

    c0 = np.array([x0_up[c0_ind], y0_up[c0_ind]])
    c1 = np.array([x0_up[c1_ind], y0_up[c1_ind]])
    
    # Get the location of the center of the RV, called "offset" here
    offset_theta = theta0_up[c0_ind] + (theta0_up[c1_ind] - theta0_up[c0_ind]) / 2.0
    offset_rad = np.array([x0_up[c1_ind//2], y0_up[c1_ind//2]])
    offset_rad = np.linalg.norm(offset_rad)

    offset_x = offset_rad * np.cos(offset_theta)
    offset_y = offset_rad * np.sin(offset_theta)
    offset = np.array([offset_x, offset_y])
    
    # Get the position of the insertion points, relative to the RV center "offset"
    # The negative and 2pi parts is to make sure the path goes in the right direction
    p1_start = c0 - offset
    rad1_start = np.linalg.norm(p1_start)
    theta1_start = np.arctan2(p1_start[1], p1_start[0])

    p1_end = c1 - offset
    rad1_end = np.linalg.norm(p1_end)
    theta1_end = np.arctan2(p1_end[1], p1_end[0])
    if theta1_end < 0:
        theta1_end += 2* np.pi
        
    # Generate the path and upsample it, very similar to the LV, but not periodic
    theta1 = np.linspace(theta1_start, theta1_end, Np)
    theta1[1:-1] += 0.2 * (np.random.random(Np-2) - 0.5)
    rad1 = np.ones_like(theta1) * offset_rad
    # The triangle here is to try to make it less round, but Im not really sure its needed
    rad1[1:-1] *= (1 + 0.2*triang(Np-2, sym=False))
    rad1[0] = rad1_start
    rad1[-1] = rad1_end

    x1 = rad1 * np.cos(theta1)
    y1 = rad1 * np.sin(theta1)


    theta1_up = np.linspace(theta1_start, theta1_end, N_up)

    spl = splrep(theta1, x1, per=False)
    x1_up = splev(theta1_up, spl)

    spl = splrep(theta1, y1, per=False)
    y1_up = splev(theta1_up, spl)

    x1_up += offset[0]
    y1_up += offset[1]
    
    # Make a mask of the regions
    mesh_range = np.arange(NN)/NN - 0.5
    xx, yy = np.meshgrid(mesh_range, mesh_range, indexing = 'ij')
    im_coords = np.array([xx.ravel(), yy.ravel()])
    
    # Size of the heart (basically LV diameter), entered as pixels in a 256 image, then scaled to -0.5 to 0.5 with the /256
    size_scale = np.random.uniform(32.0, 45.0) / 256.0

    # Randomly rotate everything
    theta = 2 * np.pi * np.random.rand()
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    # apply the rotation
    offset_s = offset * size_scale
    offset_s_r = R@offset_s

    x0_up_r = c * x0_up + -s * y0_up
    y0_up_r = s * x0_up + c * y0_up

    x1_up_r = c * x1_up + -s * y1_up
    y1_up_r = s * x1_up + c * y1_up
    
    # Scale the LV and RV paths
    lv_path = np.array([x0_up_r, y0_up_r]) * size_scale
    rv_path = np.array([x1_up_r, y1_up_r]) * size_scale
    both_path = np.hstack([lv_path, rv_path])
    
    # LV thickness (actually radius)
    lv_width = np.random.uniform(3.5, 6.0) / 256.0
    
    # Find points within lv_width from lv_path
    diff = im_coords[:, None, :] - lv_path[:, :, None]
    diff = np.hypot(diff[0], diff[1])
    min_dist = np.min(diff, axis=0)
    min_dist = np.reshape(min_dist, [NN, NN])
    
    lv_mask = min_dist < lv_width
    
    # RV_width is a little (random amount) smaller than LV
    rv_width = lv_width - 2.5*np.random.rand()/256.0
    
    diff = im_coords[:, None, :] - rv_path[:, :, None]
    diff = np.hypot(diff[0], diff[1])

    min_dist = np.min(diff, axis=0)
    min_dist = np.reshape(min_dist, [NN, NN])

    rv_mask = min_dist < rv_width
    
    # Get rid of parts of the RV mask that are in the LV mask
    rv_mask = rv_mask & ~lv_mask
    
    # full mask == 1 for LV, and == 2 for RV
    full_mask = lv_mask + 2*rv_mask
    
    # Now onto the motion generation
    ##################
    
    # This gives the radius from 1 (endocardium) to 0 (epicardium) for LV
    r0_lv = im_coords[:, lv_mask.ravel()>0].T
    lv_rad = np.hypot(r0_lv[:,0], r0_lv[:,1])
    lv_rad -= lv_rad.min()
    lv_rad /= lv_rad.max()
    lv_rad = 1.0 - lv_rad
    
    # This gives the radius from 1 (endocardium) to 0 (epicardium) for RV
    r0_rv = im_coords[:, rv_mask.ravel()>0].T
    rv_rad = np.hypot(r0_rv[:,0]-offset_s_r[0], r0_rv[:,1]-offset_s_r[1])
    rv_rad -= rv_rad.min()
    rv_rad /= rv_rad.max()
    rv_rad = 1.0 - rv_rad
    
    # Get all initial points together
    r0 = np.vstack([r0_lv, r0_rv])
    init_rad = np.hypot(r0[:,0], r0[:,1])
    init_theta = np.arctan2(r0[:,1], r0[:,0])
    
    

    # Generate the motion parameters that define general contraction motion
    r_a = np.random.uniform(0.0, 0.006) * np.ones_like(init_rad)
    r_a[:r0_lv.shape[0]] += np.random.uniform(0.003, 0.008)*lv_rad
    r_a[r0_lv.shape[0]:] += np.random.uniform(0.04, 0.14)*(init_rad[r0_lv.shape[0]:]**2.0)*rv_rad
    r_a[r0_lv.shape[0]:] += np.random.uniform(0.003, 0.008)*rv_rad

    r_b = 0.75 * r_a * np.random.rand()
    
    # Add some random twist by changing the direction away from center of LV
    theta_mod = np.random.rand() - 0.5
    theta_c = init_theta + theta_mod
    
    # Blur and add purturbations to the motion
    ##################

    # # Temporal waveform 0 to 2pi
    # t = get_temporal_waveform(Nt)
    
    # # Pertubation fields
    # r_a2, r_b2, theta2, extra_p2 = gen_ellipse_params(NN=NN, extra_poly=4)
    # r_a2 = (r_a2 - r_a2.mean()) * 0.02
    # r_b2 = (r_b2 - r_b2.mean()) * 0.02
    # theta2 = (theta2 - theta2.mean()) * 0.15

    # filt = windows.tukey(Nt, 0.6)[:, np.newaxis, np.newaxis]
    # filt[0] = 0.0
    # filt[-1] = 0.10 * np.random.rand()
    # xx = np.linspace(-1, 1, Nt)[:, np.newaxis, np.newaxis]

    # p0, p1 = extra_p2[0][np.newaxis], extra_p2[1][np.newaxis]
    # xmod = (p0 * xx**1.0 + p1 * xx**2.0) * filt * 0.04

    # p2, p3 = extra_p2[2][np.newaxis], extra_p2[3][np.newaxis]
    # ymod = (p2 * xx**1.0 + p3 * xx**2.0) * filt * 0.04

    temp_method = np.random.randint(2)
    if temp_method == 0:
        t = get_temporal_waveform(Nt)  
    elif temp_method == 1:
        t = get_temporal_waveform2(Nt)

    r_a2, r_b2, theta2, extra_p2 = gen_motion_params(NN=NN, extra_poly=4)
    # r_a2 = (r_a2 - r_a2.mean()) * np.abs((0.008 + 0.008 * np.random.standard_normal()))
    # r_b2 = (r_b2 - r_b2.mean()) * np.abs((0.004 + 0.008 * np.random.standard_normal()))
    # theta2 = (theta2 - theta2.mean()) * np.abs((0.15 + 0.005 * np.random.standard_normal()))

    r_a2 = (r_a2 - r_a2.mean()) * np.random.uniform(.010, .030)
    r_b2 = (r_b2 - r_b2.mean()) * np.random.uniform(.005, .015)
    theta2 = (theta2 - theta2.mean()) * np.random.uniform(0.10, 0.20)



    filt = np.hstack([0, np.diff(t)])[:, None, None]
    filt += .01 * np.random.standard_normal(filt.shape)

    xx0 = np.linspace(0, 1, Nt)[:, np.newaxis, np.newaxis] + np.random.uniform(-1.0, 1.0)
    xx1 = np.linspace(0, 1, Nt)[:, np.newaxis, np.newaxis] + np.random.uniform(-1.0, 1.0)
    xx2 = np.linspace(0, 1, Nt)[:, np.newaxis, np.newaxis] + np.random.uniform(-1.0, 1.0)
    xx3 = np.linspace(0, 1, Nt)[:, np.newaxis, np.newaxis] + np.random.uniform(-1.0, 1.0)

    p0, p1 = extra_p2[0][np.newaxis], extra_p2[1][np.newaxis]
    xmod = (p0 * xx0**1.0 + p1 * xx1**2.0) * filt * np.random.uniform(0.015, 0.025)

    p2, p3 = extra_p2[2][np.newaxis], extra_p2[3][np.newaxis]
    ymod = (p2 * xx2**1.0 + p3 * xx3**2.0) * filt * np.random.uniform(0.015, 0.025)


    
    # Point image indexes
    r0_NN = np.rint((r0+0.5) * NN).astype(np.int)
    
    # Put existing motion fields into image versions
    mask_NN = np.zeros((NN,NN))
    mask_NN[r0_NN[:,0], r0_NN[:,1]] = 1.0

    r_a_NN = np.zeros((NN,NN))
    r_a_NN[r0_NN[:,0], r0_NN[:,1]] = r_a.copy()

    r_b_NN = np.zeros((NN,NN))
    r_b_NN[r0_NN[:,0], r0_NN[:,1]] = r_b.copy()

    theta_c_NN = np.zeros((NN,NN), np.complex)
    theta_c_NN[r0_NN[:,0], r0_NN[:,1]] = np.exp(1j*theta_c).copy()

    r_a_NN += r_a2 * mask_NN
    r_b_NN += r_b2 * mask_NN
    theta_c_NN.real += theta2 * mask_NN
    theta_c_NN.imag += theta2 * mask_NN

    mask_NN_b = gaussian(mask_NN, motion_blur, preserve_range=True) + 1e-16
    r_a_NN = gaussian(r_a_NN, motion_blur, preserve_range=True) / mask_NN_b
    r_b_NN = gaussian(r_b_NN, motion_blur, preserve_range=True) / mask_NN_b
    theta_c_NN.real = gaussian(theta_c_NN.real, motion_blur, preserve_range=True) / mask_NN_b
    theta_c_NN.imag = gaussian(theta_c_NN.imag, motion_blur, preserve_range=True) / mask_NN_b

    xmod_out = np.zeros_like(xmod)
    ymod_out = np.zeros_like(ymod)
    for it in range(Nt):
        xmod[it] *= mask_NN
        xmod[it] = gaussian(xmod[it], motion_blur, preserve_range=True) / mask_NN_b
        xmod[it] *= mask_NN
        xmod_out[it] = blur_outer(xmod[it], mask_NN)

        ymod[it] *= mask_NN
        ymod[it] = gaussian(ymod[it], motion_blur, preserve_range=True) / mask_NN_b
        ymod[it] *= mask_NN
        ymod_out[it] = blur_outer(ymod[it], mask_NN)


    r_a_NN *= mask_NN
    r_b_NN *= mask_NN
    theta_c_NN.real *= mask_NN
    theta_c_NN.imag *= mask_NN

    r_a_out = blur_outer(r_a_NN, mask_NN)
    r_b_out = blur_outer(r_b_NN, mask_NN)
    theta_c_out = np.zeros_like(theta_c_NN)
    theta_c_out.real = blur_outer(theta_c_NN.real, mask_NN)
    theta_c_out.imag = blur_outer(theta_c_NN.imag, mask_NN)

    scaler = NN/512
    r_a_out *= scaler
    r_b_out *= scaler
    theta_c_out *= scaler
    xmod_out *= scaler
    ymod_out *= scaler

    r_a_ff = r_a_out[r0_NN[:,0], r0_NN[:,1]]
    r_b_ff = r_b_out[r0_NN[:,0], r0_NN[:,1]]
    theta_c_ff = np.angle(theta_c_out[r0_NN[:,0], r0_NN[:,1]])
    xmod_ff = xmod_out[:, r0_NN[:,0], r0_NN[:,1]]
    ymod_ff = ymod_out[:, r0_NN[:,0], r0_NN[:,1]]
    
    # Compute actual pointwise motion
    ell_x = r_a_ff[None, :] * (np.cos(t) - 1.0)[:,None] 
    ell_y = r_b_ff[None, :] * np.sin(t)[:,None] 

    dx = np.cos(theta_c_ff)[None,:] * ell_x  - np.sin(theta_c_ff)[None,:] * ell_y + xmod_ff
    dy = np.sin(theta_c_ff)[None,:] * ell_x  + np.cos(theta_c_ff)[None,:] * ell_y + ymod_ff
    
    # dx *= 0.0
    # dy *= 0.0

    # Final point cloud motion paths for RV and LV
    r = r0[None,...] + np.stack((dx, dy),2)
    r = np.concatenate([ r, np.zeros_like(r[:,:,:1]) ], 2)
    
    return {'r': r,
            't': t,
            'r_a': r_a_out,
            'r_b': r_b_out,
            'theta': np.angle(theta_c_out),
            'theta_c': theta_c_out,
            'xmod': xmod_out,
            'ymod': ymod_out,
            'full_mask': full_mask,
            'r0_lv': r0_lv}