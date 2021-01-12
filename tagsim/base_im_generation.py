import numpy as np
import os
import random

from PIL import Image
from scipy import interpolate
from skimage.morphology import opening, disk
from skimage.filters import gaussian
from scipy.signal import windows

from .perlin import generate_perlin_noise_2d


# This is a function that needs to be removed eventually, the temporal waveform
# should probably be based on the speed
def get_temporal_waveform(Nt):
    t2 = gen_1D_poly(Nt, [0, 1], order=2, scale= 1.0)
    t2 = np.hstack((0.0, t2))

    mod_lim = [0.5, 0.9]
    mod_max = np.random.rand() * (mod_lim[1] - mod_lim[0]) + mod_lim[0]
    mod_max = int(mod_max * Nt)
    mod = windows.cosine(mod_max)
    mod = 10.0 * np.random.rand() * np.hstack((0.0, mod, np.zeros(Nt-mod.size)))
    t2 += mod

    t2 = np.cumsum(t2)
    t2 *= 2*np.pi/t2[-1]
    t2 = t2[:-1]

    return t2

# Same as above, I think I can do better than this, but this fives the temporal
# plot
def get_temporal_waveform2(Nt):
    Nt2 = Nt + np.random.randint(10)

    mod_lim = [0.4, 0.9]
    skip_chance = 0.8
    sec_height = [0.0, 0.3]
    mod3_height = [0.0, 0.1]

    mod_max = np.random.uniform(mod_lim[0], mod_lim[1])

    mod_max = int(mod_max * Nt)
    mod = windows.cosine(mod_max)
    if np.random.rand() < skip_chance:
        mod = mod[1:]

    N2 = Nt2-mod.size
    height2 = np.random.uniform(sec_height[0], sec_height[1])
    mod2 = np.ones(N2) * height2

    height3 = np.random.uniform(mod3_height[0], mod3_height[1])
    mod3 = height3 * windows.hamming(N2)
    mod2 += mod3


    mod_upper = mod.copy()
    mod_upper[mod < height2] = height2
    mod[mod_max//2:] = mod_upper[mod_max//2:]

    mod[mod.size-1] = (mod[mod.size-2] + height2) / 1.9


    mod = np.hstack((0.0, mod, mod2))

    
    x_stop = np.linspace(np.pi, -np.random.rand()*np.pi, np.random.randint(3,7))
    y_stop = (np.tanh(x_stop) + 1)/2
    y_stop = np.hstack([y_stop, np.zeros(np.random.randint(3))])
    y_stop = np.hstack((np.ones(mod.size - y_stop.size), y_stop))


    mod *= y_stop

    mod += np.random.uniform(0.0, 0.03) * np.random.standard_normal(mod.size)
    mod[0] = 0
    # mod[np.random.randint(1,3)] += np.random.uniform(0.0, 3.0)
    
    t2 = np.cumsum(mod)
    t2 *= 2*np.pi/t2[-1]
    t2 = t2[:Nt]
    return t2

# Generate a random 1D function by random polynomial coefficients
def gen_1D_poly(Np, lims, order=3, scale = 1.0):
    x = np.linspace(0, 1, Np) 
    px = np.arange(order + 1)
    N = px.size

    A = np.zeros((x.size, N))

    for i in range(N):
        temp = x ** px[i]
        A[:, i] = temp.flatten()

    coeff = scale * np.random.standard_normal(N)

    b = A @ coeff

    win = np.random.rand() * (lims[1] - lims[0])
    cen = np.random.rand() * (lims[1] - lims[0] - win) + lims[0] + win / 2

    b -= b.min()
    b /= b.max()
    b *= win
    b += cen - win / 2

    return b


# Generate a random 1D poly like above, and use it to map input values to output
# values (i.e. for T1 and T2 calcualtion of the phantom from grayscale values)
def map_1Dpoly(xnew, lims, order=3, Np=256):

    x = np.linspace(-1, 1, Np)
    px = np.arange(order + 1)
    N = px.size

    A = np.zeros((x.size, N))

    for i in range(N):
        temp = x ** px[i]
        A[:, i] = temp.flatten()

    coeff = np.random.standard_normal(N)

    b = A @ coeff

    win = np.random.rand() * (lims[1] - lims[0])
    cen = np.random.rand() * (lims[1] - lims[0] - win) + lims[0] + win / 2

    b -= b.min()
    b /= b.max()
    b *= win
    b += cen - win / 2

    xx = np.linspace(0, 1, b.size)

    f = interpolate.interp1d(xx, b)
    t1 = f(xnew)

    return t1


# Generate random 2D field, with random polynomial coefficients, and then
# additional perlin noise. 
# TODO: Add control over te perlin noise, random amplitude to it
def gen_2Dpoly(NN=256, shift=True, fit_order = 3):

    x = np.linspace(-1, 1, NN) + np.random.uniform(-0.5, 0.5)
    y = np.linspace(-1, 1, NN) + np.random.uniform(-0.5, 0.5)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing="xy")

    [px, py] = np.meshgrid(range(fit_order + 1), range(fit_order + 1), indexing="ij")

    idx = (px + py) <= fit_order

    px = px[idx]
    py = py[idx]
    powers = np.vstack((px, py)).T

    N = powers.shape[0]

    A = np.zeros((xv.size, N))

    for i in range(N):
        temp = xv ** px[i] + yv ** py[i]
        A[:, i] = temp.ravel()

    coeff = np.random.standard_normal(N)

    b = A @ coeff
    b = np.reshape(b, (NN, NN))

    perlin_res = np.random.choice([2,4], 2)
    nn = generate_perlin_noise_2d((NN,NN), perlin_res)

    # print(b.min(), b.max(), nn.min(), nn.max())

    b += 4.0*nn

    poly_range = b.max() - b.min()

    b /= poly_range

    if shift:
        b += 1 - b.max()

    return b


# Generates a ellipse with random values within the given ranges
# This is mostly the same as the next function, they should be combined
#
# Essentially, the "body"-like big ellipse in the image
def gen_outer_ellipse(offset_lim=(-0.15, 0.15), radius_lim=(0.8, 1.2), NN=256):
    ox = np.random.rand() * (offset_lim[1] - offset_lim[0]) + offset_lim[0]
    oy = np.random.rand() * (offset_lim[1] - offset_lim[0]) + offset_lim[0]
    rx = np.random.rand() * (radius_lim[1] - radius_lim[0]) + radius_lim[0]
    ry = np.random.rand() * (radius_lim[1] - radius_lim[0]) + radius_lim[0]
    xx, yy = np.meshgrid(
        np.linspace((-1 + ox) / rx, (1 + ox) / rx, NN),
        np.linspace((-1 + oy) / ry, (1 + oy) / ry, NN),
        indexing="xy",
    )
    mask = xx * xx + yy * yy <= 1

    return mask

# Generates a ellipse with random values within the given ranges
# This is mostly the same as the previous function, they should be combined; the
# only difference is that this one negates the mask, excluding the inner portion
#
# Essentially, the random holes in the image
def gen_inner_ellipse(offset_lim=(-0.35, 0.35), radius_lim=(0.05, 0.2), NN=256):
    ox = np.random.rand() * (offset_lim[1] - offset_lim[0]) + offset_lim[0]
    oy = np.random.rand() * (offset_lim[1] - offset_lim[0]) + offset_lim[0]
    rx = np.random.rand() * (radius_lim[1] - radius_lim[0]) + radius_lim[0]
    ry = np.random.rand() * (radius_lim[1] - radius_lim[0]) + radius_lim[0]
    xx, yy = np.meshgrid(
        np.linspace((-1 + ox) / rx, (1 + ox) / rx, NN),
        np.linspace((-1 + oy) / ry, (1 + oy) / ry, NN),
        indexing="xy",
    )
    mask = xx * xx + yy * yy <= 1

    return ~mask


# Generates a annulus ellipse mask, (ellipse with some thickness)
def gen_hollow_ellipse(
    offset_lim=(-0.25, 0.25),
    radius_lim=(0.2, 0.4),
    drad_lim=(0.08, 0.15),
    outer_lim=(0.1, 0.3),
    NN=256,
):
    ox = np.random.rand() * (offset_lim[1] - offset_lim[0]) + offset_lim[0]
    oy = np.random.rand() * (offset_lim[1] - offset_lim[0]) + offset_lim[0]
    rx = np.random.rand() * (radius_lim[1] - radius_lim[0]) + radius_lim[0]
    ry = np.random.rand() * (radius_lim[1] - radius_lim[0]) + radius_lim[0]
    xx, yy = np.meshgrid(
        np.linspace((-1 + ox) / rx, (1 + ox) / rx, NN),
        np.linspace((-1 + oy) / ry, (1 + oy) / ry, NN),
        indexing="xy",
    )
    mask1 = xx * xx + yy * yy <= 1

    drx_inner = np.random.rand() * (drad_lim[1] - drad_lim[0]) + drad_lim[0]
    dry_inner = np.random.rand() * (drad_lim[1] - drad_lim[0]) + drad_lim[0]

    rx_inner = rx - drx_inner
    ry_inner = ry - dry_inner

    xx, yy = np.meshgrid(
        np.linspace((-1 + ox) / rx_inner, (1 + ox) / rx_inner, NN),
        np.linspace((-1 + oy) / ry_inner, (1 + oy) / ry_inner, NN),
        indexing="xy",
    )
    mask2 = xx * xx + yy * yy <= 1

    drx_outer = np.random.rand() * (outer_lim[1] - outer_lim[0]) + outer_lim[0]
    dry_outer = np.random.rand() * (outer_lim[1] - outer_lim[0]) + outer_lim[0]

    rx_outer = rx + drx_outer
    ry_outer = ry + dry_outer

    xx, yy = np.meshgrid(
        np.linspace((-1 + ox) / rx_outer, (1 + ox) / rx_outer, NN),
        np.linspace((-1 + oy) / ry_outer, (1 + oy) / ry_outer, NN),
        indexing="xy",
    )
    mask3 = xx * xx + yy * yy > 1

    return (mask1.astype(np.int) - mask2.astype(np.int) + mask3.astype(np.int)) > 0


# Calls all of the mask functions randomly and concatenates them
def make_random_mask(odds=(0.5, 0.5, 0.5), NN=512, rseed=None):

    if rseed is not None:
        random.seed(rseed)
        np.random.seed(rseed)

    all_mask = []
    all_mask.append(np.ones((NN, NN)))

    pick = np.random.rand()
    if pick > odds[0]:
        all_mask.append(gen_outer_ellipse(NN=NN))

    pick = np.random.rand()
    if pick > odds[1]:
        all_mask.append(gen_hollow_ellipse(NN=NN))

    for odd in odds[2:]:
        pick = np.random.rand()
        if pick > odd:
            all_mask.append(gen_inner_ellipse(NN=NN))

    all_mask = np.array(all_mask).astype(np.bool)

    final_mask = np.all(all_mask, 0)
    return final_mask


# This generates the underlying motion fields for the full volume simulation
def gen_motion_params(NN=256, rseed=None, extra_poly = 0):

    if rseed is not None:
        np.random.seed(rseed)
    

    yy, xx = np.meshgrid(
        np.linspace(-1, 1, NN, False), np.linspace(-1, 1, NN, False), indexing="ij"
    )
    rr = np.sqrt(xx * xx + yy * yy)

    a_range = (2, 8)
    b_range = (1.1, 1.5)

    beta = np.random.rand() * (a_range[1] - a_range[0]) + a_range[0]
    cutoff = np.random.rand() * (b_range[1] - b_range[0]) + b_range[0]

    filt = 0.5 + 1.0 / np.pi * np.arctan(beta * (cutoff - rr.ravel()) / cutoff)
    filt = np.reshape(filt, rr.shape)

    # p0 = gen_2Dpoly(NN=NN)
    # r_a = p0 * filt
    # p1 = gen_2Dpoly(NN=NN) + 0.5
    # r_b = r_a * p1

    p0 = gen_2Dpoly(NN=NN)
    r_a = p0 * filt

    # The extra scale factor here controls how round the path is, with smaller numbers meaning less round
    p1 = gen_2Dpoly(NN=NN)
    r_b = np.random.rand() * p1 * filt

    theta = (
        np.random.rand() * ((gen_2Dpoly(NN=NN) * 2 * np.pi) - np.pi)
        + np.random.rand() * 2 * np.pi
    )

    extra_p = []
    for i in range(extra_poly):
        extra_p.append(gen_2Dpoly(NN=NN, shift=False) * filt)

    return r_a, r_b, theta, extra_p

# Load in the ppm files (from an image database that is nicely more medical
# image like)
def load_coil100(ppm_file):

    with open(ppm_file, "rb") as fd:
        pnm = type("pnm", (object,), {})  ## create an empty container
        pnm.header = fd.readline().decode("ascii")
        pnm.magic = pnm.header.split()[0]
        pnm.maxsample = 1 if (pnm.magic == "P4") else 0

        while len(pnm.header.split()) < 3 + (1, 0)[pnm.maxsample]:
            s = fd.readline().decode("ascii")
            if len(s) and s[0] != "#":
                pnm.header += s
            else:
                pnm.header += ""

        pnm.width, pnm.height = [int(item) for item in pnm.header.split()[1:3]]
        pnm.samples = 3 if (pnm.magic == "P6") else 1
        if pnm.maxsample == 0:
            pnm.maxsample = int(pnm.header.split()[3])
        pnm.pixels = np.fromfile(
            fd,
            count=pnm.width * pnm.height * pnm.samples,
            dtype="u1" if pnm.maxsample < 256 else "<u2",
        )
        pnm.pixels = (
            pnm.pixels.reshape(pnm.height, pnm.width)
            if pnm.samples == 1
            else pnm.pixels.reshape(pnm.height, pnm.width, pnm.samples)
        )

        im2 = Image.fromarray((pnm.pixels / pnm.pixels.max() * 255).astype("uint8"), mode="RGB")
        im2 = im2.convert("L")

    return im2

# Pick a random image from the given folder, load it into a numpy array, and
# scale to a max of 1.0
def get_random_image(basepath, NN=256, seed=-1):

    pics = []
    for root, dirnames, filenames in os.walk(basepath):
        for filename in filenames:
            if filename.endswith((".ppm", ".jpg", ".JPEG")):
                pics.append(os.path.join(root, filename))

    if seed >= 0:
        ppath = pics[seed]
    else:
        ppath = random.choice(pics)

    if ppath.endswith(".ppm"):
        img = load_coil100(ppath)
    else:
        img = Image.open(ppath).convert("L")

    img = img.resize((NN, NN))
    img = np.asarray(img).astype("float64")
    img /= img.max()

    return img


# Generate a random image and its corresponding motion parameters, and MR
# parameters 
# The output of this is used for putting into the MR simulator to get
# cine images
def grab_random_model(
    NN=512,
    Nt=25,
    img_thresh=0.15,
    t1_lims=(900, 1200),
    t2_lims=(80, 400),
    # basepath="E:\\image_db\\",
    basepath="./image_db/",
    seed=-1,
    inflow_range=None,
    rseed=None,
    mseed=None
):
    final_mask = make_random_mask(NN=NN, rseed=rseed)
    # final_mask = np.ones_like(final_mask)
    r_a, r_b, theta, extra_p = gen_motion_params(NN=NN, rseed=mseed, extra_poly=4)
    r_a *= 0.04
    r_b *= 0.04

    filt = windows.tukey(Nt, 0.3)[:, np.newaxis, np.newaxis]
    filt[0] = 0.0
    for i in range(Nt//2, Nt):
        if filt[i] < 0.2:
            filt[i] = (0.2 + filt[i]) / 2
    xx = np.linspace(-1, 1, Nt)[:, np.newaxis, np.newaxis]

    p0, p1 = extra_p[0][np.newaxis], extra_p[1][np.newaxis]
    xmod = (p0 * xx**1.0 + p1 * xx**2.0) * filt * (.02 + .01 * np.random.standard_normal())

    p2, p3 = extra_p[2][np.newaxis], extra_p[3][np.newaxis]
    ymod = (p2 * xx**1.0 + p3 * xx**2.0) * filt * (.02 + .01 * np.random.standard_normal())

    r_a0 = r_a.copy()
    r_b0 = r_b.copy()
    theta0 = theta.copy()

    img = get_random_image(basepath, NN=NN, seed=seed)

    img_mask = img > img_thresh
    final_mask = final_mask & img_mask
    final_mask0 = final_mask.copy()
    final_mask = final_mask.ravel()

    s = img.ravel()
    # s[~final_mask] *= 0.0
    s = s[final_mask]

    if inflow_range is not None:
        inflow_mask = (img > inflow_range[0]) & (img < inflow_range[1])
        ii1 = gaussian(inflow_mask, 20.0)
        if ii1.max() > 0.0:
            ii1 /= ii1.max()
            ii1 = ii1 > 0.5
            ii1 = opening(ii1, selem=disk(5))
            inflow_mask = ii1.copy()
            inflow_lin = inflow_mask.ravel()
            inflow_lin = inflow_lin[final_mask]
        else:
            inflow_mask = None
    else:
        inflow_mask = None

    # t2 = gen_1D_poly(Nt, [0, 10], order=2)
    # t2 = np.hstack((0.0, t2))
    # t2 = np.cumsum(t2)
    # t2 *= 2*np.pi/t2[-1]
    # t2 = t2[:-1]

    temp_method = np.random.randint(2)
    if temp_method == 0:
        t2 = get_temporal_waveform(Nt)  
    elif temp_method == 1:
        t2 = get_temporal_waveform2(Nt)

    # t = np.linspace(0, np.pi * 2, Nt, endpoint=False)
    t = t2.copy()
    t0 = t.copy()
    t = np.tile(t[:, np.newaxis], [1, s.size])

    r_a = r_a.ravel()
    r_a = r_a[final_mask]

    r_b = r_b.ravel()
    r_b = r_b[final_mask]

    theta = theta.ravel()
    theta = theta[final_mask]

    xmod0 = xmod.copy()
    ymod0 = ymod.copy()

    xmod = np.reshape(xmod, [Nt, -1])
    ymod = np.reshape(ymod, [Nt, -1])

    xmod = xmod[:, final_mask]
    ymod = ymod[:, final_mask]

    r_a = np.tile(r_a[np.newaxis, :], [t.shape[0], 1])
    r_b = np.tile(r_b[np.newaxis, :], [t.shape[0], 1])

    # print(t.shape)
    # print(r_a.shape, r_b.shape, theta.shape)

    ell_x = r_a * (np.cos(t) - 1.0)
    ell_y = r_b * np.sin(t)
    dx = np.cos(theta) * ell_x - np.sin(theta) * ell_y + xmod
    dy = np.sin(theta) * ell_x + np.cos(theta) * ell_y + ymod

    y, x = np.meshgrid(
        np.linspace(-0.5, 0.5, NN, False), np.linspace(-0.5, 0.5, NN, False), indexing="ij"
    )

    x = x.ravel()
    x = x[final_mask]
    x = np.tile(x[np.newaxis, :], [t.shape[0], 1])

    y = y.ravel()
    y = y[final_mask]
    y = np.tile(y[np.newaxis, :], [t.shape[0], 1])

    z = np.zeros_like(x)

    x0 = x.copy()
    y0 = y.copy()
    z0 = z.copy()

    max_displace = np.hypot(dx, dy).max()
    displace_lim = 10/256
    descaler = 1.0
    if max_displace > displace_lim:
        descaler = displace_lim / max_displace
        dx *= descaler
        dy *= descaler

    # print(np.hypot(dx, dy).max())
    # print(x.shape, y.shape)

    # brownian_mod = 0.0
    # rrx = brownian_mod * np.random.standard_normal(x.shape)
    # rrx[0] *= 0.0
    # c_rrx = np.cumsum(rrx, axis=0)

    # rry = brownian_mod * np.random.standard_normal(y.shape)
    # rry[0] *= 0.0
    # c_rry = np.cumsum(rry, axis=0)

    x += dx 
    y += dy


    r = np.stack((x, y, z), 2)
    if rseed is not None:
        np.random.seed(rseed)
    t1 = map_1Dpoly(s, t1_lims)
    t2 = map_1Dpoly(s, t2_lims)
    if inflow_mask is not None:
        s[inflow_lin > 0] = np.random.uniform(0.20, 0.50)
        t1[inflow_lin > 0] = np.random.uniform(30, 60)
        t2[inflow_lin > 0] = np.random.uniform(10, 20)

    return r, s, t1, t2, final_mask0, r_a0, r_b0, theta0, t0, img, inflow_mask, xmod0, ymod0, descaler
