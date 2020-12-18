import numpy as np

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def roundup2(x):
    return int(np.ceil(x / 2.0)) * 2


def roundup4(x):
    return int(np.ceil(x / 2.0)) * 2


def zeropad(A, out_shape, use_gpu=False):
    out_shape = np.array(out_shape)
    in_shape = np.array(A.shape)

    # This checks for an A with more dimensions, but
    # the rest of the function can't handle it
    diff_shape = out_shape - in_shape[-len(out_shape) :]
    pad0 = diff_shape // 2
    pad1 = diff_shape - pad0

    # No zeropad needed
    if (pad1 == 0).sum() == len(pad1):
        return A

    if len(out_shape) != len(in_shape):
        print("ERROR: Cant zeropad with unequal dimensions")

    # We could instead just check if A is on the gpu...
    if use_gpu:
        out = cp.zeros(out_shape, A.dtype)
    else:
        out = np.zeros(out_shape, A.dtype)

    if len(out_shape) == 2:
        out[pad0[0] : -pad1[0], pad0[1] : -pad1[1]] = A
    elif len(out_shape) == 3:
        out[pad0[0] : -pad1[0], pad0[1] : -pad1[1], pad0[2] : -pad1[2]] = A

    return out


def crop(A, out_shape, use_gpu=False):
    out_shape = np.array(out_shape)
    in_shape = np.array(A.shape)

    # This checks for an A with more dimensions, but
    # the rest of the function can't handle it
    diff_shape = in_shape[-len(out_shape) :] - out_shape
    pad0 = diff_shape // 2
    pad1 = diff_shape - pad0

    # No crop needed
    if (pad1 == 0).sum() == len(pad1):
        return A

    if len(out_shape) != len(in_shape):
        print("ERROR: Cant zeropad with unequal dimensions")

    # We could instead just check if A is on the gpu...
    if use_gpu:
        out = cp.zeros(out_shape, A.dtype)
    else:
        out = np.zeros(out_shape, A.dtype)

    if len(out_shape) == 2:
        out = A[pad0[0] : -pad1[0], pad0[1] : -pad1[1]]
    elif len(out_shape) == 3:
        out = A[pad0[0] : -pad1[0], pad0[1] : -pad1[1], pad0[2] : -pad1[2]]

    return out


def check_traj_dens(traj, dens):
    """Does a couple checks to make sure the trajectory and density arrays
        are shaped correctly.

        Args:
            traj (float ndarray): trajectory, will eventually get reshaped to (Nx3)
            dens (float ndarray): density, will eventually get reshaped to (Nx1)

        Return traj and dens arrays int he correct format
        """
    # Check types
    if traj.dtype != np.float32:
        traj = traj.astype(np.float32)

    if dens.dtype != np.float32:
        dens = dens.astype(np.float32)

    # Transpose into contiguous memory
    if traj.shape[0] == 2 or traj.shape[0] == 3:
        traj = np.ascontiguousarray(traj.T)

    # Add a dimension of zeros if traj only has kx, ky
    if traj.shape[-1] == 2:
        traj = np.concatenate((traj, np.zeros_like(traj[..., -1:])), axis=traj.ndim - 1)

    # Flatten both arrays
    traj = np.reshape(traj, (-1, 3))
    dens = dens.ravel()

    if traj.shape[0] != len(dens):
        print("ERROR: traj and dens don't have matching sizes")

    return traj, dens
