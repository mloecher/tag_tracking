import torch
import h5py
import numpy as np

# Data generator, with augmentations

class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_idx, data_filename, dim=(32, 32), path_scale = 256, 
                debug = False, normmode=0, output_index = False, do_augs = False):
        self.list_idx = list_idx
        self.data_filename = data_filename
        self.dim = dim
        self.path_scale = path_scale
        self.debug = debug
        self.normmode = normmode
        self.output_index = output_index
        self.do_augs = do_augs

        # self.data_file = h5py.File(self.data_filename, 'r')

    def shuffle(self):
        self.list_idx = np.random.permutation(self.list_idx)

    def __len__(self):
        return len(self.list_idx)

    def __getitem__(self, index):
        # Reloading the h5py each time is necessary to allow multiprocessing.
        # It still seems hast with this in use, not much overhead        
        data_file = h5py.File(self.data_filename, 'r')

        ID = self.list_idx[index]

        # if self.debug:
            # print('dataset getitem:', index, ID)

        X = np.empty((1, 25, *self.dim), np.float32)

        tpaths = data_file['patch_paths'][ID, :, :].ravel()
        y = tpaths * self.path_scale

        im0 = data_file['patch_ims'][ID, :, :, :]

        if self.do_augs:
            im0, y = do_all_augs(im0, y)
        
        if self.normmode == 0:
            im0 = ((im0 - im0.min()) / (im0.max() - im0.min())) * 2.0 - 1.0
        elif self.normmode == 1:
            im0 = (im0 - im0.mean()) / im0.std()

        X[0] = im0


        data_file.close()

        if self.output_index:
            return X, y, ID
        else:
            return X, y


def aug_rotate(X, Y, k = 1):
    X1 = np.rot90(X, k, axes=(1,2))
    rot_idx = np.roll(np.arange(50), 25)
    Y1 = Y.copy()
    for i_ in range(k):
        Y1 = Y1[rot_idx]
        Y1[25:] *= -1.0
    return X1, Y1

def aug_flip(X, Y, k = 1):
    X1 = X.copy()
    Y1 = Y.copy()
    if k == 0:
        pass
    elif k == 1:
        X1 = np.flip(X1, k)
        Y1[25:] *= -1.0
    elif k == 2:
        X1 = np.flip(X1, k)
        Y1[:25] *= -1.0
        
    return X1, Y1

def aug_linear(X, Y, axis = 0, scale = 0.5):
    X1 = X.copy()
    Y1 = Y.copy()
    # 0.3 here is just so scale in range of 0 to 1 makes sense
    mod = 0.3 * X1.mean() * np.linspace(-scale, scale, X1.shape[axis])
    
    if axis == 0:
        X1 += mod[:, None, None]
    elif axis == 1:
        X1 += mod[None, :, None]
    elif axis == 2:
        X1 += mod[None, None, :]
        
    return X1, Y1

def aug_noise(X, Y, scale = 0.5):
    X1 = X.copy()
    Y1 = Y.copy()
    # 0.1 here is just so scale in range of 0 to 1 makes sense
    mod = 0.1 * scale * X1.mean() * np.random.standard_normal(X1.shape)
    X1 += mod
        
    return X1, Y1

def aug_temp_blur(X, Y, tf = 0, scale = 1.0):
    X1 = X.copy()
    Y1 = Y.copy()
    
    if tf > 22:
        tf = 22
        
    X1[tf] = X1[tf] + scale*(X1[tf-1] + X1[tf+1])  + scale/2.0*(X1[tf-2] + X1[tf+2])
    X1[tf] /= (1.0 + 3.0*scale)
    
    return X1, Y1

from tagsim_git.interp_temp2d import interpolate_temp2D, interpolate_temp

def aug_temp_interp(X, Y, scale = 1.0):
    tt = np.arange(25).astype(np.float)
    mod = 0.3 * scale * np.random.standard_normal(25)
    mod[0] = 0
    tt += mod
    tt[tt<0] = 0
        
    X1 = interpolate_temp2D(X, tt)
    Y1 = interpolate_temp(Y, tt)
    
    return X1, Y1



def do_all_augs(XX, YY):
   
    XX, YY = aug_rotate(XX, YY, np.random.randint(4))
    
    XX, YY = aug_flip(XX, YY, np.random.randint(3))
    
    XX, YY = aug_linear(XX, YY, np.random.randint(3), np.random.rand())
    
    XX, YY = aug_noise(XX, YY, np.random.rand())
    
    XX, YY = aug_temp_blur(XX, YY, np.random.randint(25), np.random.rand())
    
    XX, YY = aug_temp_interp(XX, YY, np.random.rand())
    
    return XX, YY



def do_all_batch_aug(XXb, YYb):

    for i in range(XXb.shape[0]):
        XX, YY = do_all_augs(XXb[i, 0], YYb[i])
        XXb[i, 0] = XX
        YYb[i] = YY