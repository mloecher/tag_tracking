
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc

class TagAnimator:
    def __init__(self, ims, tags, figsize=[8, 8], nframes=25, interval=50, scale = 1.0, shift = None, clim=None):
        print("Starting animation class . . . ", flush=True)
        
        if shift is None:
            shift = ims.shape[-1]/2.0

        self.ims = np.squeeze(ims)
        

        if tags is None:
            self.tags = tags
            self.plot_tags = False
        else:
            self.tags = np.squeeze(tags)
            self.plot_tags = True

        self.fig, self.axarr = plt.subplots(1, 1, squeeze=False, figsize=figsize)
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.im = self.axarr[0, 0].imshow(self.ims[0], cmap="gray", clim=clim)

        if self.plot_tags:
            
            self.tagmid = tags.size // 2
            xp = np.array(tags[:self.tagmid]) * scale + shift
            yp = np.array(tags[self.tagmid:]) * scale + shift

            self.pts_all, = self.axarr[0, 0].plot(
                xp, yp, linestyle="None", marker="x", markeredgecolor="r", mew=2, markersize=4
            )

            self.pts_big, = self.axarr[0, 0].plot(
                xp[0], yp[0], linestyle="None", marker="+", markeredgecolor="g", mew=4, markersize=12
            )

            self.xp = xp
            self.yp = yp

        else:
            self.xp = 0
            self.yp = 0
            self.pts_big = None

        print("Making animation . . . ", flush=True)
        self.anim = animation.FuncAnimation(
            self.fig,
            self.animate,
            init_func=self.init,
            frames=nframes,
            interval=interval,
            blit=True,
        )
        
        plt.close()

    def init(self):
        self.im.set_data(self.ims[0])
        if self.plot_tags:
            self.pts_big.set_data(self.xp[0], self.yp[0])
            return [self.im, self.pts_big]
        else:
            return [self.im,]

    def animate(self, i):
        self.im.set_data(self.ims[i])
        if self.plot_tags:
            self.pts_big.set_data(self.xp[i], self.yp[i])
            return [self.im, self.pts_big]
        else:
            return [self.im,]


def get_patch_path(ims, path, is_scaled = False, width=32):
    rad = width//2
    
    if path.ndim == 1:
        path = path[:, None]

    if not is_scaled:
        p_path = (path + 0.5)
        p_path[1] *= ims.shape[-2]
        p_path[0] *= ims.shape[-1]
    else:
        p_path = path
        
    

    im_cp = np.pad(ims, pad_width=((0,0), (rad+1,rad+1), (rad+1,rad+1)), mode='constant')

    pos1 = p_path[1,0]
    ipos1 = int(pos1)

    pos0 = p_path[0,0]
    ipos0 = int(pos0)

    im_c = im_cp[:, ipos1:ipos1+2*rad+2, ipos0:ipos0+2*rad+2]

    kim_c = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(im_c, axes=(1,2)), axes=(1,2)), axes=(1,2))

    rr = 2*np.pi*np.arange(-(rad+1), rad+1)/width
    yy, xx = np.meshgrid(rr, rr, indexing='ij')

    kim_c *= np.exp(1j*xx[np.newaxis,...]*(pos0-ipos0))
    kim_c *= np.exp(1j*yy[np.newaxis,...]*(pos1-ipos1))

    im_c2 = np.abs(np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kim_c, axes=(1,2)), axes=(1,2)), axes=(1,2)))
    im_c2 = im_c2[:, 1:-1, 1:-1]
    
    c_path = path - path[:, 0][:, np.newaxis]
    
    return im_c2, c_path