import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from matplotlib import animation, rc
from IPython.display import HTML


# Jupyter notebook compatible animation
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
