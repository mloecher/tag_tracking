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

class Animate_FullIm:
    def __init__(self, ims, tags_in=None, tags_in2=None, figsize=(8,8), dpi=100, clim_scale=(0.0, 1.0), interval=50, auto_crop = 10):
        print("Starting animation class . . . ", flush=True)
        
        self.ims = np.squeeze(ims)
        Nt = self.ims.shape[0]
        
        if tags_in is None:
            self.tags = tags_in
            self.plot_tags = False
        else:
            tags = np.squeeze(tags_in)
            if not tags.ndim == 3:
                print('ERROR: tags should be 3 dims, not', tags.ndim)
                return
            
            tags_shape = np.array(tags.shape)

            idx_dim = np.argwhere(tags_shape==2)[0][0]
            idx_time = np.argwhere(tags_shape==25)[0][0]

            idx_pts = list(range(tags.ndim))
            idx_pts.remove(idx_dim)
            idx_pts.remove(idx_time)
            idx_pts = idx_pts[0]
            
            tags = tags.transpose([idx_time, idx_dim, idx_pts])

            self.tags = tags
            self.plot_tags = True

        if tags_in2 is None:
            self.tags2 = tags_in2
            self.plot_tags2 = False
        else:
            tags2 = np.squeeze(tags_in2)
            if not tags2.ndim == 3:
                print('ERROR: tags should be 3 dims, not', tags2.ndim)
                return
            
            tags2_shape = np.array(tags2.shape)

            idx_dim2 = np.argwhere(tags2_shape==2)[0][0]
            idx_time2 = np.argwhere(tags2_shape==25)[0][0]

            idx_pts2 = list(range(tags2.ndim))
            idx_pts2.remove(idx_dim2)
            idx_pts2.remove(idx_time2)
            idx_pts2 = idx_pts2[0]
            
            tags2 = tags2.transpose([idx_time2, idx_dim2, idx_pts2])

            self.tags2 = tags2
            self.plot_tags2 = True

        self.fig, self.axarr = plt.subplots(1, 1, squeeze=False, figsize=figsize)
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        clim = [self.ims.min() + clim_scale[0] * self.ims.max(), clim_scale[1] * self.ims.max()]
        self.im = self.axarr[0, 0].imshow(self.ims[0], cmap="gray", clim=clim, extent=(0, self.ims.shape[2], self.ims.shape[1], 0))
        
        if auto_crop is not None:
            self.axarr[0, 0].set_xlim([tags[:,0,:].min()-auto_crop, tags[:,0,:].max()+auto_crop])
            self.axarr[0, 0].set_ylim([tags[:,1,:].max()+auto_crop, tags[:,1,:].min()-auto_crop])
        
        if self.plot_tags:          
            self.pts_plot, = self.axarr[0, 0].plot(
                self.tags[0,0], self.tags[0,1], linestyle="None", 
                marker="+", fillstyle='full', markeredgecolor="#ff1744", markerfacecolor ='#ff1744',mew=3, markersize=15
            )
        else:
            self.pts_plot = None

        if self.plot_tags2:          
            self.pts_plot2, = self.axarr[0, 0].plot(
                self.tags2[0,0], self.tags2[0,1], linestyle="None", 
                marker="x", fillstyle='full', markeredgecolor="#76ff03", markerfacecolor ='#76ff03',mew=3, markersize=10
            )
        else:
            self.pts_plot2 = None

        print("Making animation . . . ", flush=True)
        self.anim = animation.FuncAnimation(
            self.fig,
            self.animate,
            init_func=self.init,
            frames=Nt,
            interval=interval,
            blit=True,
        )
        
        plt.close()

    def init(self):
        self.im.set_data(self.ims[0])
        
        out = [self.im,]

        if self.plot_tags:
            self.pts_plot.set_data(self.tags[0,0], self.tags[0,1])
            out.append(self.pts_plot)

        if self.plot_tags2:
            self.pts_plot2.set_data(self.tags2[0,0], self.tags2[0,1])
            out.append(self.pts_plot2)


        return out

    def animate(self, i):
        self.im.set_data(self.ims[i])

        out = [self.im,]

        if self.plot_tags:
            self.pts_plot.set_data(self.tags[i,0], self.tags[i,1])
            out.append(self.pts_plot)

        if self.plot_tags2:
            self.pts_plot2.set_data(self.tags2[i,0], self.tags2[i,1])
            out.append(self.pts_plot2)

        return out