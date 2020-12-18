import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from matplotlib import animation, rc
from IPython.display import HTML

import pandas as pd
import seaborn as sns
import torch

# Make plots to quickly compare model data to the reference data
def get_ref_plots(ref_data, model, device, batch_size = 16, dpi = 200):
    
    model.eval()
        

    all_y1 = []

    N = ref_data['x'].shape[0]
    N_batches = N // batch_size + 1

    with torch.no_grad():  
        for i in range(N_batches):
            x = ref_data['x'][i*batch_size:(i+1)*batch_size]
            y_true = ref_data['y'][i*batch_size:(i+1)*batch_size]

            x = torch.from_numpy(x).to(device)
            y_true = torch.from_numpy(y_true).to(device)
    #         print(x.shape)

            y_pred = model(x)

            all_y1.append(y_pred.detach().cpu().numpy())

    all_y1 = np.vstack(all_y1)
    all_y0 = ref_data['y']

    all_y0 = all_y0.reshape([all_y0.shape[0], 2, 25])
    all_y1 = all_y1.reshape([all_y1.shape[0], 2, 25])


    ####### Plot 1
    sns.set_context('poster')
    sns.set_style("ticks")
    fig = plt.figure(figsize=(12,12), dpi = dpi)
    plt.scatter(all_y0.ravel(), all_y1.ravel(), marker='x', color='r', alpha=0.2)
    plt.plot([-10, 10], [-10, 10])
    plt.axhline(color = '0.5', linestyle=':', zorder = 0)
    plt.axvline(color = '0.5', linestyle=':', zorder = 0)

    fig.tight_layout()

    # To remove the huge white borders
    # fig.gca().margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    regression1 = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    ####### Plot 2
    fig = plt.figure(figsize=(12,12), dpi = dpi)

    plt.hexbin(all_y0.ravel(), all_y1.ravel(), gridsize = 40, cmap='Reds', bins='log', extent=(-10,10,-10,10))
    plt.ylim(-10,10)
    plt.xlim(-10,10)
    plt.plot([-10, 10], [-10, 10], 'g:')

    fig.tight_layout()

    # To remove the huge white borders
    # fig.gca().margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    regression2 = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    ####### Plot 3
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(10,8), dpi = dpi)

    diff = all_y0 - all_y1
    diff = np.hypot(diff[:,0], diff[:,1])

    df = pd.DataFrame(diff).melt()

    sns.lineplot(x="variable", y="value", data=df, ci='sd')
    plt.ylim((0,4))

    fig.tight_layout()

    # To remove the huge white borders
    # fig.gca().margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    regression3 = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()


    ####### Plot 4
    sns.set_context('poster')
    sns.set_style("ticks")

    fig = plt.figure(figsize=(10,6), dpi = dpi)

    data0, data1 = all_y0.ravel(), all_y1.ravel()

    mean      = np.mean([data0, data1], axis=0)
    diff      = data0 - data1                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, marker='x', color='#4527a0', alpha=0.1)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle=':')
    plt.axhline(md - 1.96*sd, color='gray', linestyle=':')

    plt.xlim(-10, 10)
    plt.ylim(-4, 4)

    sns.despine()

    plt.gca().annotate('{:.2f}'.format(np.round(md, 2)),
                    xy=(12, md),
                    horizontalalignment='right',
                    verticalalignment='center',
                    fontsize=24,
                    xycoords='data',
                    annotation_clip=False)

    plt.gca().annotate('{:.2f}'.format(np.round(md + 1.96*sd, 2)),
                    xy=(12, md + 1.96*sd),
                    horizontalalignment='right',
                    verticalalignment='center',
                    fontsize=24,
                    xycoords='data',
                    annotation_clip=False)

    plt.gca().annotate('{:.2f}'.format(np.round(md - 1.96*sd, 2)),
                    xy=(12, md - 1.96*sd),
                    horizontalalignment='right',
                    verticalalignment='center',
                    fontsize=24,
                    xycoords='data',
                    annotation_clip=False)

    fig.tight_layout()

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    regression4 = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()


    image = Image.fromarray(regression1, 'RGB')
    image.thumbnail((512,512), Image.BICUBIC)
    regression1 = np.array(image).transpose([2,0,1])

    image = Image.fromarray(regression2, 'RGB')
    image.thumbnail((512,512), Image.BICUBIC)
    regression2 = np.array(image).transpose([2,0,1])

    image = Image.fromarray(regression3, 'RGB')
    image.thumbnail((512,512), Image.BICUBIC)
    regression3 = np.array(image).transpose([2,0,1])

    image = Image.fromarray(regression4, 'RGB')
    image.thumbnail((512,512), Image.BICUBIC)
    regression4 = np.array(image).transpose([2,0,1])

    return regression1, regression2, regression3, regression4

# Make an figure with image data, and 1 or 2 tracks overlayed on top
def plot_lines(ims, tags0, tags1 = None, figsize=[8, 8], nframes=25, scale = 1.0, shift = None):
    ims = np.squeeze(ims)
    
    if shift is None:
            shift = ims.shape[-1]/2.0
    
    fig, axarr = plt.subplots(1, 1, squeeze=False, figsize=figsize)
#     fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.subplots_adjust(0, 0, 1, 1)
    im = axarr[0, 0].imshow(ims[0], cmap="gray")

    tags0 = np.squeeze(np.array(tags0))
    tagmid = tags0.size // 2
    xp = np.array(tags0[:tagmid]) * scale + shift
    yp = np.array(tags0[tagmid:]) * scale + shift

    pts_all, = axarr[0, 0].plot(
        xp, yp, color='g', marker="x", markeredgecolor="g", mew=2, markersize=6
    )
    
    if tags1 is not None:
        tags1 = np.squeeze(np.array(tags1))
        tagmid = tags1.size // 2
        xp = np.array(tags1[:tagmid]) * scale + shift
        yp = np.array(tags1[tagmid:]) * scale + shift

        pts_all, = axarr[0, 0].plot(
            xp, yp, color='r', marker="x", markeredgecolor="r", mew=2, markersize=6
        )
    
    axarr[0, 0].axes.get_xaxis().set_visible(False)
    axarr[0, 0].axes.get_yaxis().set_visible(False)
    axarr[0, 0].set_frame_on(False)

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
