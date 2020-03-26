"""
File: vis_utils.py
Author: Nrupatunga
Email: nrupatunga.tunga@gmail.com
Github: https://github.com/nrupatunga
Description: visualization
"""
import numpy as np
import visdom


class Visualizer():

    def __init__(self, env='default', port=8097, **kwargs):
        self.vis = visdom.Visdom(env=env, port=port, **kwargs)
        self.index = 1

    def plot_curves(self, d, iters, title='loss', xlabel='iters',
                    ylabel='accuracy', env='curves'):
        self.index = iters
        name = list(d.keys())
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        self.vis.line(Y=y,
                      X=np.array([self.index]),
                      win=title, env=env,
                      opts=dict(legend=name, title=title, xlabel=xlabel, ylabel=ylabel),
                      update=None if self.index == 0 else 'append')

    def plot_hist(self, d, iters, title='loss', xlabel='iters', ylabel='accuracy'):
        name = list(d.keys())
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        self.vis.line(Y=y,
                      X=np.array([self.index]),
                      win=title,
                      opts=dict(legend=name, title=title, xlabel=xlabel, ylabel=ylabel),
                      update=None if self.index == 0 else 'append')
        self.index = iters

    def plot_image_np(self, image, title, env='images'):
        self.vis.image(image, win=title, env=env, opts=dict(title=title))

    def plot_image_opencv(self, bgr, title, env='images'):
        if len(bgr.shape) == 2:
            rgb = bgr
        else:
            rgb = bgr[..., ::-1]
            rgb = np.transpose(rgb, axes=[2, 0, 1])
        self.vis.image(rgb, win=title, env=env, opts=dict(title=title))

    def plot_images_np(self, images, title, env='images'):
        self.vis.images(images, win=title, env=env, opts=dict(title=title))

    def plot_images_plt(self, plt, title, env='images'):
        self.vis.matplot(plt, win=title, env=env, opts=dict(title=title))
