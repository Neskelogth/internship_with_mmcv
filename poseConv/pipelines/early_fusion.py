from ..dataset.builder import PIPELINES
import torch
import matplotlib.pyplot as plt
import numpy as np


@PIPELINES.register_module()
class StackFrames:

    def __call__(self, results):

        # print(results.keys())

        # from frames x w x h x channels to frames x channels x w x h
        new_imgs = np.transpose(results['imgs'], axes=(0, 3, 1, 2))

        total_channels = results['heatmap_imgs'].shape[1] + new_imgs.shape[1]
        final_imgs = np.empty((new_imgs.shape[0], total_channels, new_imgs.shape[2], new_imgs.shape[3]))

        for i in range(new_imgs.shape[0]):
            final_imgs[i, :results['heatmap_imgs'].shape[1], :, :] = results['heatmap_imgs'][i]
            final_imgs[i, -new_imgs.shape[1]:, :, :] = new_imgs[i]

        final_imgs = np.transpose(final_imgs, axes=(1, 0, 2, 3))

        results['imgs'] = final_imgs


        return results

    def __repr__(self):
        return f'{self.__class__.__name__}'
