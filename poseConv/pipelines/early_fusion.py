from ..dataset.builder import PIPELINES
import numpy as np
import matplotlib.pyplot as plt


@PIPELINES.register_module()
class StackFrames:

    def __call__(self, results):

        # from frames x w x h x channels to frames x channels x w x h
        # print(results['imgs'].shape, results['heatmap_imgs'].shape)
        new_imgs = np.transpose(results['imgs'], axes=(2, 1, 0, 3, 4))
        new_hms = np.transpose(results['heatmap_imgs'], axes=(2, 1, 0, 3, 4))

        total_channels = new_imgs.shape[1] + new_hms.shape[1]
        total_frames = new_imgs.shape[0]
        # print(total_channels, total_frames)
        final_imgs = np.empty((total_frames, total_channels) + new_imgs.shape[2:])

        for i in range(total_frames):
            final_imgs[i, :new_hms.shape[1], :, :, :] = new_hms[i]
            final_imgs[i, -new_imgs.shape[1]:, :, :, :] = new_imgs[i]

        final_imgs = np.transpose(final_imgs, axes=(2, 1, 0, 3, 4))
        results['imgs'] = final_imgs

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}'
