from ..dataset.builder import PIPELINES
import torch


@PIPELINES.register_module()
class StackFrames:
    def __init__(self, all_frames=True):
        self.all_frames = all_frames

    def __call__(self, rgb_frames, pose_frames):

        assert rgb_frames.shape == pose_frames.shape, ('The frames in rgb modality and pose '
                                                       'modality must have the same shape')

        if self.all_frames:
            return torch.stack((rgb_frames, pose_frames))



    def __repr__(self):
        return f'{self.__class__.__name__} (one by one: {self.all_frames})'
