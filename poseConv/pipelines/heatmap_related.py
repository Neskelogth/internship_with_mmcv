# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from ..dataset.builder import PIPELINES


@PIPELINES.register_module()
class GeneratePoseTarget:
    """Generate pseudo heatmaps based on joint coordinates and confidence.

    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".

    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
    """

    def __init__(self,
                 sigma=0.6,
                 use_score=True,
                 double=False,
                 left_kp=(1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp=(2, 4, 6, 8, 10, 12, 14, 16),
                 scaling=1.,
                 eps=1e-3):

        self.sigma = sigma
        self.use_score = use_score
        self.double = double

        self.left_kp = left_kp
        self.right_kp = right_kp
        self.scaling = scaling
        self.eps = eps

    def generate_a_heatmap(self, arr, centers, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: img_h * img_w.
            centers (np.ndarray): The coordinates of corresponding keypoints (of multiple persons). Shape: M * 2.
            max_values (np.ndarray): The max values of each keypoint. Shape: M.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for center, max_value in zip(centers, max_values):
            if max_value < self.eps:
                continue

            mu_x, mu_y = center[0], center[1]
            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / 2 / sigma ** 2)
            patch = patch * max_value
            arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

    def generate_heatmap(self, arr, kps, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            arr (np.ndarray): The array to store the generated heatmaps. Shape: V * img_h * img_w.
            kps (np.ndarray): The coordinates of keypoints in this frame. Shape: M * V * 2.
            max_values (np.ndarray): The confidence score of each keypoint. Shape: M * V.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        num_kp = kps.shape[1]
        for i in range(num_kp):
            self.generate_a_heatmap(arr[i], kps[:, i], max_values[:, i])

    def gen_an_aug(self, results):
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results['keypoint']
        kp_shape = all_kps.shape

        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results['img_shape']

        # scale img_h, img_w and kps
        img_h = int(img_h * self.scaling + 0.5)
        img_w = int(img_w * self.scaling + 0.5)
        all_kps[..., :2] *= self.scaling

        num_frame = kp_shape[1]
        num_c = all_kps.shape[2]
        ret = np.zeros([num_frame, num_c, img_h, img_w], dtype=np.float32)

        for i in range(num_frame):
            # M, V, C
            kps = all_kps[:, i]
            # M, C
            kpscores = all_kpscores[:, i] if self.use_score else np.ones_like(all_kpscores[:, i])

            self.generate_heatmap(ret[i], kps, kpscores)
        return ret

    def __call__(self, results):
        heatmap = self.gen_an_aug(results)
        key = 'heatmap_imgs' if 'imgs' in results else 'imgs'

        # if self.double:
        #     indices = np.arange(heatmap.shape[1], dtype=np.int64)
        #     left, right = (self.left_kp, self.right_kp)
        #     for l, r in zip(left, right):  # noqa: E741
        #         indices[l] = r
        #         indices[r] = l
        #     heatmap_flip = heatmap[..., ::-1][:, indices]
        #     heatmap = np.concatenate([heatmap, heatmap_flip])
        results[key] = heatmap
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp})')
        return repr_str

# The Input will be a feature map ((N x T) x H x W x K), The output will be
# a 2D map: (N x H x W x [K * (2C + 1)])
# N is #clips x #crops, K is num_kpt
# @PIPELINES.register_module()
# class Heatmap2Potion:
#
#     def __init__(self, C, option='full'):
#         self.C = C
#         self.option = option
#         self.eps = 1e-4
#         assert isinstance(C, int)
#         assert C >= 2
#         assert self.option in ['U', 'N', 'I', 'full']
#
#     def __call__(self, results):
#         heatmaps = results['imgs']
#
#         if 'clip_len' in results:
#             clip_len = results['clip_len']
#         else:
#             # Just for Video-PoTion generation
#             clip_len = heatmaps.shape[0]
#
#         C = self.C
#         heatmaps = heatmaps.reshape((-1, clip_len) + heatmaps.shape[1:])
#         # num_clip, clip_len, C, H, W
#         heatmaps = heatmaps.transpose(0, 1, 3, 4, 2)
#
#         # t in {0, 1, 2, ..., clip_len - 1}
#         def idx2color(t):
#             st = np.zeros(C, dtype=np.float32)
#             ed = np.zeros(C, dtype=np.float32)
#             if t == clip_len - 1:
#                 ed[C - 1] = 1.
#                 return ed
#             val = t / (clip_len - 1) * (C - 1)
#             bin_idx = int(val)
#             val = val - bin_idx
#             st[bin_idx] = 1.
#             ed[bin_idx + 1] = 1.
#             return (1 - val) * st + val * ed
#
#         heatmaps_wcolor = []
#         for i in range(clip_len):
#             color = idx2color(i)
#             heatmap = heatmaps[:, i]
#             heatmap = heatmap[..., None]
#             heatmap = np.matmul(heatmap, color[None, ])
#             heatmaps_wcolor.append(heatmap)
#
#         # The shape of each element is N x H x W x K x C
#         heatmap_S = np.sum(heatmaps_wcolor, axis=0)
#         # The shape of U_norm is N x 1 x 1 x K x C
#         U_norm = np.max(
#             np.max(heatmap_S, axis=1, keepdims=True), axis=2, keepdims=True)
#         heatmap_U = heatmap_S / (U_norm + self.eps)
#         heatmap_I = np.sum(heatmap_U, axis=-1, keepdims=True)
#         heatmap_N = heatmap_U / (heatmap_I + 1)
#         if self.option == 'U':
#             heatmap = heatmap_U
#         elif self.option == 'I':
#             heatmap = heatmap_I
#         elif self.option == 'N':
#             heatmap = heatmap_N
#         elif self.option == 'full':
#             heatmap = np.concatenate([heatmap_U, heatmap_I, heatmap_N],
#                                      axis=-1)
#
#         # Reshape the heatmap to 4D
#         heatmap = heatmap.reshape(heatmap.shape[:3] + (-1, ))
#         results['imgs'] = heatmap
#         return results
