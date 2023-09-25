import mmcv
import os
import json

import numpy as np

from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS

from tqdm import tqdm


def adjust_missing_people(kps, scores, people, frames):

    if len(list(set(people))) == 1:
        return kps.reshape((frames, -1, 17, 2)), scores.reshape(frames, -1, 17)

    # Different frames may have different numbers of people
    max_people = max(people)
    new_kps = np.array(list())
    new_scores = np.array(list())
    scores_idx = 0
    idx = 0
    for frame_people in people:
        points_number = 17 * frame_people
        kps_number = points_number * 2
        partial_kps = kps[idx: idx + kps_number]
        partial_scores = scores[scores_idx: scores_idx + points_number]
        scores_idx += points_number
        idx += kps_number
        if max_people - frame_people > 0:
            scores_to_add = np.array([0] * ((max_people - frame_people) * 17))
            kps_to_add = np.array([0] * ((max_people - frame_people) * 17 * 2))

            partial_scores = np.append(partial_scores, scores_to_add)
            partial_kps = np.append(partial_kps, kps_to_add)

        new_kps = np.append(new_kps, partial_kps)
        new_scores = np.append(new_scores, partial_scores)

    new_kps = new_kps.reshape((frames, max_people, 17, 2))
    new_scores = new_scores.reshape((frames, max_people, 17))

    return new_kps, new_scores


def handle_person_kps(person):

    person = person.reshape((25, 3))
    confidences = np.empty((17,))
    kps = np.empty((17, 2))

    kps[0, 0], kps[0, 1] = person[0, 0], person[0, 1]
    confidences[0] = person[0, 2]

    kps[1, 0], kps[1, 1] = person[16, 0], person[16, 1]
    confidences[1] = person[16, 2]
    kps[2, 0], kps[2, 1] = person[15, 0], person[15, 1]
    confidences[2] = person[15, 2]

    kps[3, 0], kps[3, 1] = person[18, 0], person[18, 1]
    confidences[3] = person[18, 2]
    kps[4, 0], kps[4, 1] = person[17, 0], person[17, 1]
    confidences[4] = person[17, 2]

    kps[5, 0], kps[5, 1] = person[5, 0], person[5, 1]
    confidences[5] = person[5, 2]
    kps[6, 0], kps[6, 1] = person[2, 0], person[2, 1]
    confidences[6] = person[2, 2]

    kps[7, 0], kps[7, 1] = person[6, 0], person[6, 1]
    confidences[7] = person[6, 2]
    kps[8, 0], kps[8, 1] = person[3, 0], person[3, 1]
    confidences[8] = person[3, 2]

    kps[9, 0], kps[9, 1] = person[7, 0], person[7, 1]
    confidences[9] = person[7, 2]
    kps[10, 0], kps[10, 1] = person[4, 0], person[4, 1]
    confidences[10] = person[4, 2]

    kps[11, 0], kps[11, 1] = person[12, 0], person[12, 1]
    confidences[11] = person[12, 2]
    kps[12, 0], kps[12, 1] = person[9, 0], person[9, 1]
    confidences[12] = person[9, 2]

    kps[13, 0], kps[13, 1] = person[13, 0], person[13, 1]
    confidences[13] = person[13, 2]
    kps[14, 0], kps[14, 1] = person[10, 0], person[10, 1]
    confidences[14] = person[10, 2]

    kps[15, 0], kps[15, 1] = person[14, 0], person[14, 1]
    confidences[15] = person[14, 2]
    kps[16, 0], kps[16, 1] = person[11, 0], person[11, 1]
    confidences[16] = person[11, 2]

    return kps, confidences


def body_25_to_coco(kps):

    new_kps = np.empty((kps.shape[0], 17, 2))
    kps_scores = np.empty((kps.shape[0], 17))

    for i in range(len(kps)):
        new_kps[i], kps_scores[i] = handle_person_kps(kps[i])

    return new_kps, kps_scores


@DATASETS.register_module()
class PoseDataset(BaseDataset):

    def __init__(self,
                 ann_file,  # Is a directory in case the json format is used
                 pipeline,
                 split=None,
                 valid_ratio=None,
                 box_thr=None,
                 class_prob=None,
                 origin='pkl',
                 **kwargs):

        self.split = split
        self.origin = origin
        super().__init__(ann_file, pipeline, start_index=0, modality='Pose', **kwargs)

        # box_thr, which should be a string
        self.box_thr = box_thr
        self.class_prob = class_prob
        if self.box_thr is not None:
            assert box_thr in [.5, .6, .7, .8, .9]

        # Thresholding Training Examples
        self.valid_ratio = valid_ratio
        if self.valid_ratio is not None and isinstance(self.valid_ratio, float) and self.valid_ratio > 0:
            self.video_infos = [
                x for x in self.video_infos
                if x['valid'][self.box_thr] / x['total_frames'] >= valid_ratio
            ]
            for item in self.video_infos:
                assert 'box_score' in item, 'if valid_ratio is a positive number, item should have field `box_score`'
                anno_inds = (item['box_score'] >= self.box_thr)
                item['anno_inds'] = anno_inds
        for item in self.video_infos:
            item.pop('valid', None)
            item.pop('box_score', None)

        logger = get_root_logger()
        logger.info(f'{len(self)} videos remain after valid thresholding')

    def load_annotations(self):

        if self.origin == 'pkl':
            return self.load_pkl_annotations()
        else:
            return self.load_json_annotations_skeletons()


    def load_pkl_annotations(self):
        data = mmcv.load(self.ann)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'frame_dir'
            split = set(split[self.split])
            data = [x for x in data if x[identifier] in split]
            if 'train' not in self.split:
                data = data[:4000]
            else:
                # data = [item for item in data if item['frame_dir'] == 'S006C001P019R001A022']
                data = data[:10000]

        for item in data:
            item['frame_dir'] = os.path.join(self.data_prefix, item['frame_dir'])

        return data

    def load_json_annotations_skeletons(self):

        results = list()

        split_clips = open(os.path.join('./splits/', self.split + '.txt')).readlines()
        split_clips = [item.strip() for item in split_clips]
        folder_list = os.listdir(self.ann)
        folder_list = [item.replace('_rgb', '') for item in folder_list]
        folder_list = [file for file in folder_list if file in split_clips]
        if 'train' not in self.split:
            folder_list = folder_list[:4000]
        else:
            folder_list = folder_list[:10000]

        for folder in tqdm(folder_list):
            result = dict()
            folder_path = os.path.join(self.ann, folder + '_rgb')
            frame_list = os.listdir(folder_path)
            frame_list.sort()
            result['frame_dir'] = folder
            result['total_frames'] = len(frame_list)
            result['original_shape'] = (1080, 1920)
            result['img_shape'] = (1080, 1920)
            result['label'] = int(folder[-3:]) - 1
            result['keypoint'] = np.array(list())
            result['keypoint_score'] = np.array(list())
            people = list()
            for frame in frame_list:
                frame_path = os.path.join(folder_path, frame)
                body_25_skeletons = json.load(open(frame_path, 'r'))['people']
                kps = list()
                for person in body_25_skeletons:
                    kps.append(person['pose_keypoints_2d'])

                # shape of the body25 json output: people x 25 x 3
                # (25 keypoints, x, y and confidence for each one of them)
                kps = np.array(kps).reshape((-1, 25, 3))
                kps, confidences = body_25_to_coco(kps)
                # print(kps.shape, confidences.shape, len(result['keypoint']))
                people.append(kps.shape[0])
                result['keypoint'] = np.append(result['keypoint'], kps)
                result['keypoint_score'] = np.append(result['keypoint_score'], confidences)

            result['keypoint'], result['keypoint_score'] = adjust_missing_people(result['keypoint'],
                                                                                 result['keypoint_score'],
                                                                                 people, len(frame_list))

            result['keypoint'] = result['keypoint'].reshape(len(frame_list), -1, 17, 2)
            result['keypoint_score'] = result['keypoint'].reshape(len(frame_list), -1, 17)
            result['keypoint_score'] = np.transpose(result['keypoint_score'], (1, 0, 2))
            result['keypoint'] = np.transpose(result['keypoint'], (1, 0, 2, 3))
            results.append(result)

        for item in results:
            item['frame_dir'] = os.path.join(self.data_prefix, item['frame_dir'])

        return results
