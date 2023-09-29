import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class RGBPoseHeadCat(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 dropout=0.5,
                 loss_cls=None,
                 init_std=0.01,
                 temporal=False,
                 **kwargs):

        if loss_cls is None:
            loss_cls = dict(type='CrossEntropyLoss')
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout = dropout
        self.init_std = init_std
        self.dropout_layer = None
        self.pooling_layer = nn.AdaptiveAvgPool3d(1)
        self.temporal = temporal
        self.loss_weights = [1.]
        self.loss_components = ['total']

        if self.dropout != 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        self.fc_layer = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc_layer, std=self.init_std)

    def forward(self, x):

        x_rgb, x_pose = x[0], x[1]

        if self.temporal:
            x_rgb = torch.transpose(torch.transpose(x_rgb, 2, 0), 1, 2)
            x_pose = torch.transpose(torch.transpose(x_pose, 2, 0), 1, 2)

            assert x_rgb.size()[0] == x_pose.size()[0], ('The number of frames selected in the RGB stream and '
                                                         'the Pose stream should be the same')
            assert x_rgb.size()[1] == x_pose.size()[1], ('The number of people in the frames selected in the RGB '
                                                         'stream and the Pose stream should be the same')

            x = None

            frame_number = len(x_pose)
            people_number = len(x_pose[0])

            for frame in range(frame_number):
                frame_features = None
                for person in range(people_number):
                    person_features = torch.cat((x_pose[frame, person], x_rgb[frame, person]))
                    person_features = person_features.view((1, ) + person_features.shape)
                    if frame_features is None:
                        frame_features = person_features
                    else:
                        frame_features = torch.cat((frame_features, person_features))
                frame_features = frame_features.view((1, ) + frame_features.shape)
                if x is None:
                    x = frame_features
                else:
                    x = torch.cat((x, frame_features))

            assert x is not None, 'The features cannot be None'
            x = torch.transpose(torch.transpose(x, 2, 0), 1, 0)
            x = self.pooling_layer(x)
            x = x.view(x.size(0), -1)

        else:
            x_rgb, x_pose = self.pooling_layer(x[0]), self.pooling_layer(x[1])
            x_rgb = x_rgb.view(x_rgb.size(0), -1)
            x_pose = x_pose.view(x_pose.size(0), -1)
            x = torch.cat((x_pose, x_rgb), dim=-1)

        assert x.shape[1] == self.in_channels, (f'The number of channels should be '
                                                f'{self.in_channels}, found {x.shape[1]}')

        if self.dropout_layer is not None:
            x = self.dropout_layer(x)

        scores = self.fc_layer(x)
        return scores
