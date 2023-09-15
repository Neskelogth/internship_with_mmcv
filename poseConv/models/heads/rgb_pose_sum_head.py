import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class RGBPoseHeadSum(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 dropout=0.5,
                 loss_cls=None,
                 init_std=0.01,
                 **kwargs):

        if loss_cls is None:
            loss_cls = dict(type='CrossEntropyLoss')

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.dropout = dropout
        self.init_std = init_std
        self.dropout_layer = None
        self.pooling_layer = nn.AdaptiveAvgPool3d(1)

        if self.dropout != 0:
            self.dropout_layer = nn.Dropout(p=self.dropout)

        self.fc_layer = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        normal_init(self.fc_layer, std=self.init_std)

    def forward(self, x):

        x_rgb, x_pose = self.pooling_layer(x[0]), self.pooling_layer(x[1])

        assert x_rgb.shape == x_pose.shape, 'The dimension of the rgb features and the pose features should be the same'

        x_rgb = x_rgb.view(x_rgb.size(0), -1)
        x_pose = x_pose.view(x_pose.size(0), -1)

        x = torch.add(x_pose, x_rgb)

        assert x.shape[1] == self.in_channels

        if self.dropout_layer is not None:
            x = self.dropout_layer(x)

        scores = self.fc_layer(x)
        return scores