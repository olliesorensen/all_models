import torch
import torch.nn as nn
import torch.nn.functional as F

from guide_ddrnet import DualResNet_Backbone
from guide_modules import Guided_Upsampling_Block


class GuideDepth(nn.Module):
    def __init__(self,
            tasks, 
            pretrained=True,
            up_features=[64, 32, 16], 
            inner_features=[64, 32, 16]):
        super(GuideDepth, self).__init__()

        self.tasks = tasks

        self.feature_extractor = DualResNet_Backbone(
                pretrained=pretrained, 
                features=up_features[0])

        self.up_1 = Guided_Upsampling_Block(in_features=up_features[0],
                                   expand_features=inner_features[0],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_2 = Guided_Upsampling_Block(in_features=up_features[1],
                                   expand_features=inner_features[1],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")
        self.up_3 = Guided_Upsampling_Block(in_features=up_features[2],
                                   expand_features=inner_features[2],
                                   out_features=up_features[1],
                                   kernel_size=3,
                                   channel_attention=True,
                                   guide_features=3,
                                   guidance_type="full")

        # define task specific layers
        if all (k in tasks for k in ('seg', 'depth', 'normal')):           
            self.pred_task1 = self.conv_layer([filter[0], 13], pred=True)
            self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
            self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)            
        else:
            self.pred_task1 = self.conv_layer([filter[0], 19], pred=True)
            self.pred_task2 = self.conv_layer([filter[0], 10], pred=True)
            self.pred_task3 = self.conv_layer([filter[0], 1], pred=True) 
            
        self.decoders = nn.ModuleList([self.pred_task1, self.pred_task2, self.pred_task3])

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block


    def forward(self, x):
        _, _, im_h, im_w = x.shape

        y = self.feature_extractor(x)

        x_half = F.interpolate(x, scale_factor=.5)
        x_quarter = F.interpolate(x, scale_factor=.25)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')#, align_corners=True)
        y = self.up_1(x_quarter, y)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')#,align_corners=True)
        y = self.up_2(x_half, y)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')#, align_corners=True)
        y = self.up_3(x, y)

        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](y), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)

        return out