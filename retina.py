import torch
import math
import utils
import config as cfg

from resnet import resNet
from fpn import PyramidFeatures
from loss import Loss
from inference import Inference

import torch.nn.functional as F

class classier(torch.nn.Module):
    def __init__(self):
        super(classier, self).__init__()
        nums_anchor = len(cfg.anchor_ratio)*len(cfg.anchor_scale)

        self.conv1 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)

        self.output = torch.nn.Conv2d(cfg.fpn_channels, cfg.num_classes*nums_anchor, kernel_size=3, padding=1)

        for m in self.modules():
            if(isinstance(m, torch.nn.Conv2d)):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(m.bias, 0)

        bias_value = -math.log((1 - cfg.class_prior_prob) / cfg.class_prior_prob)
        torch.nn.init.constant_(self.output.bias, bias_value)

    def forward(self, x):
        cls_preds = []
        for x_per_level in x:
            x_per_level = self.conv1(x_per_level)
            x_per_level.relu_()

            x_per_level = self.conv2(x_per_level)
            x_per_level.relu_()

            x_per_level = self.conv3(x_per_level)
            x_per_level.relu_()

            x_per_level = self.conv4(x_per_level)
            x_per_level.relu_()

            logits = self.output(x_per_level)
            cls_preds.append(logits)

        return cls_preds


class regressier(torch.nn.Module):
    def __init__(self):
        super(regressier, self).__init__()
        nums_anchor = len(cfg.anchor_ratio) * len(cfg.anchor_scale)

        self.conv1 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1)

        self.output = torch.nn.Conv2d(cfg.fpn_channels, nums_anchor*4, kernel_size=3, padding=1)

        for m in self.modules():
            if (isinstance(m, torch.nn.Conv2d)):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        reg_preds = []
        for x_per_level in x:
            x_per_level = self.conv1(x_per_level)
            x_per_level.relu_()

            x_per_level = self.conv2(x_per_level)
            x_per_level.relu_()

            x_per_level = self.conv3(x_per_level)
            x_per_level.relu_()

            x_per_level = self.conv4(x_per_level)
            x_per_level.relu_()

            logits = self.output(x_per_level)
            reg_preds.append(logits)

        return reg_preds

class retinanet(torch.nn.Module):
    def __init__(self, is_train=True):
        super(retinanet, self).__init__()
        self.is_train=is_train

        self.resnet = resNet()
        self.fpn = PyramidFeatures()
        self.classier=classier()
        self.regressier=regressier()

        self.loss=Loss()
        self.inference=Inference()

    def forward(self, images, ori_img_shape=None,res_img_shape=None,pad_img_shape=None,gt_bboxes=None,gt_labels=None):
        c3,c4,c5=self.resnet(images)
        features = self.fpn([c3,c4,c5])

        cls_preds=self.classier(features)
        reg_preds=self.regressier(features)

        if(self.is_train):
            cls_loss,reg_loss=self.loss(cls_preds,reg_preds,gt_bboxes,gt_labels,pad_img_shape)
            return {"cls_loss":cls_loss,"reg_loss":reg_loss}
        else:
            scores, bboxes, labels = self.inference(cls_preds,reg_preds,res_img_shape,pad_img_shape)
            scale_factor = ori_img_shape.float() / res_img_shape.float()
            scale_factor = torch.cat([scale_factor, scale_factor], dim=1)
            bboxes = bboxes * scale_factor
            return scores, bboxes, labels
