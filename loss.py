import torch
import utils
import math
from torch import nn
import torch.nn.functional as F
import config as cfg

class FocalLoss(torch.nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.gamma = cfg.focal_loss_gamma
        self.alpha = cfg.focal_loss_alpha

    def forward(self, logits, targets,avg_factor):
        """
        计算分类分支的loss, 即focal loss.
        :param logits: 神经网络分类分支的输出. type为tensor, shape为(cumsum_5(N*ni),80),其中N是batch size, ni为第i层feature map的样本数量
        :param targets: 表示分类分之的targer labels, type为tensor, shape为(cumsum_5(N*ni),), 其中N是batch size, 正样本的label介于[0,79], 负样本的label为-1
        :return loss: 所有anchor point的loss之和.
        """
        num_classes = logits.shape[1]
        device = targets.device
        dtype = targets.dtype
        class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0)

        t = targets.unsqueeze(1)
        p = torch.sigmoid(logits)
        p = p.clamp(min=1e-6,max=1-1e-6)

        term1 = (1-p)**self.gamma*torch.log(p)
        term2 = p**self.gamma*torch.log(1-p)

        loss = -(t == class_range).float()*term1*self.alpha - ((t != class_range)*(t >= 0)).float()*term2*(1-self.alpha)

        loss = loss.sum() /avg_factor
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self,preds,targets,avg_factor):
        loss=torch.abs(preds-targets)
        loss=loss.sum()/avg_factor
        return loss

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.class_loss = FocalLoss()
        self.reg_loss = L1Loss()

        self.base_anchors=utils.compute_base_anchors()

    def compute_valid_flag(self, pad_img_shape, scales, num_anchors, device):
        valid_flag_per_img=[]
        for i, scale in enumerate(scales):
            stride=float(cfg.fpn_strides[i])
            h_fpn = scale[0]
            w_fpn = scale[1]
            h_valid = math.ceil(pad_img_shape[0]/stride)
            w_valid = math.ceil(pad_img_shape[1]/stride)

            y_valid = torch.zeros((h_fpn,), device=device, dtype=torch.bool)
            x_valid = torch.zeros((w_fpn,), device=device, dtype=torch.bool)
            x_valid[:w_valid] = 1
            y_valid[:h_valid] = 1

            y_valid,x_valid = torch.meshgrid(y_valid,x_valid)
            y_valid=y_valid.reshape(-1)
            x_valid=x_valid.reshape(-1)
            valid_flag_per_level=y_valid&x_valid
            valid_flag_per_level=valid_flag_per_level.view(-1,1).repeat(1,num_anchors).view(-1)
            valid_flag_per_img.append(valid_flag_per_level)
        valid_flag_per_img=torch.cat(valid_flag_per_img,dim=0)
        return valid_flag_per_img

    def compute_targets(self,anchors,valids, gt_bboxes, gt_labels):
        targets=[]

        for i, valid in enumerate(valids):
            target_per_img={}
            gt_bbox_per_img=gt_bboxes[i]
            gt_label_per_img=gt_labels[i]
            valid_anchor = anchors[valid]

            assigned_gt_inds = torch.full((valid_anchor.size(0),), fill_value=-1, device=valid_anchor.device,
                                          dtype=torch.long)
            overlaps = utils.compute_iou_xyxy(gt_bbox_per_img, valid_anchor)
            max_overlap, argmax_overlap = overlaps.max(dim=0)
            max_gt_overlap, argmax_gt_overlap = overlaps.max(dim=1)

            neg_inds = max_overlap < cfg.neg_th
            pos_inds = max_overlap >= cfg.pos_th
            assigned_gt_inds[neg_inds] = 0
            assigned_gt_inds[pos_inds] = argmax_overlap[pos_inds] + 1

            for j in range(gt_bbox_per_img.size(0)):
                index = (overlaps[j, :] == max_gt_overlap[j])
                assigned_gt_inds[index] = j + 1

            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(1)
            neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(1)
            pos_anchors = valid_anchor[pos_inds]
            pos_gt_bboxes = gt_bbox_per_img[assigned_gt_inds[pos_inds] - 1, :]
            pos_gt_labels = gt_label_per_img[assigned_gt_inds[pos_inds] - 1]
            neg_gt_labels = pos_gt_labels.new_full((neg_inds.size(0),), cfg.num_classes)
            mean = pos_anchors.new_tensor(cfg.encode_mean).view(1, 4)
            std = pos_anchors.new_tensor(cfg.encode_std).view(1, 4)

            reg_target_per_img = utils.reg_encode(pos_anchors, pos_gt_bboxes, mean, std)
            target_per_img["valid_inds"] = valid
            target_per_img["pos_inds"] = pos_inds
            target_per_img["neg_inds"] = neg_inds
            target_per_img["cls_labels"] = torch.cat([pos_gt_labels, neg_gt_labels], dim=0)
            target_per_img["reg_targets"] = reg_target_per_img

            targets.append(target_per_img)

        return targets

    def forward(self, cls_preds, reg_preds, gt_bboxes, gt_labels, pad_img_shape):
        scales=[cls_pred.shape[-2:] for cls_pred in cls_preds]
        device=cls_preds[0].device
        dtype=cls_preds[0].dtype
        num_anchors=self.base_anchors[0].size(0)

        anchors=utils.compute_anchors(self.base_anchors,scales,device,dtype)
        anchors=torch.cat(anchors)
        valids=[]
        for i in range(len(pad_img_shape)):
            valids_per_img=self.compute_valid_flag(pad_img_shape[i],scales,num_anchors,device)
            valids.append(valids_per_img)
        targets = self.compute_targets(anchors,valids,gt_bboxes,gt_labels)

        cls_preds_batch=[]
        cls_target_batch=[]
        reg_preds_batch=[]
        reg_target_batch=[]
        num_pos=0
        for i in range(len(gt_bboxes)):
            target_per_img=targets[i]
            cls_preds_per_img=[cls_pred[i].permute(1,2,0).reshape(-1,cfg.num_classes) for cls_pred in cls_preds]
            reg_preds_per_img=[reg_pred[i].permute(1,2,0).reshape(-1,4) for reg_pred in reg_preds]
            cls_preds_per_img = torch.cat(cls_preds_per_img, dim=0)
            reg_preds_per_img = torch.cat(reg_preds_per_img, dim=0)

            valid_inds_per_img=target_per_img["valid_inds"]
            cls_preds_per_img=cls_preds_per_img[valid_inds_per_img]
            reg_preds_per_img=reg_preds_per_img[valid_inds_per_img]

            pos_inds_per_img=target_per_img["pos_inds"]
            neg_inds_per_img=target_per_img["neg_inds"]

            pos_cls_pred=cls_preds_per_img[pos_inds_per_img]
            neg_cls_pred=cls_preds_per_img[neg_inds_per_img]
            pos_reg_pred=reg_preds_per_img[pos_inds_per_img]

            cls_preds_batch.append(torch.cat([pos_cls_pred,neg_cls_pred],dim=0))
            reg_preds_batch.append(pos_reg_pred)

            cls_target_batch.append(target_per_img["cls_labels"])
            reg_target_batch.append(target_per_img["reg_targets"])

            num_pos+=pos_inds_per_img.size(0)
        cls_preds_batch=torch.cat(cls_preds_batch)
        cls_target_batch=torch.cat(cls_target_batch)
        reg_preds_batch=torch.cat(reg_preds_batch)
        reg_target_batch=torch.cat(reg_target_batch)

        cls_loss=self.class_loss(cls_preds_batch,cls_target_batch, num_pos)
        reg_loss=self.reg_loss(reg_preds_batch,reg_target_batch, num_pos)

        return cls_loss,reg_loss
