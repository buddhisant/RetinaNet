import torch
import config as cfg
import utils

import torch.nn.functional as F

class Inference():
    def __init__(self):
        self.base_anchors = utils.compute_base_anchors()

    def __call__(self, cls_preds,reg_preds,res_img_shape,pad_img_shape):
        scales = [cls_pred.shape[-2:] for cls_pred in cls_preds]
        device = cls_preds[0].device
        dtype = cls_preds[0].dtype

        anchors = utils.compute_anchors(self.base_anchors, scales, device, dtype)

        res_img_shape=res_img_shape.squeeze(0)

        cls_preds=[cls_pred.squeeze(0).permute(1,2,0).reshape(-1,cfg.num_classes) for cls_pred in cls_preds]
        reg_preds=[reg_pred.squeeze(0).permute(1,2,0).reshape(-1,4) for reg_pred in reg_preds]

        candidate_anchors=[]
        candidate_scores=[]
        candidate_factors=[]
        for i in range(len(cls_preds)):
            anchors_per_level=anchors[i]
            cls_preds_per_level=cls_preds[i]
            reg_preds_per_level=reg_preds[i]

            cls_preds_per_level=cls_preds_per_level.sigmoid()
            max_scores, max_inds=cls_preds_per_level.max(dim=1)
            topk=min(cfg.topk_condidate,cls_preds_per_level.size(0))
            _, topk_inds=torch.topk(max_scores,k=topk)
            candidate_anchors.append(anchors_per_level[topk_inds,:])
            candidate_scores.append(cls_preds_per_level[topk_inds,:])
            candidate_factors.append(reg_preds_per_level[topk_inds,:])

        candidate_anchors=torch.cat(candidate_anchors,dim=0)
        candidate_factors=torch.cat(candidate_factors,dim=0)
        candidate_scores=torch.cat(candidate_scores,dim=0)

        mean = candidate_anchors.new_tensor(cfg.encode_mean).view(1, 4)
        std = candidate_anchors.new_tensor(cfg.encode_std).view(1, 4)
        candidate_bboxes=utils.reg_decode(candidate_anchors,candidate_factors,mean,std,res_img_shape)
        pos_mask=candidate_scores>=cfg.pos_th_test
        pos_location = torch.nonzero(pos_mask, as_tuple=False)
        pos_inds = pos_location[:, 0]
        pos_labels = pos_location[:, 1]

        scores=candidate_scores[pos_inds,pos_labels]
        bboxes=candidate_bboxes[pos_inds,:]
        scores, bboxes, labels = utils.mc_nms(scores, bboxes, pos_labels, cfg.nms_th)
        scores = scores[:cfg.nms_post]
        bboxes = bboxes[:cfg.nms_post]
        labels = labels[:cfg.nms_post]

        return scores,bboxes,labels
