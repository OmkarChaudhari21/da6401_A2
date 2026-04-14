import torch
import torch.nn as nn

class IoULoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        x1_p = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        y1_p = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        x2_p = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        y2_p = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        x1_t = target_boxes[:, 0] - target_boxes[:, 2] / 2
        y1_t = target_boxes[:, 1] - target_boxes[:, 3] / 2
        x2_t = target_boxes[:, 0] + target_boxes[:, 2] / 2
        y2_t = target_boxes[:, 1] + target_boxes[:, 3] / 2

        x1_i = torch.max(x1_p, x1_t)
        y1_i = torch.max(y1_p, y1_t)
        x2_i = torch.min(x2_p, x2_t)
        y2_i = torch.min(y2_p, y2_t)

        inter = torch.clamp(x2_i - x1_i, min=0) * torch.clamp(y2_i - y1_i, min=0)

        area_p = pred_boxes[:, 2] * pred_boxes[:, 3]
        area_t = target_boxes[:, 2] * target_boxes[:, 3]

        union = area_p + area_t - inter + self.eps
        iou = inter / union

        loss = 1 - iou

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss