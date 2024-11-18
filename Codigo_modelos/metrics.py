import torch
import torch.nn as nn
from typing import Any


class DICE_BCE_Loss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        intersection = 2*(logits * targets).sum() + self.smooth
        union = (logits + targets).sum() + self.smooth
        dice_loss = 1. - intersection / union

        # Imprimir valor de dice_loss
        print(f'DICE Loss: {dice_loss.item()}')

        loss = nn.BCELoss()
        bce_loss = loss(logits, targets)
        # Imprimir valor de bce_loss
        print(f'BCE Loss: {bce_loss.item()}')

        return dice_loss + bce_loss
    

class DICE_BCE_Loss2(nn.Module):
    def __init__(self, smooth=1, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        intersection = 2*(logits * targets).sum() + self.smooth
        union = (logits + targets).sum() + self.smooth
        dice_loss = 1. - intersection / union
        #print(f'DICE Loss: {dice_loss.item()}')

        bce_loss = nn.BCELoss()(logits, targets)
        #print(f'BCE Loss: {bce_loss.item()}')

        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

SMOOTH = 1e-8
    
def calculate_metrics(pred_mask: Any, true_mask: Any) -> torch.Tensor:
    #pred_mask = torch.where(pred_mask > 0.5, 1., 0.).float()
    pred_mask = pred_mask.float()
    true_mask = true_mask.float()

    intersection = torch.sum(pred_mask * true_mask)
    union = torch.sum((pred_mask + true_mask) > 0.5)
    #union = torch.sum(pred_mask + true_mask)

    # Add a small epsilon to the denominator to avoid division by zero
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    miou= iou.mean()
    dice_coefficient = (2 * intersection + SMOOTH) / (
        torch.sum(pred_mask) + torch.sum(true_mask) + SMOOTH
    )
    pixel_accuracy = torch.sum(pred_mask == true_mask) / true_mask.numel() 

    return miou.item(), dice_coefficient.item(), pixel_accuracy.item()

SMOOTH = 1e-8

def calculate_metrics2(logits: torch.Tensor, true_mask: torch.Tensor, num_classes: int) -> tuple:
    # Umbralizar logits para convertirlos en una máscara binaria
    pred_mask = torch.where(logits > 0.5, 1., 0.).float()
    true_mask = true_mask.float()

    iou_per_class = []
    accuracy_per_class = []

    for class_idx in range(num_classes):
        pred_class = (pred_mask == class_idx).float()
        true_class = (true_mask == class_idx).float()

        intersection = torch.sum(pred_mask + true_mask) 
        union = torch.sum(pred_class + true_class)
        class_iou = (intersection + SMOOTH) / (union - intersection + SMOOTH)
        iou_per_class.append(class_iou)

        class_accuracy = torch.sum(pred_class == true_class) / true_class.numel()
        accuracy_per_class.append(class_accuracy)

    mIoU = torch.mean(torch.tensor(iou_per_class))
    mPA = torch.mean(torch.tensor(accuracy_per_class))

    # Cálculo del Dice Coefficient
    intersection = 2 * torch.sum(pred_mask * true_mask)
    union = torch.sum(pred_mask) + torch.sum(true_mask)
    dice_coefficient = (intersection + SMOOTH) / (union + SMOOTH)

    # Cálculo de la precisión global de píxeles
    #overall_pixel_accuracy = torch.sum(pred_mask == true_mask) / true_mask.numel()

    #return mIoU.item(), mPA.item(), dice_coefficient.item(), overall_pixel_accuracy.item()
    return mIoU.item(), mPA.item(), dice_coefficient.item()