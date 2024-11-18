"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.atrous_block6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_1x1_output(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcite, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = self.global_avgpool(x)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se

class DeepLabV3PlusCustom(nn.Module):
    def __init__(self, num_classes, in_channels=6):
        super(DeepLabV3PlusCustom, self).__init__()
        # Utilizar un backbone como ResNet
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # ASPP Module
        self.aspp = ASPP(in_channels=2048, out_channels=256)
        
        # Decoder
        self.decoder_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.se = SqueezeExcite(256)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.decoder_conv1(x)
        x = F.relu(x)
        x = self.decoder_conv2(x)
        x = F.relu(x)
        x = self.se(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.classifier(x)
        return x

model = DeepLabV3PlusCustom(num_classes=1, in_channels=6)
"""

import torch
from torchvision import transforms as T
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from metrics import calculate_metrics, DICE_BCE_Loss, DICE_BCE_Loss2
from my_dataset import My_dataloader
from modules_DeepLabV3 import DeepLabV3Plus

from typing import Any
import click
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SMOOTH = 1e-8

def calculate_metrics(pred_mask: Any, true_mask: Any) -> torch.Tensor:
    pred_mask = pred_mask.float()
    true_mask = true_mask.float()

    intersection = torch.sum(pred_mask * true_mask)
    union = torch.sum((pred_mask + true_mask) > 0.5)

    # Add a small epsilon to the denominator to avoid division by zero
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    dice_coefficient = (2 * intersection + SMOOTH) / (
        torch.sum(pred_mask) + torch.sum(true_mask) + SMOOTH
    )
    pixel_accuracy = torch.sum(pred_mask == true_mask) / true_mask.numel() 

    return iou.item(), dice_coefficient.item(), pixel_accuracy.item()




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
        

def dice_coeff(logits, targets):
    logits = torch.where(logits>0.5, 1., 0.)
    intersection = 2*(logits*targets).sum()
    union = (logits + targets).sum()
    if union == 0:
        return 1
    dice_coeff = intersection / union
    return dice_coeff.item()

def miou_binary(pred, truth):
    pred_bi = torch.where(pred > 0.5, 1., 0.)   # [N, C, H, W]
    inter = pred_bi + truth
    area_inter = torch.histc(inter.float(), bins=3, min=0, max=2)
    area_inter = area_inter[0:3:2]
    area_pred = torch.histc(pred, bins=2, min=0, max=1)
    area_truth = torch.histc(truth.float(), bins=2, min=0, max=1)
    area_union = area_pred + area_truth - area_inter
    iou = area_inter / (area_union + 1e-7)
    miou = iou.mean()
    return miou.item()

def oa_binary(pred, truth):
    ''' des: calculate overall accuracy (2-class classification) for each batch
        input: 
            pred(4D tensor), and truth(4D tensor)
    '''
    pred_bi = torch.where(pred>0.5, 1., 0.)   # [N,C,H,W]
    inter = pred_bi+truth
    area_inter = torch.histc(inter.float(), bins=3, min=0, max=2)
    area_inter = area_inter[0:3:2]
    area_pred = torch.histc(pred, bins=2, min=0, max=1)
    oa = area_inter/(area_pred+0.0000001)
    oa = oa.mean()
    return oa.item()

#oa_bin=oa_binary(logits, targets)
#print(oa_bin)
def pixel_acc(logits, targets):
    logits = logits.float()
    targets = targets.float()
    pixel_accuracy = torch.sum(logits == targets) / targets.numel()
    return pixel_accuracy.item()




# Define the evaluate function
def evaluate(model, data_loader, loss_fn):
    model.eval()  # Activate evaluation mode
    test_loss, test_dice, test_iou, test_pixel= 0, 0, 0, 0

    with torch.no_grad():  # Disable gradient calculation
        for images, masks in data_loader:
            images = images.to(device)  # Convert images to float32
            masks = masks.to(device)  # Convert masks to float32
            logits = model(images)
            loss = loss_fn(logits, masks)
            logits = logits > 0.5
            test_loss += loss.item()
            #test_dice += dice_coeff(logits, masks)
            #test_miou += miou_binary(logits, masks)
            #test_pixel  += pixel_acc(logits, masks)

            test_iou_1, test_dice_1, test_pixel_1 = calculate_metrics(logits, masks)

            test_iou += test_iou_1 
            test_dice += test_dice_1
            test_pixel += test_pixel_1

    # Average the values
    test_loss /= len(data_loader)
    test_dice /= len(data_loader)
    test_iou /= len(data_loader)
    test_pixel /= len(data_loader)
    
    return test_loss, test_dice, test_iou, test_pixel

# Load the saved model
#loss = DiceLoss()
loss = DICE_BCE_Loss2()
model = DeepLabV3Plus(num_classes=1).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001, betas=(0.9, 0.999))
model.load_state_dict(torch.load('./output/best_model3.pth'))

# Assuming `test_loader` is defined
###
transform_data=T.Compose([T.Resize([256, 256])])
data_dir_test='/home/cornelius/Documentos/Glaciar_segmentation_datasets/dataset_glacier/images_data_base/testing/'
Testing_dataset=My_dataloader(data_dir_test,transform=transform_data )
test_loader=DataLoader(Testing_dataset, batch_size=1, num_workers=8, shuffle=False)
###


test_loss, test_dice, test_iou, test_pixel = evaluate(model, test_loader, loss)

print(f"Test Loss: {test_loss:.4f} | Test DICE Coeff: {test_dice:.4f} | Test mIoU: {test_iou:.4f} | Test pixel: {test_pixel:.4f}")