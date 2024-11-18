import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
from metrics import calculate_metrics, DICE_BCE_Loss, DICE_BCE_Loss2
from my_dataset import My_dataloader
from modules_DeepLabV3 import DeepLabV3Plus
from modules_ResUnet import DeepResUNet
from modules_Unet import Unet
from typing import Any
import click
from tqdm import tqdm
import matplotlib.pyplot as plt

CLASSES = 1
INPUT = (256, 256)

@click.command()
@click.option("-D", "--data-dir", type=str, required=True, help="Path for Data Directory")

def main(data_dir):
    click.secho(message="🔎 Evaluation...", fg="blue")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Función de perdida
    #criterion = DiceLoss()
    criterion = DICE_BCE_Loss2()
    # Modelo
    #model = DeepLabV3Plus(num_classes=CLASSES).to(device)
    #model = Unet(6,1).to(device)
    model = DeepResUNet(6,1).to(device)


    model.load_state_dict(torch.load("./output/best_model_DeepResUnet_L5_L8_C2L1_P1.pth"))
    #model.to(device)
    model.eval()
    # Inicializar variables
    total_loss_eval = 0.0
    total_iou_eval = 0.0
    total_pixel_accuracy_eval = 0.0
    total_dice_coefficient_eval = 0.0
    # Pre-procesamiento de datos
    eval_transform=T.Compose([T.Resize([INPUT[0], INPUT[1]])])
    eval_dataset=My_dataloader(data_dir, transform=eval_transform)
    eval_dataloader=DataLoader(eval_dataset, batch_size=1, num_workers=8, shuffle=False)
    click.echo(message=f"\n{click.style('Evaluation Size: ', fg='blue')}{eval_dataset.__len__()}\n")
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluation", unit="image")
    # Calculo de gradiente
    with torch.no_grad():
        for images, masks in eval_dataloader:
            images, masks = images.to(device), masks.to(device)
            output = model(images)

            eval_loss = criterion(output, masks)
            #print()
            total_loss_eval += eval_loss.item()

            pred_masks = output > 0.5
            #pred_masks = output
            iou_eval, dice_coefficient_eval, pixel_accuracy_eval = calculate_metrics(
                pred_masks, masks
            )

            total_iou_eval += iou_eval
            total_dice_coefficient_eval += dice_coefficient_eval
            total_pixel_accuracy_eval += pixel_accuracy_eval
        
        # Displaying metrics in the progress bar description
        eval_dataloader.set_postfix(
            total_loss_eval=eval_loss.item(),
            eval_iou=iou_eval,
            eval_pix_acc=pixel_accuracy_eval,
            eval_dice_coef=dice_coefficient_eval,
        )
    #total_loss_eval /= len(eval_dataloader)
    avg_total_loss = total_loss_eval / len(eval_dataloader)
    avg_iou_eval = total_iou_eval / len(eval_dataloader)
    avg_pixel_accuracy_eval = total_pixel_accuracy_eval / len(eval_dataloader)
    avg_dice_coefficient_eval = total_dice_coefficient_eval / len(eval_dataloader)

    print(
                f"Avg Eval Loss: {avg_total_loss:.4f}\n"
                f"Avg mIoU EVal: {avg_iou_eval:.4f}\n"
                f"Avg Pix Acc EVal: {avg_pixel_accuracy_eval:.4f}\n"
                f"Avg Dice Coeff EVal: {avg_dice_coefficient_eval:.4f}\n"
                f"{'-'*50}"
            )
    
    click.secho(message="🎉 Evaluation Done!", fg="blue")
    return

if __name__ == "__main__":
    main()
    



    