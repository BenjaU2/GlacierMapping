import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split

from metrics import calculate_metrics, DICE_BCE_Loss, DICE_BCE_Loss2
from my_dataset import My_dataloader
from modules_DeepLabV3 import DeepLabV3Plus
from modules_ResUnet import DeepResUNet
from modules_Unet import Unet

import click
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import rasterio
import numpy as np

def save_pred_img(array, output_path):
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count='float32'
    ) as dst:
        dst.write(array,1)

def display_single_image(image, mask, pred):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    fig.tight_layout()

    # Imagen RGB
    rgb_image = image[..., [2, 1, 0]]

    # Imprimir la imagen original
    ax[0].imshow(rgb_image)
    ax[0].set_title('Imagen Original')

    # Imprimir la máscara
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Máscara')

    # Imprimir la predicción de la máscara
    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title('Predicción')

    plt.show()


CLASSES = 1
INPUT = (256, 256)

@click.command()
@click.option("-T", "--data-dir", type = str, required = True, help = "Path for data train directory")

def main(data_dir):
    click.secho(message="🔎 Evaluation...", fg="blue")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Función de perdida
    #criterion = DiceLoss()
    criterion = DICE_BCE_Loss2()
    # Modelo
    model = DeepLabV3Plus(num_classes=CLASSES).to(device)
    #model = Unet(6,1).to(device)
    #model = DeepResUNet(6,1)


    model.load_state_dict(torch.load("./output/best_model3.pth"))
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
                f"Avg IoU EVal: {avg_iou_eval:.4f}\n"
                f"Avg Pix Acc EVal: {avg_pixel_accuracy_eval:.4f}\n"
                f"Avg Dice Coeff EVal: {avg_dice_coefficient_eval:.4f}\n"
                f"{'-'*50}"
            )
   
    
    click.secho(message="🎉 Evaluation Done!", fg="blue")

    ########################## Implementar como guardar las imagenes predichas #########################
    


    return

if __name__ == "__main__":
    main()
