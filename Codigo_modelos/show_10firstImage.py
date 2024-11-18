import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split

from metrics import calculate_metrics, calculate_metrics2, DICE_BCE_Loss, DICE_BCE_Loss2
from my_dataset import My_dataloader


from modules_DeepLabV3 import DeepLabV3Plus
from modules_ResUnet import DeepResUNet
from modules_Unet import Unet

from typing import Any
import click
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import matplotlib.pyplot as plt
"""
# Función para calcular métricas
def calculate_metrics(pred_mask: Any, true_mask: Any) -> torch.Tensor:
    SMOOTH = 1e-6  # Puedes ajustar este valor si es necesario
    pred_mask = pred_mask.float()
    true_mask = true_mask.float()

    intersection = torch.sum(pred_mask * true_mask)
    union = torch.sum((pred_mask + true_mask) > 0.5)

    iou = (intersection + SMOOTH) / (union + SMOOTH)
    dice_coefficient = (2 * intersection + SMOOTH) / (
        torch.sum(pred_mask) + torch.sum(true_mask) + SMOOTH
    )
    pixel_accuracy = torch.sum(pred_mask == true_mask) / true_mask.numel()

    return iou.item(), dice_coefficient.item(), pixel_accuracy.item()
"""

# Función para mostrar la imagen, máscara y predicción
def display_single_image(image, mask, pred):
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    #fig.tight_layout()
    # Ajustar el espaciado entre los subplots
    fig.subplots_adjust(wspace=0.001)  # Disminuye este valor para acercar las imágenes

    # Si la imagen tiene más de 3 canales, mostrar los primeros 3 como RGB
    #rgb_image = image[..., :3]
    rgb_image = image[..., [2, 1, 0]]

    ax[0].imshow(rgb_image)
    ax[0].set_title('Imagen Original')
    ax[0].set_ylabel(r'$\Delta_{i+1}$', fontsize=15)
    #ax[0].axis('off') 
    ax[0].set_ylabel(r'$\Delta_{i+1}$', fontsize=15)


    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[1].axis('off') 

    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title('DeepLabV3Plus')
    ax[2].axis('off')   

    # Añadir las métricas en la parte inferior de la figura
    #fig.suptitle(f'IoU: {iou:.4f}, DICE: {dice_coefficient:.4f}, Pixel Accuracy: {pixel_accuracy:.4f}', fontsize=16, y=0.05)
    # Añadir las métricas en la parte inferior de la imagen de predicción
    """
    ax[2].text(0.5, -0.1, f'IoU: {iou:.4f} DICE: {dice_coefficient:.4f} Pixel Accuracy: {pixel_accuracy:.4f}', transform=ax[2].transAxes, fontsize=12)
    """
    
    ax[2].text(0.5, -0.01, f'IoU: {iou:.4f}, DICE: {dice_coefficient:.4f}, PA: {pixel_accuracy:.4f}',
               ha='center', va='top', transform=ax[2].transAxes, fontsize=12)
   
    

    plt.show()

#data_dir = "/home/cornelius/Documentos/Glaciar_dataset_C2_L1/Analisis_temporal/GEE/RECORTE_GEE_LT05_L1TP_003070_19910531/"
data_dir="/home/cornelius/Documentos/Glaciar_dataset_C2_L1/glaciar_data_L5_L8/glaciar_data_L5L8/testing/"
#data_dir = "/home/cornelius/Documentos/Glaciar_dataset_C2_L1/Analisis_temporal/USGS/COMPARAR_USGS/LT05_L1TP_003070_19910726_20200915_02_T1_parches1/"

INPUT = (256, 256)
CLASSES = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = DeepLabV3Plus(num_classes=CLASSES)
#model = DeepResUNet(6,1)
model = Unet(6,1)
model.load_state_dict(torch.load("./output/best_model_Unet_L5_L8_C2L1_P1.pth"))
model.to(device)

eval_transform=T.Compose([T.Resize([INPUT[0], INPUT[1]])])
eval_dataset=My_dataloader(data_dir, transform=eval_transform)
eval_dataloader=DataLoader(eval_dataset, batch_size=1, num_workers=8, shuffle=False)

# Evaluar el modelo y mostrar las primeras 10 imágenes con sus métricas
model.eval()  # Establecer el modelo en modo de evaluación
with torch.no_grad():
    for i, (image, mask) in enumerate(eval_dataloader):
        """
       
        if i >= 20:  # Solo procesar las primeras 10 imágenes
            break
        """

        if i < 100:  # Saltar las primeras 10 imágenes
            continue
        if i >= 120:  # Dejar de procesar después de la imagen 20
            break
        #"""

        # Mover los datos al dispositivo (CPU o GPU)
        image = image.to(device)
        mask = mask.to(device)

        # Obtener la predicción
        pred = model(image)
        #pred = torch.sigmoid(pred)  # Aplicar sigmoide para obtener la predicción binaria
        pred = (pred > 0.5) # Convertir a binario

        # Calcular las métricas
        iou, dice_coefficient, pixel_accuracy = calculate_metrics(pred, mask)

        # Mostrar la imagen, máscara y predicción
        display_single_image(image.cpu().squeeze().permute(1, 2, 0), mask.cpu().squeeze(), pred.cpu().squeeze())

        # Imprimir las métricas
        #print(f'Imagen {i+1}: Nombre: = {image_name:.4f}')
