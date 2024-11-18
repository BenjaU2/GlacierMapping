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
import rasterio
import os
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
    ax[0].axis('off') 
    
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[1].axis('off') 

    ax[2].imshow(pred, cmap='gray')
    ax[2].set_title('Unet')
    ax[2].axis('off')   

    # Añadir las métricas en la parte inferior de la figura
    #fig.suptitle(f'IoU: {iou:.4f}, DICE: {dice_coefficient:.4f}, Pixel Accuracy: {pixel_accuracy:.4f}', fontsize=16, y=0.05)
    # Añadir las métricas en la parte inferior de la imagen de predicción
    """
    ax[2].text(0.5, -0.1, f'IoU: {iou:.4f} DICE: {dice_coefficient:.4f} Pixel Accuracy: {pixel_accuracy:.4f}', transform=ax[2].transAxes, fontsize=12)
    """
    plt.show()

def save_pred_img(array, output_path):
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype='float32'  # Asegurarse de que se guarda como float32
    ) as dst:
        dst.write(array, 1)

data_dir = "/home/cornelius/Documentos/Glaciar_dataset_C2_L1/Analisis_temporal/USGS/Recortar_patchs_L5/2010/"
output_folder = "/home/cornelius/Documentos/Glaciar_dataset_C2_L1/Analisis_temporal/USGS/Recortar_patchs_L5/2010/pred_img/"

INPUT = (256, 256)
CLASSES = 1
counter = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = DeepLabV3Plus(num_classes=CLASSES)
model = Unet(6,1)
model.load_state_dict(torch.load("./output/best_model_Unet_L5_L8_C2L1_P1.pth"))
model.to(device)

eval_transform=T.Compose([T.Resize([INPUT[0], INPUT[1]])])
eval_dataset=My_dataloader(data_dir, transform=eval_transform)
eval_dataloader=DataLoader(eval_dataset, batch_size=1, num_workers=8, shuffle=False)

# Evaluar el modelo y mostrar las primeras 10 imágenes con sus métricas
model.eval()  # Establecer el modelo en modo de evaluación

# Iterar sobre las imagenes de prueba
for images, masks in eval_dataloader:
    with torch.no_grad():
        pred = model(images.to(device)).cpu().detach()
        pred = (pred > 0.5) # Convertir a binario

    # Suponiendo que solo tienes un batch con una sola imagen
    image = images[0]
    mask = masks[0]
    pred = pred[0]

    # Guardar la imagen predicha
    output_path = os.path.join(output_folder, f"pred_image_{counter}.TIF")
    save_pred_img(pred.squeeze().numpy(), output_path)
    print(f"La imagen predicha {counter} ha sido guardada en: {output_path}")
    # Mostrar la imagen
    display_single_image(image.permute(1, 2, 0).numpy(), mask.squeeze().numpy(), pred.squeeze().numpy())

    # Incrementar el contador
    counter += 1

"""
with torch.no_grad():
    for i, (image, mask) in enumerate(eval_dataloader):
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
"""