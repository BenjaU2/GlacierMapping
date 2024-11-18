"""
import rasterio
import numpy as np
import matplotlib.pyplot as plt

input_dir = "/home/cornelius/Documentos/Glaciar_dataset_C2_L1/images/LC08_L1TP_001071_20190623_20200827_02_T1_023_018.TIF"

with rasterio.open(input_dir) as src:
    green_band = src.read(2).astype(float)
    swir_band = src.read(5).astype(float)
    nir_band = src.read(4).astype(float)

# Evitar división por cero
np.seterr(divide='ignore', invalid='ignore')

# Calcular NDSI y NDWI
ndsi = (green_band - swir_band) / (green_band + swir_band)
ndwi = (green_band - nir_band) / (green_band + nir_band)

# Calcular la combinación NDSI1
ndsi1 = ndsi - ndwi

# Definir umbral para nieve/glaciares
ndsi1_threshold = 0.678

# Crear máscara binaria
mask = np.where(ndsi1 > ndsi1_threshold, 255, 0).astype(np.uint8)

# Visualizar la máscara
plt.figure(figsize=(10, 6))
plt.imshow(mask, cmap='coolwarm')
plt.title('Máscara de Nieve/Glaciares')
plt.colorbar(label='Valores de la máscara')
plt.show()
"""

import os
import rasterio
import numpy as np

input_dir="/home/cornelius/Documentos/Glaciar_dataset_C2_L1/Analisis_temporal/USGS/Recortar_patchs_L5/2010/images/"
output_dir = '/home/cornelius/Documentos/Glaciar_dataset_C2_L1/Analisis_temporal/USGS/Recortar_patchs_L5/2010/masks/'
# Convert to mask
def mask_image(file_path, output_dir):
    with rasterio.open(file_path) as src:
        green_band = src.read(2).astype(float)
        swir_band = src.read(5).astype(float)
        nir_band = src.read(4).astype(float)
        
    # Evitar división por cero
    np.seterr(divide='ignore', invalid='ignore')

    # Calcular NDSI
    ndsi = (green_band - swir_band) / (green_band + swir_band)
    ndwi = (green_band - nir_band) / (green_band + nir_band)

    ndsi1=ndsi-ndwi

    # Definir umbral para nieve/glaciares
    ndsi1_threshold = 0.67

    # Crear máscara binaria
    mask = np.where(ndsi1 > ndsi1_threshold, 255, 0).astype(np.uint8)

    # Obtener el nombre base del archivo original y crear un nuevo nombre para la máscara
    base_name = os.path.basename(file_path)
    mask_name = f"{os.path.splitext(base_name)[0]}.TIF"
    output_dir = os.path.join(output_dir, mask_name)
    save_image(mask, output_dir)

def save_image(mask, file_path):
    # Guardar la máscara como una nueva imagen
    with rasterio.open(
        file_path,
        #mask_output_path,
        'w',
        driver='GTiff',
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=np.uint8,
        #crs=src.crs,
        #transform=src.transform,
    ) as dst:
        dst.write(mask, 1)

# Iterar sobre los archivos en el directorio de entrada
for filename in os.listdir(input_dir):
    if filename.endswith('.TIF'):
        file_path = os.path.join(input_dir, filename)
        mask_image(file_path, output_dir)

print(f"Máscara guardada")
