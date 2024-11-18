import numpy as np
import rasterio
import sys
import os
from glob import glob
"""# Verificar si se proporcionó un argumento de línea de comandos
if len(sys.argv) != 2:
    print("Uso: python script.py <directorio_escena> ejem: LC08..._T1")
    sys.exit(1)

# Obtener el argumento de línea de comandos que representa la dirección
direccion = sys.argv[1]

def open_and_area(directory):
    img_list = []
    for img in img_list:
        try:
            path =glob(f'{directory}/*.TIF')[0]
            img_list.append(path)
        except IndexError:
            raise FileNotFoundError(f"No file found.")
    print(img_list)  
    
    
    for image in img_list:
        image_array = rasterio.open(image)
        image
    
open_and_area(direccion)
"""



# Ruta de la carpeta con las imágenes
carpeta_imagenes = '/home/cornelius/Documentos/Glaciar_dataset_C2_L1/Analisis_temporal/USGS/Recortar_patchs_L5/pred_img/'
lista_areas = []

# Función para calcular el área de una imagen binaria
def calcular_area_imagen(imagen_path, resolucion_pixel):
    with rasterio.open(imagen_path) as src:
        imagen = src.read(1)  # Lee la primera banda de la imagen
        # Cuenta los píxeles blancos (1)
        pixeles_blancos = (imagen == 1).sum()
        # Calcula el área: número de píxeles blancos * área de cada píxel
        area = pixeles_blancos * (resolucion_pixel ** 2)
        return area

# Listar todas las imágenes en la carpeta
imagenes = [f for f in os.listdir(carpeta_imagenes) if f.endswith('.TIF')]

# Procesar solo las primeras 4 imágenes
for imagen in imagenes[:4]:
    imagen_path = os.path.join(carpeta_imagenes, imagen)
    area = calcular_area_imagen(imagen_path, resolucion_pixel=30)
    lista_areas.append(area)

# Sumar las áreas de las 4 imágenes
area_total = sum(lista_areas)

print("Áreas de cada imagen:", lista_areas)
print("Área total de las primeras 4 imágenes:", area_total)