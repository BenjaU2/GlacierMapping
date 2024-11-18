import sys
import os
from glob import glob
import xml.etree.ElementTree as ET
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import math
from rasterio.plot import show
import matplotlib.patches as patches


# Verificar si se proporcionó un argumento de línea de comandos
if len(sys.argv) != 2:
    print("Uso: python script.py <directorio_escena> ejem: LC08..._T1")
    sys.exit(1)

# Obtener el argumento de línea de comandos que representa la dirección
direccion = sys.argv[1]


def open_bands_and_deskew(list_bands, directory):
    # Cargando nombres de archivos de bandas de interés
    bands_files = []
    for band in list_bands:
        try:
            #path = glob(f'../{directory}/*_B{band}.TIF')[0]
            path = glob(f'{directory}/*_B{band}.TIF')[0]
            bands_files.append(path)
            print(f"Band {band} exists.")
        except IndexError:
            raise FileNotFoundError(f"No file found for band {band}.")


    # Intentando cargar archivo de metadatos
    metadata_filename = f"{directory}/*.xml"
    try:
        metadata_path = glob(metadata_filename)[0]
        print(f"Metadafile exists.")
    except IndexError:
        raise FileNotFoundError(f"No metadata file found for this scene.")


    # Cargando los archivos TIF
    bands_digital_number = []

    for band_name in bands_files:
        band_array = rasterio.open(band_name)
        bands_digital_number.append(band_array.read(1))
        band_array.close()

    print(f"{len(bands_digital_number)} bands have been read.")

    # Apilando las bandas
    stack_image = np.stack(bands_digital_number, axis=-1)
    print(f"Stacked image has shape: {stack_image.shape}")

    # Alineando imagen
    print("Aligning image")


    # Giving values Reflectance
    with open(metadata_path) as f:
        # Loading metada.xml file
        tree = ET.parse(f)
        root = tree.getroot()

        MULT = float (root.find(f'.//REFLECTANCE_MULT_BAND_1').text)
        ADD = float (root.find(f'.//REFLECTANCE_ADD_BAND_1').text)
        SUN = float (root.find(f'.//SUN_ELEVATION').text)

        UL_X = float(root.find(f'.//CORNER_UL_PROJECTION_X_PRODUCT').text)
        UL_Y = float(root.find(f'.//CORNER_UL_PROJECTION_Y_PRODUCT').text)
        LR_X = float (root.find(f'.//CORNER_LR_PROJECTION_X_PRODUCT').text)
        LR_Y = float (root.find(f'.//CORNER_LR_PROJECTION_Y_PRODUCT').text)
        if UL_Y < 0.0:
                UL_Y = 10000000 + UL_Y

        if LR_Y < 0.0:
                LR_Y = 10000000 + LR_Y
    ###########################################################
    #SITE_X, SITE_Y = 303375.34, 8460479.97 #(Cerca)
    stack_image=((stack_image*MULT) + ADD)/((math.sin(math.radians(SUN))))
    SITE_X, SITE_Y = 305293.51, 8458795.55

    #SITE_X, SITE_Y = 303907.23, 8460976.19
    X = stack_image.shape[1] * (SITE_X - UL_X) / (LR_X - UL_X)
    Y = stack_image.shape[0] * (SITE_Y - UL_Y) / (LR_Y - UL_Y)

    X = int(X)
    Y = int(Y)

    # Asegurarse de que las coordenadas de recorte no se salgan de los límites de la imagen
    Y_start = max(0, Y - 256)
    Y_end = min(stack_image.shape[0], Y + 256)
    X_start = max(0, X - 256)
    X_end = min(stack_image.shape[1], X + 256)

    img_final = stack_image[Y_start:Y_end, X_start:X_end, :]
    ###########################################################

    return img_final

image = open_bands_and_deskew([2,3,4,5,6,7], direccion)
print("Image shape: ", image.shape)
directorio_parches = f'{direccion}_parches_512x512/'
if not os.path.exists(directorio_parches):
    os.makedirs(directorio_parches)

alto = image.shape[0]
ancho = image.shape[1]

parche_size = 512

profile = {
    'count' : image.shape[2],
    'height' : parche_size,
    'width' : parche_size,
    'dtype' : 'float64'
}

parche = image.transpose(2, 0, 1)

nombre_parche = f"{direccion}.TIF"
ruta_parche = os.path.join(directorio_parches, nombre_parche)

with rasterio.open(ruta_parche, 'w', **profile) as dst:
            dst.write(parche)



plt.figure(figsize=(10, 10))
plt.imshow(image[:, :, 0])
plt.title('Zonas de recorte')
plt.axis('off')
plt.show()




"""
Hola chat, tengo una carpeta donde tengo una imagen multiespectral con el nombre: Imagen_original_6_bandas.TIF, después en la misma carpeta tengo 3 imagenes completamente binarias con los nombres Predicción1.TIF y Prediccion2.TIF, teniendo como pixel de interes el blanco, las imagenes binarias son la predicción temporal de una zona, entoces me gustaria dibujar la imagen RGB y los bordes de la imagenes binarias, para poder vizualizar la variación de mi area de interes, los bordes de mi area de interes tendran que variar porque los boordes de las imagenes binarias variaran .
"""
