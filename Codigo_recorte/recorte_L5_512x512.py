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
            #LC08_L1TP_003070_20140607_20200911_02_T1_B1.TIF
            #LT05_L1GS_003070_20110530_20200822_02_T2_B1.TIF
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
        print(metadata_path)
    except IndexError:
        raise FileNotFoundError(f"No metadata file found for this scene.")


    # Cargando los archivos TIF
    bands_digital_number = []

    for band_name in bands_files:
        band_array = rasterio.open(band_name)
        bands_digital_number.append(band_array.read(1))
        band_array.close()
    print("########################")
    print(type(bands_digital_number))
    print(f"{len(bands_digital_number)} bands have been read.")

    # Giving values Reflectance
    with open(metadata_path) as f:
        # Loading metada.xml file
        tree = ET.parse(f)
        root = tree.getroot()
        # Inicializar listas para almacenar los valores de las bandas
        REFLECTANCE_MULT_BAND = []
        REFLECTANCE_ADD_BAND = []

        # Recorrer las bandas del 1 al 7
        for band in [1,2,3,4,5,7]:
            # Obtener valores de REFLACTANCIE_MULT_BAND
            mult_value = float(root.find(f'.//REFLECTANCE_MULT_BAND_{band}').text)
            REFLECTANCE_MULT_BAND.append(mult_value)

            # Obtener valores de REFLECTANCE_ADD_BAND
            add_value = float(root.find(f'.//REFLECTANCE_ADD_BAND_{band}').text)
            REFLECTANCE_ADD_BAND.append(add_value)

        print(REFLECTANCE_MULT_BAND)
        print(REFLECTANCE_ADD_BAND)
        SUN = float (root.find(f'.//SUN_ELEVATION').text)
        ################## #######################################
        UL_X = float(root.find(f'.//CORNER_UL_PROJECTION_X_PRODUCT').text)
        UL_Y = float(root.find(f'.//CORNER_UL_PROJECTION_Y_PRODUCT').text)
        LR_X = float (root.find(f'.//CORNER_LR_PROJECTION_X_PRODUCT').text)
        LR_Y = float (root.find(f'.//CORNER_LR_PROJECTION_Y_PRODUCT').text)
        if UL_Y < 0.0:
                UL_Y = 10000000 + UL_Y

        if LR_Y < 0.0:
                LR_Y = 10000000 + LR_Y

    # Calcular la reflactancia para cada banda
    bands_digital_number_new = []
    for m in range(len(bands_digital_number)):
        # Aplicar la fórmula a cada banda
        reflectance_band = ((bands_digital_number[m] * REFLECTANCE_MULT_BAND[m]) + REFLECTANCE_ADD_BAND[m]) / (math.sin(math.radians(SUN)))
        bands_digital_number_new.append(reflectance_band)
    print(bands_digital_number_new[0].shape)


    # Apilando las bandas
    stack_image = np.stack(bands_digital_number_new, axis=-1)
    print(f"Stacked image has shape: {stack_image.shape}")

    # Alineando imagen
    print("Aligning image")
    # Shape de la imagen de salida
    print(f'Final shape of image is {stack_image.shape}.')

    ###########################################################
    #SITE_X, SITE_Y = 303375.34, 8460479.97 #(Cerca)
    SITE_X, SITE_Y = 305293.51, 8458795.55
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

image = open_bands_and_deskew([1,2,3,4,5,7], direccion)
print(type(image))
print("Image shape. ",image.shape)
#directorio_parches = f'{direccion}_parches/'
directorio_parches = f'{direccion}_parches_L5_512x512/'

# Crear el directorio si no existe
if not os.path.exists(directorio_parches):
    os.makedirs(directorio_parches)

#height, width, _ = original_image_shape
height = image.shape[0]
width = image.shape[1]

parche_size = 512

profile = {
'count' : image.shape[2],
'height' : parche_size,
'width' : parche_size,
'dtype' : 'float64'
}

parche = image.transpose(2, 0, 1)
print("Parche shape. ", parche.shape)
nombre_parche = f"{direccion}.TIF"
ruta_parche = os.path.join(directorio_parches, nombre_parche)


with rasterio.open(ruta_parche, 'w', **profile) as dst:
    dst.write(parche)

# Mostrar la imagen con los recortes
plt.figure(figsize=(10, 10))
plt.imshow(image[:, :, 0])
plt.title('Zonas de recorte')
plt.axis('off')
plt.show()
