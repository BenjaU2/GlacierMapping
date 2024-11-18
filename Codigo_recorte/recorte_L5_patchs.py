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

    return stack_image

def extract_patches_with_padding(image, patch_size=256):
    """
    Función para recortar una imagen grande en parches de tamaño fijo,
    añadiendo relleno (padding) a los parches de los bordes que no alcanzan el tamaño completo.

    :param image: Imagen de entrada de tamaño (h, w, c), donde h es la altura, w es el ancho, y c es el número de canales.
    :param patch_size: Tamaño de los parches (ancho y alto, por defecto 256).
    :return: Lista de parches de tamaño (256, 256, 6).
    """
    image_patches = []

    # Obtener el tamaño de la imagen
    height, width, channels = image.shape

    # Recorrer la imagen en pasos del tamaño del parche
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            # Definir los límites de los parches
            i_end = min(i + patch_size, height)  # Limitar al borde de la imagen
            j_end = min(j + patch_size, width)

            # Recortar el parche
            patch = image[i:i_end, j:j_end, :]

            # Si el parche es más pequeño que el tamaño objetivo (256, 256), añadir padding
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                patch = np.pad(patch,
                               ((0, patch_size - patch.shape[0]),  # Padding en altura
                                (0, patch_size - patch.shape[1]),  # Padding en ancho
                                (0, 0)),  # No agregar padding en los canales
                               mode='constant', constant_values=0)  # Usar relleno con valor 0 (negro)

            image_patches.append(patch)

    print(f"Total patches extracted: {len(image_patches)}")
    return image_patches

image = open_bands_and_deskew([1,2,3,4,5,7], direccion)
print(type(image))
print("Tamaño de la imagene apilada ", image.shape)
#directorio_parches = f'{direccion}_parches/'

# Llamar a la función con la imagen apilada
image_patches = extract_patches_with_padding(image)

# Verificar el tamaño de uno de los parches
print(f"Tamaño de un parche: {image_patches[0].shape}")
print(f"Valor maximo de pixel del primer patch: {np.max(image_patches[0])}")


# Llamada a la función para guardar los parches
directorio_parches = f'{direccion}_parches/'

# Crear el directorio si no existe
if not os.path.exists(directorio_parches):
    os.makedirs(directorio_parches)

#height, width, _ = original_image_shape
height = image.shape[0]
width = image.shape[1]

parche_size = 256

profile = {
'count' : image.shape[2],
'height' : parche_size,
'width' : parche_size,
'dtype' : 'float64'
}

# Otra opción para recorrer:
for i, y in enumerate(range(0, height, parche_size)):
    for j, x in enumerate(range(0, width, parche_size)):
        parches = image[y:y+parche_size, x:x+parche_size]
        parches = parches.transpose(2, 0, 1)
        nombre_parche = f"{direccion}_{i+1:03d}_{j+1:03d}.TIF"
        ruta_parche = os.path.join(directorio_parches, nombre_parche)

        with rasterio.open(ruta_parche, 'w', **profile) as dst:
            dst.write(parches)



#save_patches_to_directory(image, directorio_parches, patch_size=256)

# Llamar a la función para visualizar los cortes en la primera banda
#plot_image_with_patches(image)
#directorio_parches = f'{direccion}_parches/'



# Extraer la primera banda
first_band = image[:, :, 0]

# Normalizar la banda a 8 bits (0-255) para visualización
first_band_normalized = cv2.normalize(first_band, None, 0, 255, cv2.NORM_MINMAX)
first_band_normalized = np.uint8(first_band_normalized)

# Convertir la banda a una imagen en escala de grises
image_with_grid = cv2.cvtColor(first_band_normalized, cv2.COLOR_GRAY2BGR)

# Dibujar los recortes en la imagen
for y in range(0, height, parche_size):
    for x in range(0, width, parche_size):
        cv2.rectangle(image_with_grid, (x, y), (x + parche_size, y + parche_size), (0, 255, 0), 2)



############################################
# Guardar la imagen con el grid de recortes en un directorio específico
output_directory ="/home/cornelius/Documentos/Glaciar_dataset_C2_L1/landsat5_C2L1/patch_008066/Grid_patch_008066/"
#output_directory = f'{direccion}_grid/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

output_path = os.path.join(output_directory, f'{direccion}_grid.png')
cv2.imwrite(output_path, image_with_grid)

print(f'Imagen con grid guardada en: {output_path}')
#########################################################


# Mostrar la imagen con los recortes
plt.figure(figsize=(10, 10))
plt.imshow(image_with_grid)
plt.title('Zonas de recorte patch 008066')
plt.axis('off')
plt.show()

