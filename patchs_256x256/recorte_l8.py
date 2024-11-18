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

def adjust_size(img, output_size):
  shape = img.shape
  x, y = shape[1], shape[0]
  if y < x:
    if y % output_size != 0:
      y = y- (y % output_size)
    return img[:y,:x-(x-y)]
  else:
    if x % output_size != 0:
      x = shape[1] - (x % output_size)
    return img[:y-(y-x),:x]

def crop_image(img=None, output_size=256):
  img_gray = img[:, :, 0]
  img_gray = cv2.GaussianBlur(img[:,:,0], (11, 11), 0)
  val, bin_mask = cv2.threshold(img_gray,1,255,cv2.THRESH_BINARY)
  edged = cv2.Canny(np.uint8(bin_mask), 10, 250, apertureSize=3)
  (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  idx = 0
  for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      if w>200 and h>200:
          idx+=1
          pad_x, pad_y = 100, 100
          new_img=img[y+pad_y:y+h-pad_y,x+pad_x:x+w-pad_x]

  new_img = adjust_size(new_img, output_size=output_size)
  return new_img

def deskew_image(image_array):
    # image_array with shape (xxxx,yyyy, channels)
    img_before = image_array[:, :, 0]
    img_before_gray = img_before.copy()

    val, bin_mask = cv2.threshold(np.uint8(img_before_gray),0,255,cv2.THRESH_BINARY)

    img_edges = cv2.Canny(np.uint8(bin_mask), 100, 100, apertureSize=3)

    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []

    if (lines is not None):
        for x1, y1, x2, y2 in lines[0]:
          angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
          angles.append(angle)

        median_angle = np.median(angles)

    if median_angle <= 0.0:
       median_angle = 90 + median_angle

    if (median_angle != 0):
          img_rotated = ndimage.rotate(image_array, median_angle, order=2)
    else:
          img_rotated = image_array

    img_crop = crop_image(img_rotated, output_size=256)
    # Retorna la imagen de n bandas recortada en un factor de 256
    return img_crop

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
    deskew = deskew_image(stack_image)

    # Shape de la imagen de salida
    print(f'Final shape of image is {deskew.shape}.')



    # Giving values Reflectance
    with open(metadata_path) as f:
        # Loading metada.xml file
        tree = ET.parse(f)
        root = tree.getroot()

        MULT = float (root.find(f'.//REFLECTANCE_MULT_BAND_1').text)
        ADD = float (root.find(f'.//REFLECTANCE_ADD_BAND_1').text)
        SUN = float (root.find(f'.//SUN_ELEVATION').text)


    #return deskew

    return ((deskew * MULT) + ADD)/((math.sin(math.radians(SUN))))
    #return (deskew * MULT) + ADD



image = open_bands_and_deskew([2,3,4,5,6,7], direccion)
print("Image shape: ", image.shape)


"""

directorio_parches = f'{direccion}_parches/'

alto = image.shape[0]
ancho = image.shape[1]

parche_size = 256

profile = {
    'count' : image.shape[2],
    'height' : parche_size,
    'width' : parche_size,
    'dtype' : 'float64'
}

if not os.path.exists(directorio_parches):
    os.makedirs(directorio_parches)

for i, y in enumerate(range(0, alto, parche_size)):
    for j, x in enumerate(range(0, ancho, parche_size)):
        parche = image[y:y+parche_size, x:x+parche_size]
        parche = parche.transpose(2, 0, 1)
        nombre_parche = f"{direccion}_{i+1:03d}_{j+1:03d}.TIF"
        ruta_parche = os.path.join(directorio_parches, nombre_parche)

        with rasterio.open(ruta_parche, 'w', **profile) as dst:
            dst.write(parche)


# Extraer la primera banda
first_band = image[:, :, 0]

# Normalizar la banda a 8 bits (0-255) para visualización
first_band_normalized = cv2.normalize(first_band, None, 0, 255, cv2.NORM_MINMAX)
first_band_normalized = np.uint8(first_band_normalized)

# Convertir la banda a una imagen en escala de grises
image_with_grid = cv2.cvtColor(first_band_normalized, cv2.COLOR_GRAY2BGR)

# Dibujar los recortes en la imagen
for y in range(0, alto, parche_size):
    for x in range(0, ancho, parche_size):
        cv2.rectangle(image_with_grid, (x, y), (x + parche_size, y + parche_size), (0, 255, 0), 2)



############################################
# Guardar la imagen con el grid de recortes en un directorio específico
output_directory ="/home/cornelius/Documentos/Glaciar_dataset_C2_L1/patch_003070/Grid_patch_003070/"
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
plt.title('Zonas de recorte patch 003070')
plt.axis('off')
plt.show()
"""
