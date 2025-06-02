
""" Scrip para obtener vectores de caracteristicas basado en : Transformada de Fourier, textura, forma y color 
de cada una de las imágenes de entrada. Unión de los resultados y normalización de los vectores mediante Z-Score 
y normalización Min-Máx. Se guardan todos los resultados en formato CSV."""

# Cargar librerías necesarias
import os
import pandas as pd
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage import color, img_as_ubyte, img_as_float
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Rutas a las carpetas de imágenes
smaller_than = r"C:\Users\tere1\OneDrive\Escritorio\TFM\segmentacion de callos\results\smaller_than"
higher_than = r"C:\Users\tere1\OneDrive\Escritorio\TFM\segmentacion de callos\results\higher_than"

# Función para cargar la imagen en color
def load_image(image_path):
    return cv2.imread(image_path)

# procesar cada imagen de una carpeta
def process_images_in_folder(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        img = load_image(image_path)
        if img is None:
            continue
        data.append((filename, img))
    return data

# Función para extraer características GLCM
def extract_glcm_features(folder_path, label=None):
    data = []
    bins32 = np.linspace(0, 255, 33).astype(np.uint8)  # 32 niveles
    images_data = process_images_in_folder(folder_path)  
    
    for filename, img in images_data:
        gray = color.rgb2gray(img)
        gray = img_as_ubyte(gray)
        inds = np.digitize(gray, bins32)

        glcm = graycomatrix(
            inds,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=32,
            symmetric=False,
            normed=False
        )

        features = {
            'filename': filename,
            'contrast': np.mean(graycoprops(glcm, 'contrast')),
            'dissimilarity': np.mean(graycoprops(glcm, 'dissimilarity')),
            'homogeneity': np.mean(graycoprops(glcm, 'homogeneity')),
            'energy': np.mean(graycoprops(glcm, 'energy')),
            'correlation': np.mean(graycoprops(glcm, 'correlation')),
            'ASM': np.mean(graycoprops(glcm, 'ASM'))
        }

        if label is not None:
            features['label'] = label

        data.append(features)

    return pd.DataFrame(data)


# Función para extraer propiedades de forma
def extract_shape_features(folder_path, label_name=None):
    data = []
    images_data = process_images_in_folder(folder_path)  
    
    for filename, image in images_data:
        # Si es RGB, convertir a escala de grises
        if image.ndim == 3:
            image = color.rgb2gray(image)

        # Umbral automático con Otsu
        thresh = threshold_otsu(image)
        binary = image > thresh

        # Etiquetar regiones conectadas
        labeled = label(binary)

        # Extraer propiedades de todas las regiones
        regions = regionprops(labeled)

        if regions:
            # Seleccionar la región más grande (la principal)
            largest_region = max(regions, key=lambda r: r.area)

            # Extraer las propiedades de la región más grande
            props = {
                'filename': filename,
                'area': largest_region.area,
                'perimeter': largest_region.perimeter,
                'eccentricity': largest_region.eccentricity,
                'extent': largest_region.extent,
                'solidity': largest_region.solidity,
                'orientation': largest_region.orientation,
                'major_axis_length': largest_region.major_axis_length,
                'minor_axis_length': largest_region.minor_axis_length
            }

            if label_name:
                props['label'] = label_name

            data.append(props)

    return pd.DataFrame(data)


# Función para realizar la Transformada de Fourier y extraer características
def extract_fourier_features(folder_path, label=None):
    data = []
    images_data = process_images_in_folder(folder_path)  
    
    for filename, img in images_data:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = img_as_float(gray)  

        # Transformada de Fourier y desplazar el cero a la mitad
        f_transform = np.fft.fft2(gray)
        f_transform_shifted = np.fft.fftshift(f_transform)  # Desplazar las frecuencias a la zona central

        # Calcula la magnitud de la transformada de Fourier
        magnitude = np.abs(f_transform_shifted)

        # Características estadísticas de la magnitud
        mean_magnitude = np.mean(magnitude)  
        std_magnitude = np.std(magnitude)    

        # Para obtener información sobre las frecuencias de bajo y alto rango
        low_freq_magnitude = np.mean(magnitude[:magnitude.shape[0]//2, :magnitude.shape[1]//2])  # Frecuencias bajas
        high_freq_magnitude = np.mean(magnitude[magnitude.shape[0]//2:, magnitude.shape[1]//2:])  # Frecuencias altas

        features = {
            'filename': filename,
            'mean_magnitude': mean_magnitude,
            'std_magnitude': std_magnitude,
            'low_freq_magnitude': low_freq_magnitude,
            'high_freq_magnitude': high_freq_magnitude
        }

        if label is not None:
            features['label'] = label

        data.append(features)

    return pd.DataFrame(data)


# Función para extraer características de color
def extract_color_features(folder_path, label=None):
    data = []
    
    images_data = process_images_in_folder(folder_path)
    
    for filename, img in images_data:
        # Convertir la imagen a espacio de color HSV
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Extraer los canales HSV
        hue = hsv_image[:, :, 0].mean()      # Canal Hue 
        saturation = hsv_image[:, :, 1].mean() # Canal Saturation 
        brightness = hsv_image[:, :, 2].mean() # Canal Value 

        # Extraer los canales RGB
        rgb_mean = np.mean(img, axis=(0, 1))  # Promedio de los canales RGB

        features = {
            'filename': filename,
            'rgb_mean_r': rgb_mean[2],   # Canal rojo
            'rgb_mean_g': rgb_mean[1],   # Canal verde
            'rgb_mean_b': rgb_mean[0],   # Canal azul
            'hue': hue,
            'saturation': saturation,
            'brightness': brightness
        }
        
        if label is not None:
            features['label'] = label

        data.append(features)

    return pd.DataFrame(data)



#Normalización Z-score (media 0 desvt 1)
def normalize_zscore(*dataframes):
    # Creamos un diccionario para almacenar los dataframes normalizados
    normalized_dfs = {}
    
    # Iterar sobre cada dataframe
    for name, df in zip(['glcm', 'shape', 'fourier', 'color'], dataframes):
        # Guardamos las etiquetas y nombres de archivo
        labels = df['label']
        filenames = df['filename']
        
        # Seleccionar solo las columnas numéricas
        X = df.drop(columns=['label', 'filename'])
        
        # Normalizar los datos con Z-score
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Crear un nuevo DataFrame con los datos normalizados
        normalized_df = pd.DataFrame(X_scaled, columns=X.columns)
        normalized_df['filename'] = filenames
        normalized_df['label'] = labels
        
        # Guardamos el dataframe normalizado
        normalized_dfs[name] = normalized_df
        
    return normalized_dfs


# Normalización min-max
def normalize_minmax(*dataframes):
    normalized_dfs = {}
    
    for name, df in zip(['glcm', 'shape', 'fourier', 'color'], dataframes):
        labels = df['label']
        filenames = df['filename']
        
        features = df.drop(columns=['filename', 'label'])  
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        
        normalized_df = pd.DataFrame(scaled_features, columns=features.columns)
        normalized_df['filename'] = filenames
        normalized_df['label'] = labels
        
        normalized_dfs[name] = normalized_df
    
    return normalized_dfs



# Aplicar funciones a las carpetas y etiquetas proporcionadas
df_smaller_glcm = extract_glcm_features(smaller_than, label='smaller')
df_higher_glcm = extract_glcm_features(higher_than, label='higher')

df_shape_smaller = extract_shape_features(smaller_than, label_name='smaller')
df_shape_higher = extract_shape_features(higher_than, label_name='higher')

df_smaller_fourier = extract_fourier_features(smaller_than, label='smaller')
df_higher_fourier = extract_fourier_features(higher_than, label='higher')

df_smaller_color = extract_color_features(smaller_than, label='smaller')
df_higher_color = extract_color_features(higher_than, label='higher')

# Mostrar cuantas imagenes procesa
print("Imágenes procesadas (smaller GLCM):", len(df_smaller_glcm))
print("Imágenes procesadas (higher GLCM):", len(df_higher_glcm))
print("Imágenes procesadas (smaller Shape):", len(df_shape_smaller))
print("Imágenes procesadas (higher Shape):", len(df_shape_higher))
print("Imágenes procesadas (smaller Fourier):", len(df_smaller_fourier))
print("Imágenes procesadas (higher Fourier):", len(df_higher_fourier))
print("Imágenes procesadas (smaller color):", len(df_smaller_color))
print("Imágenes procesadas (higher color):", len(df_higher_color))

# Llamar la función para normalizar todos los DataFrames
# Concatenar 'smaller' y 'higher' de cada grupo
glcm_all = pd.concat([df_smaller_glcm, df_higher_glcm], ignore_index=True)
shape_all = pd.concat([df_shape_smaller, df_shape_higher], ignore_index=True)
fourier_all = pd.concat([df_smaller_fourier, df_higher_fourier], ignore_index=True)
color_all = pd.concat([df_smaller_color, df_higher_color], ignore_index=True)

# Normalización Z-score
normalized_z = normalize_zscore(glcm_all, shape_all, fourier_all, color_all)

# Normalización MinMax
normalized_mm = normalize_minmax(glcm_all, shape_all, fourier_all, color_all)

# Acceder al DataFrame normalizado GLCM (z-score)
#df_glcm_z = normalized_z['glcm']

# Acceder al mismo, pero con Min-Max
#df_glcm_mm = normalized_mm['glcm']


# Guardar los originales
# Guardar las características 
df_smaller_glcm.to_csv("features_glcm_smaller.csv", index=False)
df_higher_glcm.to_csv("features_glcm_higher.csv", index=False)

df_shape_smaller.to_csv("features_shape_smaller.csv", index=False)
df_shape_higher.to_csv("features_shape_higher.csv", index=False)

df_smaller_fourier.to_csv("features_fourier_smaller.csv", index=False)
df_higher_fourier.to_csv("features_fourier_higher.csv", index=False)

df_smaller_color.to_csv("features_color_smaller.csv", index=False)
df_higher_color.to_csv("features_color_higher.csv", index=False)

# Guardar las características normalizadas por Z-score
df_glcm_z = normalized_z['glcm']
df_glcm_z.to_csv("features_glcm_all_zscore.csv", index=False)

df_shape_z = normalized_z['shape']
df_shape_z.to_csv("features_shape_all_zscore.csv", index=False)

df_fourier_z = normalized_z['fourier']
df_fourier_z.to_csv("features_fourier_all_zscore.csv", index=False)

df_color_z = normalized_z['color']
df_color_z.to_csv("features_color_all_zscore.csv", index=False)

# Guardar las características normalizadas por Min-Max
df_glcm_mm = normalized_mm['glcm']
df_glcm_mm.to_csv("features_glcm_all_minmax.csv", index=False)

df_shape_mm = normalized_mm['shape']
df_shape_mm.to_csv("features_shape_all_minmax.csv", index=False)

df_fourier_mm = normalized_mm['fourier']
df_fourier_mm.to_csv("features_fourier_all_minmax.csv", index=False)

df_color_mm = normalized_mm['color']
df_color_mm.to_csv("features_color_all_minmax.csv", index=False)