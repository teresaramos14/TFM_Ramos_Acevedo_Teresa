## 📁 Contenido del repositorio

### 📂 `Segmentación`
Contiene los elementos necesarios para detectar y recortar automáticamente callos desde imágenes de placas de Petri:

- `Recortes_callos.py`: script de segmentación que utiliza un modelo YOLOv8 entrenado.
- `cuadrantes_a_recortar_higher.csv` y `cuadrantes_a_recortar_smaller.csv`: listados de celdas a recortar según el tamaño del callo.
- `Selección_tfm.xls`: archivo Excel con la selección de placas y celdas utilizadas para el análisis.

---

### 📂 `Extracción_vectores_características`
Incluye el procesamiento de imágenes para obtener representaciones numéricas de cada callo:

- `Extraccion_vectores_normalizacion.py`: script para extraer características de:
  - Color
  - Textura
  - Forma
  - Transformada de Fourier
  - También realiza normalización con Min-Max y Z-Score
- CSVs: archivos de vectores extraídos y normalizados, organizados por tipo y nombre de muestra.

---

### 📂 `Clustering`
Contiene los resultados y configuraciones de agrupamiento no supervisado:

- Subcarpetas con cada algoritmo utilizado:
  - `KMeans/`
  - `KMedoids/`
  - `MeanShift/`
  - `DBSCAN/`
  - `GMM/`
  - `Hierarchical/`
- Cada carpeta incluye los resultados por vector de características.
