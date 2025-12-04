# 🛰️ Segmentación semántica del dataset FLAIR con SegNet  
### Adaptación del código base de *“Beyond RGB: Very High Resolution Urban Remote Sensing With Multimodal Deep Networks”*

**Autor:** Avdoni Sanchez Reinoso  
**Programa:** Maestría en Geomática  
**Año:** 2025  

---

## 📘 Descripción general

Este repositorio contiene una implementación del modelo **SegNet** para **segmentación semántica** aplicada al dataset **FLAIR #1**.  
La arquitectura y la organización del código se basan en el trabajo de:

> Audebert, N., Le Saux, B., & Lefèvre, S.  
> *Beyond RGB: Very High Resolution Urban Remote Sensing With Multimodal Deep Networks.*

Las principales características de este proyecto son:

- Uso de **imágenes aéreas multiespectrales** (RGB + NIR + nDSM) del dataset FLAIR.  
- Implementación de **SegNet** como red encoder–decoder totalmente convolucional.  
- Clasificación de **13 clases** de uso/cobertura del suelo.  
- Entrenamiento **desde cero** (sin VGG-16 preentrenado).  
- Entrenamiento optimizado con **mixed precision (AMP)**, scheduler de LR y checkpoints periódicos.  
- Generación de salidas de inferencia en formato RGB / Ground Truth / Predicción.

---

## 🎯 Objetivo del proyecto

El objetivo de este trabajo es:

- **Replicar y adaptar** la estructura del código de Audebert et al. para trabajar con el dataset **FLAIR #1**.  
- Entrenar un modelo SegNet capaz de segmentar 13 clases a partir de 5 canales de entrada (R, G, B, NIR, nDSM).  
- **Evaluar el rendimiento** del modelo (accuracy, F1-score por clase, Kappa, matriz de confusión).  
- Explorar **hiperparámetros adecuados** para este conjunto de datos bajo un entorno computacional limitado, con el fin de obtener resultados comparables con otros trabajos de segmentación semántica en teledetección.

---

## 🗂️ Dataset FLAIR

El dataset **FLAIR** es provisto por el Institut National de l’Information Géographique et Forestière (IGN, Francia):  
🔗 https://ignf.github.io/FLAIR/

Cada parche incluye:

- Imagen aérea de **512×512 px** a **0.2 m** de resolución espacial.  
- 5 canales: **Red, Green, Blue, Near Infrared (NIR) y nDSM**.  
- Máscara de segmentación a 512×512 px con **19 clases**, de las cuales en este proyecto se usan **13** (baseline).

### Clases utilizadas (13)

1. Building  
2. Pervious surface  
3. Impervious surface  
4. Bare soil  
5. Water  
6. Coniferous  
7. Deciduous  
8. Brushwood  
9. Vineyard  
10. Herbaceous vegetation  
11. Agricultural land  
12. Plowed land  
13. Other  

---

### 🔍 Uso parcial del dataset FLAIR

El dataset completo FLAIR contiene más de **60 000 parches** en train/val, además de dos conjuntos de test.  
Debido a las **limitaciones computacionales** del entorno local (en particular una GPU NVIDIA RTX 5060 Ti de 8 GB de VRAM), en este trabajo **no se utilizó la totalidad del dataset**.

En su lugar, se seleccionó un **subconjunto representativo**, garantizando que cada imagen incluyera al menos una de las 13 clases objetivo. El tamaño final empleado fue:

- **12 000 imágenes** para entrenamiento (`train`)  
- **2 400 imágenes** para validación (`val`)  
- **2 400 imágenes** para prueba (`test`)

Este muestreo mantiene la presencia de las clases principales y hace viable entrenar SegNet sin desbordar los recursos de hardware, preservando la utilidad del modelo para análisis y comparación de resultados.

---

## 💻 Entorno de ejecución

El código fue ejecutado en un entorno local con las siguientes características principales:

- **Python 3.10**  
- Entorno virtual gestionado con **Anaconda**  
- **PyTorch (Nightly)** con soporte para **CUDA 12.8** (necesario para GPUs NVIDIA serie 5000).  
- Aceleración por GPU: **NVIDIA RTX 5060 Ti (8 GB VRAM)**  

Librerías principales:

- `torch`, `torchvision`  
- `numpy`  
- `matplotlib`  
- `scikit-learn`  
- `scikit-image`  
- `tifffile`  
- `tqdm`

Las versiones exactas pueden gestionarse mediante un archivo `requirements.txt`.

---

## 🧠 Arquitectura del modelo (SegNet)

El modelo implementado es una versión clásica de **SegNet**, adaptada a:

- **5 canales de entrada** (`IN_CHANNELS = 5`) para trabajar con RGB + NIR + nDSM.  
- **13 clases de salida** (`N_CLASSES = 13`).

Características principales:

- Encoder–decoder simétrico basado en bloques conv–BatchNorm–ReLU.  
- Uso de `MaxPool2d` con `return_indices=True` y `MaxUnpool2d` para preservar información espacial.  
- Última capa con salida de logits para aplicar `CrossEntropyLoss`.  
- Entrenamiento **desde cero**, sin inicialización con VGG-16 preentrenado.

---

## ⚙️ Hiperparámetros principales

Algunos de los hiperparámetros más relevantes utilizados en los experimentos:

| Parámetro       | Valor                   | Descripción                                          |
|----------------|-------------------------|------------------------------------------------------|
| `WINDOW_SIZE`  | (256, 256)             | Tamaño de los recortes aleatorios (random crops)    |
| `BATCH_SIZE`   | 10                   | Dependiente de la VRAM disponible                    |
| `IN_CHANNELS`  | 5                      | R, G, B, NIR, nDSM                                   |
| `N_CLASSES`    | 13                     | Número de clases de salida                           |
| `LR`           | 0.005                   | Learning rate base                                  |
| `optimizer`    | SGD                    | con `momentum=0.9`, `weight_decay=5e-4`             |
| `scheduler`    | MultiStepLR            | `milestones=[25, 35, 45]`, `gamma=0.1`              |
| `save_epoch`   | 5                      | Frecuencia de guardado de checkpoints               |
| `AMP`          | Activado               | `torch.amp.autocast` + `GradScaler`                 |

Se utiliza además `data augmentation` sencillo (rotaciones y flips) en el conjunto de entrenamiento, junto con recortes aleatorios controlados por `WINDOW_SIZE`.

---

## 📂 Estructura del repositorio

Estructura del proyecto:

📁 raíz del repositorio
│
├── README.md                       # Este archivo
├── requirements.txt                # Dependencias del proyecto
├── SegNet_Data_FLAIR.ipynb              # Notebook principal
│
├── Evaluacion_mod_FLAIR/           # Matrix y parámetros de evaluación
├── Mod_SegNet_FLAIR_epoch/         # Checkpoints del modelo por época
├── Graf_perd/                      # Gráficas de pérdida (loss vs epochs)
├── Predic_FLAIR_img/               # Ejemplos RGB / Ground Truth / Predicción
├── Inferencias_FLAIR_tiles/        # Predicciones por tile en test
└── modelo_SegNet_FLAIR/            # Modelo entrenado


## 🚀 Entrenamiento

El entrenamiento del modelo se realiza mediante la función `train_model`, la cual implementa un ciclo completo de optimización incluyendo:

- Iteración sobre el `DataLoader` de entrenamiento.
- Uso de **mixed precision (AMP)** para mejorar la eficiencia y reducir el consumo de VRAM.
- Actualización del *scheduler* al final de cada época para ajustar la tasa de aprendizaje.
- Ejecución de procesos adicionales cada `save_epoch` épocas:
  - Guardado de la curva de pérdida acumulada en la carpeta `Graf_perd/`.
  - Evaluación del modelo utilizando el conjunto de validación.
  - Generación y almacenamiento de ejemplos RGB / Ground Truth / Predicción en `Predic_FLAIR_img/`.
  - Almacenamiento de un *checkpoint* con estado del modelo, optimizador y métricas dentro de `Mod_SegNet_FLAIR_epoch/`.

### Ejemplo de ejecución del entrenamiento

"train_model(net, train_loader, val_loader, optimizer, epochs=50, scheduler=scheduler)"

## Evaluación del modelo

El modelo fue evaluado utilizando un conjunto de prueba independiente del entrenamiento y validación.  
La evaluación incluye:

- Exactitud global (Overall Accuracy)
- Matriz de confusión normalizada
- F1-score por clase
- Kappa de Cohen
- Visualizaciones comparativas entre *Ground Truth* y *Predicción*
- Inferencias colorizadas por tile

La función de evaluación genera automáticamente figuras y métricas, y permite almacenar los resultados para futuras comparaciones entre configuraciones o modelos.

---

## Inferencia y visualización de resultados

El notebook incluye herramientas para:

- Realizar inferencia sobre imágenes del conjunto de prueba.
- Guardar resultados visuales en carpetas dedicadas.
- Generar composiciones RGB–GT–Predicción para inspección manual.
- Exportar las predicciones en formato PNG colorizado.

Las predicciones se almacenan respetando el nombre original del parche, lo que facilita su trazabilidad y comparación.

---

## Limitaciones del presente experimento

Aunque los resultados obtenidos permiten evaluar el rendimiento de SegNet sobre FLAIR, existen ciertas limitaciones:

- No se utilizó preentrenamiento en ImageNet (como en VGG-16), lo cual podría mejorar la generalización.
- Se trabajó con una fracción del dataset debido a restricciones de cómputo.
- A pesar de incorporar data augmentation, podrían explorarse técnicas más avanzadas.
- La arquitectura SegNet es relativamente antigua en comparación con modelos modernos como DeepLabv3+, U-Net++, HRNet o SegFormer.

Estas limitaciones abren oportunidades para investigaciones futuras.

---

## Referencias

Audebert, N., Le Saux, B., & Lefèvre, S. (2018).  
**Beyond RGB: Very High Resolution Urban Remote Sensing With Multimodal Deep Networks.**  
ISPRS Journal of Photogrammetry and Remote Sensing.

IGN (2023).  
**FLAIR: A Nationwide Dataset for Land Cover Semantic Segmentation.**  
https://ignf.github.io/FLAIR/

---

## Licencia

Este repositorio contiene únicamente código desarrollado por el autor.  
El dataset FLAIR **no se redistribuye** y debe obtenerse desde el sitio oficial del Institut National de l’Information Géographique et Forestière (IGN).  
El uso del código está permitido para fines académicos y experimentales, salvo indicación contraria.

---

