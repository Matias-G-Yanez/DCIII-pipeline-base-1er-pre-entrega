# Checkpoint: Pipeline de Entrenamiento, Validación y Clasificador Base

Repositorio oficial correspondiente a la pre-entrega de **Data Science III: Deep Learning con PyTorch** en Coderhouse.

## 1. Especificaciones del Entorno y Hardware
- **Versión de PyTorch**: `2.11.0+cu130`
- **Dispositivo**: Acelerador gráfico detectado automáticamente (`cuda`).
- **Reproducibilidad**: Semilla global fijada (`seed = 42`) para control de aleatoriedad.

## 2. Arquitectura del Modelo (`nn.Module`)
Se implementó un Perceptrón Multicapa (MLP) modular diseñado para clasificación multiclase:
- **Capa 1**: `Linear(20, 64)` + `ReLU`
- **Capa 2**: `Linear(64, 32)` + `ReLU`
- **Capa de Salida**: `Linear(32, 3)` (emite logits para 3 clases).
- **Parámetros entrenables**: 3,523

## 3. Hiperparámetros y Entrenamiento
- **Optimizador**: `Adam`
- **Learning Rate (`lr`)**: `0.001`
- **Función de Pérdida**: `nn.CrossEntropyLoss`
- **Batch Size**: `32`
- **Épocas**: `25`

## 4. Interpretación de las Curvas de Aprendizaje
- **Curva de Pérdida**: La pérdida de entrenamiento descendió de forma monótona desde `1.04` hasta `0.12`. La pérdida de validación acompañó la tendencia a la baja hasta estabilizarse en torno a `0.33`, demostrando una adecuada generalización sin divergencias propias del sobreajuste prematuro.
- **Curva de Precisión (Accuracy)**: La precisión en validación escaló exitosamente desde un 60% inicial hasta un **88.75%**, validando que el clasificador aprende patrones genuinos sobre datos no vistos.

## 5. Estructura del Repositorio
```text
├── data/                  # Datasets o archivos de referencia
├── notebooks/             # Código fuente del pipeline
│   └── pipeline_base.ipynb
├── loss_curve.png         # Gráfica de convergencia del modelo
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Documentación técnica
