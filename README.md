---
title: Alertas Tempranas Permanencia ISC
emoji: 🎓
colorFrom: green
colorTo: green
sdk: docker
app_port: 8501
pinned: false
short_description: Alertas tempranas de deserción - ISC UdeC Fusagasugá
---

# Prototipo · Sistema de Alertas Tempranas para la Permanencia

Interfaz en Streamlit que estima el riesgo de deserción de los Creadores de Oportunidades de Ingeniería de
Sistemas y Computación (UdeC, Fusagasugá).

## Estructura

```
app/
├── app.py                       Interfaz (3 pestañas: Creador de Oportunidades, grupo CSV, sobre el modelo)
├── train_model.py               Entrena el modelo final desde models/*.csv
├── modelo/
│   ├── modelo_desercion.joblib  Pipeline entrenado (scikit-learn)
│   └── metadata.json            Variables, categorías, umbrales y métricas
├── analisis/
│   ├── comparar_modelos.py      Comparación de 5 modelos con validación cruzada 5×5
│   └── comparacion_modelos.csv  Resultados
├── ejemplo_grupo_sintetico.csv  25 Creadores de Oportunidades ficticios para probar la carga por lote
├── requirements.txt
└── .streamlit/config.toml       Tema (azul institucional)
```

## Ejecutar localmente

```bash
# 1. Traer el repositorio a esta carpeta (si aún no está)
git init && git remote add origin https://github.com/Jpvillarraga/Desercion_estudiantil.git
git pull origin main

# 2. Entorno e instalación
python3 -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt

# 3. (Opcional) reentrenar el modelo — necesita models/X_train.csv, etc.
python app/train_model.py

# 4. Abrir la interfaz
cd app && streamlit run app.py
```

Se abre en http://localhost:8501.

## Modelo

Regresión logística (L2, C=0.5), sin variables sensibles. Umbrales: bajo < 15 %,
medio 15–40 %, alto ≥ 40 %. Ver la pestaña "Sobre el modelo" y `analisis/`.

## Nota para el repositorio

El `.gitignore` actual excluye `*.joblib` y `models/`. Para versionar el modelo agrega:

```
!app/modelo/
!app/modelo/*.joblib
```
