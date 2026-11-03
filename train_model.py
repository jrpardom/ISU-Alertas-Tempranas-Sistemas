"""Entrena el modelo final del Sistema de Alertas Tempranas y lo guarda en modelo/.

Uso (desde la raíz del repositorio):
    python app/train_model.py

Lee models/X_train.csv, X_test.csv, y_train.csv, y_test.csv (variables ya
recategorizadas, sin identificadores) y produce:
    app/modelo/modelo_desercion.joblib   pipeline de scikit-learn
    app/modelo/metadata.json             variables, categorías, umbrales y métricas
"""
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent
REPO = APP_DIR.parent
DATA = REPO / "models"
OUT = APP_DIR / "modelo"

# Variables excluidas a propósito: datos sensibles (Ley 1581 de 2012) y de salud.
# ORIENTACION_CAT además se comporta como artefacto de recolección
# ("No responde" = 41 % de deserción frente a 3 %), no como causa.
EXCLUIDAS = [
    "ORIENTACION_CAT", "ES_GRUPO_ETNICO", "GRUPO_ETNICO_CAT", "VICTIMA_VIOLENCIA",
    "CONDICION_DISCAPACIDAD", "IMC_CAT", "HABITOS_RIESGO", "VIVE_PERSONAS_PATOLOGIAS",
]

# Umbrales de alerta sobre la probabilidad estimada
UMBRAL_MEDIO = 0.15
UMBRAL_ALTO = 0.40


def cargar():
    X = pd.concat([pd.read_csv(DATA / "X_train.csv", index_col=0),
                   pd.read_csv(DATA / "X_test.csv", index_col=0)])
    y = pd.concat([pd.read_csv(DATA / "y_train.csv", index_col=0).iloc[:, 0],
                   pd.read_csv(DATA / "y_test.csv", index_col=0).iloc[:, 0]])
    X.columns = [c.lstrip("﻿") for c in X.columns]
    X = X.astype(str)
    return X[[c for c in X.columns if c not in EXCLUIDAS]], y.astype(int)


def construir(cols):
    prep = ColumnTransformer([("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cols)])
    clf = LogisticRegression(C=0.5, max_iter=5000)
    return Pipeline([("prep", prep), ("clf", clf)])


def main():
    X, y = cargar()
    cols = list(X.columns)
    pipe = construir(cols)

    # Evaluación honesta: probabilidades fuera de muestra con validación cruzada repetida
    aucs, aps, briers, recs, precs = [], [], [], [], []
    for rep in range(5):
        pr = cross_val_predict(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=rep),
                               method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, pr)); aps.append(average_precision_score(y, pr))
        briers.append(brier_score_loss(y, pr))
        recs.append(recall_score(y, pr >= UMBRAL_MEDIO)); precs.append(precision_score(y, pr >= UMBRAL_ALTO))

    pipe.fit(X, y)
    OUT.mkdir(exist_ok=True)
    joblib.dump(pipe, OUT / "modelo_desercion.joblib")

    meta = {
        "modelo": "Regresión logística (L2, C=0.5)",
        "sklearn_version": sklearn.__version__,
        "n_registros": int(len(y)),
        "tasa_desercion": round(float(y.mean()), 3),
        "variables": cols,
        "categorias": {c: sorted(X[c].unique().tolist()) for c in cols},
        "moda": {c: X[c].mode().iat[0] for c in cols},
        "excluidas": EXCLUIDAS,
        "umbrales": {"medio": UMBRAL_MEDIO, "alto": UMBRAL_ALTO},
        "metricas": {
            "roc_auc": round(float(np.mean(aucs)), 3),
            "pr_auc": round(float(np.mean(aps)), 3),
            "brier": round(float(np.mean(briers)), 3),
            "recall_alerta_media_o_alta": round(float(np.mean(recs)), 3),
            "precision_alerta_alta": round(float(np.mean(precs)), 3),
        },
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta["metricas"], indent=2))
    print("Modelo guardado en", OUT)


if __name__ == "__main__":
    main()
