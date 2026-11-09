"""Prototipo — Sistema Inteligente de Alertas Tempranas para la Permanencia.

Ingeniería de Sistemas y Computación · Universidad de Cundinamarca, Fusagasugá.

Ejecutar (desde la carpeta app/ para que cargue el tema de .streamlit/):
    cd app
    streamlit run app.py
"""
from pathlib import Path
import io
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODELO = APP_DIR / "modelo" / "modelo_desercion.joblib"
META = APP_DIR / "modelo" / "metadata.json"

st.set_page_config(
    page_title="Alertas Tempranas · Permanencia ISC",
    page_icon=":material/school:",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Estilo (Minimalismo suizo · verde institucional UdeC · Fira Sans)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;600&family=Fira+Sans:wght@400;500;600;700&display=swap');
:root{
  /* Paleta institucional UdeC (manual de identidad visual) */
  --udec-verde-oscuro:#00482B; --udec-verde:#007B3E; --udec-lima:#79C000; --udec-dorado:#DAAA00;
  --primary:var(--udec-verde); --primary-soft:#E6F2EB; --fg:#1A2E22; --muted:#4D4D4D;
  --card:#FFFFFF; --border:#DCE6DF; --bg:#F6F9F7;
  /* Semáforo de riesgo (siempre acompañado de texto) */
  --alto:#B91C1C; --alto-bg:#FEF2F2; --medio:#8A6A00; --medio-bg:#FDF8E6;
  --bajo:#00796B; --bajo-bg:#E8F5F3;
}
html, body, [class*="css"], .stMarkdown, button, input, label { font-family:'Fira Sans', system-ui, sans-serif; }
.block-container{ padding-top:4rem; max-width:1180px; }
h1,h2,h3{ color:var(--fg); letter-spacing:-0.01em; }
.app-header{ display:flex; flex-direction:column; gap:.25rem; border-bottom:3px solid var(--udec-verde); padding-bottom:1rem; margin-bottom:1.25rem; position:relative; }
.app-header::after{ content:''; position:absolute; left:0; bottom:-3px; width:96px; height:3px; background:var(--udec-lima); }
.app-kicker{ font-size:.8rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--udec-verde); }
.app-header h1.app-title{ font-size:clamp(1.4rem, 1.1rem + 1.4vw, 2rem) !important; font-weight:700; color:var(--udec-verde-oscuro); margin:0; padding:0; line-height:1.2; }
.app-sub{ color:var(--muted); font-size:1rem; margin:0; }
.card{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:1.25rem 1.4rem; }
.result{ border-left-width:6px; }
.result.alto{ border-left-color:var(--alto); background:var(--alto-bg); }
.result.medio{ border-left-color:var(--medio); background:var(--medio-bg); }
.result.bajo{ border-left-color:var(--bajo); background:var(--bajo-bg); }
.level{ display:inline-flex; align-items:center; gap:.4rem; font-weight:700; font-size:.85rem; letter-spacing:.06em; text-transform:uppercase; padding:.25rem .65rem; border-radius:999px; background:#fff; border:1px solid currentColor; }
.level.alto{ color:var(--alto);} .level.medio{ color:var(--medio);} .level.bajo{ color:var(--bajo);}
.prob{ font-family:'Fira Code', monospace; font-size:2.6rem; font-weight:600; color:var(--fg); line-height:1.1; margin:.6rem 0 .2rem; }
.prob-label{ color:var(--muted); font-size:.92rem; }
.meter{ position:relative; height:10px; border-radius:999px; background:linear-gradient(90deg,var(--bajo) 0 15%, var(--medio) 15% 40%, var(--alto) 40% 100%); opacity:.85; margin:1rem 0 .35rem; }
.meter .pin{ position:absolute; top:-5px; width:4px; height:20px; background:var(--fg); border-radius:2px; transform:translateX(-50%); }
.meter-scale{ position:relative; height:1rem; font-size:.75rem; color:var(--muted); font-family:'Fira Code', monospace; }
.meter-scale span{ position:absolute; transform:translateX(-50%); }
.meter-scale span:first-child{ transform:none; } .meter-scale span:last-child{ transform:translateX(-100%); }
.factor{ display:flex; justify-content:space-between; gap:1rem; padding:.55rem 0; border-bottom:1px solid var(--border); font-size:.95rem; }
.factor:last-child{ border-bottom:none; }
.factor .val{ color:var(--muted); }
.factor .tag{ font-family:'Fira Code', monospace; font-size:.8rem; white-space:nowrap; }
.tag.up{ color:var(--alto);} .tag.down{ color:var(--bajo);}
.note{ color:var(--muted); font-size:.88rem; line-height:1.5; }
.kpi{ font-family:'Fira Code', monospace; font-size:1.6rem; font-weight:600; color:var(--fg); }
.kpi-label{ color:var(--muted); font-size:.85rem; }
.card.kpi-card{ height:100%; min-height:110px; }
div[data-testid="stForm"]{ background:var(--card); border:1px solid var(--border); border-radius:12px; }
.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button{ min-height:44px; border-radius:8px; font-weight:600; }
.stFormSubmitButton>button[kind="primaryFormSubmit"]{ background:var(--udec-verde); border-color:var(--udec-verde); }
.stFormSubmitButton>button[kind="primaryFormSubmit"]:hover{ background:var(--udec-verde-oscuro); border-color:var(--udec-verde-oscuro); }
.kpi-card{ border-top:3px solid var(--udec-verde); }
@media (prefers-reduced-motion: reduce){ *{ transition:none !important; animation:none !important; } }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Textos legibles para variables y categorías
# ---------------------------------------------------------------------------
ETIQUETAS = {
    "CIENCIAS BÁSICAS_bin": "Promedio en Ciencias Básicas",
    "CIENCIAS BÁSICAS DE INGENIERÍA_bin": "Promedio en Ciencias Básicas de Ingeniería",
    "INGENIERÍA APLICADA_bin": "Promedio en Ingeniería Aplicada",
    "INSTITUCIONAL_bin": "Promedio en área Institucional",
    "FUENTES_FINANCIACION": "Fuente de financiación",
    "DIFICULTAD_PAGO_MATRICULA": "Dificultad para pagar la matrícula",
    "CARGA_LABORAL": "Carga laboral",
    "TURNO_TRABAJO_ESTUDIO": "Jornada de trabajo / estudio",
    "RECURSOS_TECNOLOGICOS": "Recursos tecnológicos",
    "SISBEN": "Registrado en SISBÉN",
    "ES_VULNERABLE": "Población vulnerable (registro institucional)",
    "PERSONAS_A_CARGO": "Personas a cargo",
    "NUM_HIJOS": "Número de hijos",
    "CARGA_HOGAR": "Composición del hogar",
    "VIVE_PADRES_FAMILIAR": "Vive con padres o familiares",
    "VIVE_PERSONAS_MAYORES": "Vive con personas mayores",
    "HERMANOS_CAT": "Hermanos",
    "NIV_EDUCATIVO_PADRE_CAT": "Nivel educativo del padre",
    "NIV_EDUCATIVO_MADRE_CAT": "Nivel educativo de la madre",
    "TIPO_VIVIENDA_RESIDENCIA": "Tipo de vivienda",
    "ZONA_VIVIENDA_RESIDENCIA": "Zona de residencia",
    "COMUNA": "Comuna / procedencia",
    "HORAS_FUERA_CASA": "Horas fuera de casa",
    "FRECUENCIA_USO_CANALES_ATENCION": "Uso de canales de atención al estudiante",
}

VALORES = {
    "CON_CARGAS": "Sí, tiene personas a cargo", "SIN_CARGAS": "No tiene personas a cargo",
    "ALTA_>20": "Alta (más de 20 h/semana)", "PARCIAL_1_20": "Parcial (1 a 20 h/semana)", "NO_TRABAJA": "No trabaja",
    "BASICOS": "Básicos", "COMPLETOS": "Completos", "SIN_RECURSOS": "Sin recursos",
    "NINGUNO": "Ninguno", "POCOS_1_2": "1 o 2", "VARIOS_3_MAS": "3 o más",
    "CON_NINOS_1_2": "Con 1 o 2 niños", "CON_NINOS_3_MAS": "Con 3 o más niños",
    "HOGAR_SIN_NINOS": "Hogar sin niños", "SOLO_SIN_NINOS": "Vive solo, sin niños", "DESCONOCIDO": "Sin información",
    "ALTO": "Alto", "MEDIO": "Medio", "BAJO": "Bajo", "-1": "Sin información",
    "Sí": "Sí", "Si": "Sí",
}

SECCIONES = {
    "Desempeño académico": ["CIENCIAS BÁSICAS_bin", "CIENCIAS BÁSICAS DE INGENIERÍA_bin",
                            "INGENIERÍA APLICADA_bin", "INSTITUCIONAL_bin"],
    "Situación económica y laboral": ["FUENTES_FINANCIACION", "DIFICULTAD_PAGO_MATRICULA", "CARGA_LABORAL",
                                      "TURNO_TRABAJO_ESTUDIO", "RECURSOS_TECNOLOGICOS", "SISBEN", "ES_VULNERABLE"],
    "Hogar y familia": ["PERSONAS_A_CARGO", "NUM_HIJOS", "CARGA_HOGAR", "VIVE_PADRES_FAMILIAR",
                        "VIVE_PERSONAS_MAYORES", "HERMANOS_CAT", "NIV_EDUCATIVO_PADRE_CAT", "NIV_EDUCATIVO_MADRE_CAT"],
    "Vivienda y entorno": ["TIPO_VIVIENDA_RESIDENCIA", "ZONA_VIVIENDA_RESIDENCIA", "COMUNA",
                           "HORAS_FUERA_CASA", "FRECUENCIA_USO_CANALES_ATENCION"],
}

NIVELES = {
    "alto": ("Riesgo alto", "Priorizar contacto con el estudiante: tutoría académica y remisión a Bienestar Universitario esta semana."),
    "medio": ("Riesgo medio", "Hacer seguimiento: revisar avance en las áreas con notas bajas y ofrecer acompañamiento."),
    "bajo": ("Riesgo bajo", "Sin acción especial. Mantener el seguimiento habitual del semestre."),
}


def etiqueta(col: str) -> str:
    return ETIQUETAS.get(col, col.replace("_", " ").capitalize())


def valor(v: str) -> str:
    return VALORES.get(str(v), str(v))


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    if not MODELO.exists() or not META.exists():
        return None, None
    return joblib.load(MODELO), json.loads(META.read_text(encoding="utf-8"))


modelo, meta = cargar_modelo()

st.markdown(
    """
<div class="app-header">
  <span class="app-kicker">Universidad de Cundinamarca · Ingeniería de Sistemas y Computación · Fusagasugá</span>
  <h1 class="app-title">Alertas tempranas para la permanencia estudiantil</h1>
  <p class="app-sub">Estima el riesgo de deserción de un estudiante a partir de su desempeño académico y su caracterización socioeconómica.</p>
</div>
""",
    unsafe_allow_html=True,
)

if modelo is None:
    st.error("No se encontró el modelo entrenado. Ejecuta primero:  `python app/train_model.py`")
    st.stop()

VARIABLES = meta["variables"]
U_MEDIO, U_ALTO = meta["umbrales"]["medio"], meta["umbrales"]["alto"]


def nivel(p: float) -> str:
    return "alto" if p >= U_ALTO else "medio" if p >= U_MEDIO else "bajo"


def contribuciones(fila: pd.DataFrame) -> pd.DataFrame:
    """Aporte de cada variable al log-odds frente al perfil más común (moda)."""
    prep, clf = modelo.named_steps["prep"], modelo.named_steps["clf"]
    coef = clf.coef_[0]
    base = pd.DataFrame([meta["moda"]])[VARIABLES]
    dif = (prep.transform(fila[VARIABLES]) - prep.transform(base))[0] * coef
    nombres = prep.get_feature_names_out()
    aportes = {}
    for n, d in zip(nombres, dif):
        col = next(c for c in VARIABLES if n.startswith(f"cat__{c}_"))
        aportes[col] = aportes.get(col, 0.0) + d
    out = pd.DataFrame({"variable": list(aportes), "aporte": list(aportes.values())})
    out["valor"] = [valor(fila.iloc[0][c]) for c in out["variable"]]
    return out[out["aporte"].abs() > 0.05].sort_values("aporte", ascending=False)


def tarjeta_resultado(p: float):
    nv = nivel(p)
    titulo, accion = NIVELES[nv]
    pos = min(max(p * 100, 1), 99)
    st.markdown(
        f"""
<div class="card result {nv}" role="status" aria-live="polite">
  <span class="level {nv}">{titulo}</span>
  <div class="prob">{p*100:.0f}%</div>
  <div class="prob-label">Probabilidad estimada de deserción</div>
  <div class="meter" aria-hidden="true" style="background:linear-gradient(90deg,var(--bajo) 0 {U_MEDIO*100:.0f}%, var(--udec-dorado) {U_MEDIO*100:.0f}% {U_ALTO*100:.0f}%, var(--alto) {U_ALTO*100:.0f}% 100%)"><span class="pin" style="left:{pos}%"></span></div>
  <div class="meter-scale"><span style="left:0">0%</span><span style="left:{U_MEDIO*100:.0f}%">{U_MEDIO*100:.0f}%</span><span style="left:{U_ALTO*100:.0f}%">{U_ALTO*100:.0f}%</span><span style="left:100%">100%</span></div>
  <p class="note" style="margin-top:1rem"><strong>Acción sugerida:</strong> {accion}</p>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
tab_ind, tab_lote, tab_modelo = st.tabs(["Evaluar estudiante", "Evaluar grupo (CSV)", "Sobre el modelo"])

with tab_ind:
    col_form, col_res = st.columns([3, 2], gap="large")
    with col_form:
        with st.form("evaluar"):
            entrada = {}
            for i, (seccion, cols) in enumerate(SECCIONES.items()):
                with st.expander(seccion, expanded=i == 0):
                    c1, c2 = st.columns(2)
                    for j, col in enumerate(cols):
                        opciones = meta["categorias"][col]
                        defecto = opciones.index(meta["moda"][col])
                        with (c1 if j % 2 == 0 else c2):
                            entrada[col] = st.selectbox(etiqueta(col), opciones, index=defecto,
                                                        format_func=valor, key=col)
            enviado = st.form_submit_button("Calcular riesgo", type="primary", use_container_width=True)
        st.caption("Los campos vienen con el valor más frecuente de la cohorte. Ajusta los que conozcas del estudiante.")

    with col_res:
        if enviado:
            fila = pd.DataFrame([entrada])[VARIABLES]
            p = float(modelo.predict_proba(fila)[0, 1])
            tarjeta_resultado(p)
            aportes = contribuciones(fila)
            st.markdown("#### Factores que más influyen")
            if aportes.empty:
                st.markdown('<p class="note">El perfil coincide con el más común de la cohorte; ningún factor se aparta de forma notable.</p>', unsafe_allow_html=True)
            else:
                top = pd.concat([aportes.head(4), aportes[aportes["aporte"] < 0].tail(2)]).drop_duplicates("variable")
                html = ['<div class="card">']
                for _, r in top.iterrows():
                    up = r["aporte"] > 0
                    html.append(
                        f'<div class="factor"><div><div>{etiqueta(r["variable"])}</div>'
                        f'<div class="val">{r["valor"]}</div></div>'
                        f'<span class="tag {"up" if up else "down"}">{"▲ aumenta" if up else "▼ reduce"}</span></div>'
                    )
                html.append("</div>")
                st.markdown("".join(html), unsafe_allow_html=True)
                st.markdown('<p class="note">Comparado con el perfil más común de la cohorte. Es una orientación para la conversación con el estudiante, no un diagnóstico.</p>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="card"><h4 style="margin-top:0">Cómo usarlo</h4>'
                '<p class="note">1. Completa las cuatro secciones del formulario.<br>'
                '2. Pulsa <strong>Calcular riesgo</strong>.<br>'
                '3. Revisa el nivel de alerta, la acción sugerida y los factores que más pesan.</p>'
                f'<p class="note"><strong>Niveles:</strong> bajo (&lt;{U_MEDIO*100:.0f}%), medio ({U_MEDIO*100:.0f}–{U_ALTO*100:.0f}%), alto (≥{U_ALTO*100:.0f}%).</p></div>',
                unsafe_allow_html=True,
            )

with tab_lote:
    st.markdown("#### Evaluar un grupo de estudiantes")
    st.markdown(
        '<p class="note">Sube un CSV con una fila por estudiante y estas columnas (mismos valores que el formulario). '
        'Puedes incluir una columna <code>ID</code> con un código interno; no incluyas nombres ni documentos.</p>',
        unsafe_allow_html=True,
    )
    plantilla = pd.DataFrame([{**{"ID": "EST-001"}, **meta["moda"]}])[["ID"] + VARIABLES]
    st.download_button("Descargar plantilla CSV", plantilla.to_csv(index=False).encode("utf-8-sig"),
                       "plantilla_alertas.csv", "text/csv")
    archivo = st.file_uploader("Archivo CSV", type=["csv"])
    if archivo is not None:
        # Acepta CSV con coma o punto y coma (Excel en español) y en UTF-8 o Windows-1252
        crudo = archivo.getvalue()
        df, error = None, None
        for codificacion in ("utf-8-sig", "cp1252"):
            try:
                df = pd.read_csv(io.BytesIO(crudo), dtype=str, encoding=codificacion, sep=None, engine="python")
                break
            except Exception as e:  # noqa: BLE001
                error = e
        if df is None:
            st.error(f"No se pudo leer el archivo: {error}")
            st.stop()
        df.columns = [c.strip() for c in df.columns]
        faltan = [c for c in VARIABLES if c not in df.columns]
        if faltan:
            st.error("Faltan columnas en el archivo: " + ", ".join(faltan))
        else:
            X = df[VARIABLES].fillna("").astype(str)
            desconocidos = {c: sorted(set(X[c]) - set(meta["categorias"][c])) for c in VARIABLES}
            desconocidos = {c: v for c, v in desconocidos.items() if v}
            if desconocidos:
                st.warning("Hay valores no reconocidos (se tratarán como neutros): "
                           + "; ".join(f"{etiqueta(c)}: {', '.join(v)}" for c, v in desconocidos.items()))
            prob = modelo.predict_proba(X)[:, 1]
            res = pd.DataFrame({
                "ID": df["ID"] if "ID" in df.columns else np.arange(1, len(df) + 1),
                "Probabilidad": np.round(prob * 100, 1),
                "Nivel": [NIVELES[nivel(p)][0] for p in prob],
            }).sort_values("Probabilidad", ascending=False)
            k1, k2, k3, k4 = st.columns(4)
            for col, lab, n in [(k1, "Estudiantes", len(res)),
                                (k2, "Riesgo alto", (res["Nivel"] == "Riesgo alto").sum()),
                                (k3, "Riesgo medio", (res["Nivel"] == "Riesgo medio").sum()),
                                (k4, "Riesgo bajo", (res["Nivel"] == "Riesgo bajo").sum())]:
                col.markdown(f'<div class="card kpi-card"><div class="kpi">{n}</div><div class="kpi-label">{lab}</div></div>',
                             unsafe_allow_html=True)
            st.write("")
            st.dataframe(
                res, hide_index=True, use_container_width=True,
                column_config={"Probabilidad": st.column_config.ProgressColumn(
                    "Probabilidad de deserción (%)", min_value=0, max_value=100, format="%.1f")},
            )
            buf = io.StringIO(); res.to_csv(buf, index=False)
            st.download_button("Descargar resultados", buf.getvalue().encode("utf-8-sig"),
                               "resultados_alertas.csv", "text/csv")

with tab_modelo:
    m = meta["metricas"]
    st.markdown(f"#### {meta['modelo']}")
    st.markdown(
        f'<p class="note">Entrenado con {meta["n_registros"]} registros históricos de Ingeniería de Sistemas y Computación '
        f'(tasa de deserción {meta["tasa_desercion"]*100:.1f}%). Métricas obtenidas con validación cruzada estratificada (5×5).</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    for c, (lab, v) in zip(cols, [("ROC AUC", f"{m['roc_auc']:.2f}"),
                                  ("Desertores detectados (alerta media o alta)", f"{m['recall_alerta_media_o_alta']*100:.0f}%"),
                                  ("Aciertos en alerta alta", f"{m['precision_alerta_alta']*100:.0f}%"),
                                  ("Brier (calibración, menor es mejor)", f"{m['brier']:.3f}")]):
        c.markdown(f'<div class="card kpi-card"><div class="kpi">{v}</div><div class="kpi-label">{lab}</div></div>',
                   unsafe_allow_html=True)
    st.write("")
    st.markdown("##### Variables excluidas a propósito")
    st.markdown(
        '<p class="note">No se usan orientación sexual, pertenencia étnica, víctima de violencia, discapacidad, IMC, '
        'hábitos de riesgo ni patologías en el hogar. Son datos sensibles (Ley 1581 de 2012) y generarían alertas '
        'discriminatorias. Además, "orientación: no responde" se comporta como un artefacto de la encuesta y no como causa.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("##### Limitaciones")
    st.markdown(
        '<p class="note">Prototipo académico. Las predicciones apoyan, pero no reemplazan, el criterio de docentes, tutores y '
        'Bienestar Universitario. El modelo debe reentrenarse cada semestre y vigilarse por sesgos entre grupos.</p>',
        unsafe_allow_html=True,
    )
