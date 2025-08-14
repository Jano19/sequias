# -*- coding: utf-8 -*-
"""
Sistema de Monitoreo de Sequías Agrícolas - Versión 5.1 (Mejoras integrales)
- Reemplazo completo con mejoras:
  • Reintentos + caché local para NASA POWER.
  • Outliers por cuantiles, imputación estacional mejorada.
  • Ingeniería de variables: lags 7/15/30, acumulados, anomalías z, SPI-30 aprox.
  • Calibración de probabilidades (Platt/Isotónica) y ajuste de umbral por objetivo (F1 o Recall mínimo).
  • SMOTE adaptativo (evita ValueError de k_neighbors en conjuntos pequeños).
  • Optuna opcional para búsqueda bayesiana (fallback a GridSearchCV).
  • PDF en A4 horizontal (landscape), tablas con encabezado repetido y métricas/tasas detalladas.
  • Interpretabilidad opcional: SHAP y Partial Dependence (si paquetes presentes).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
    matthews_corrcoef, brier_score_loss, balanced_accuracy_score,
    cohen_kappa_score, roc_curve, auc
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from scipy import stats
from scipy.stats import shapiro, normaltest, friedmanchisquare, levene, kruskal, zscore, gamma
from statsmodels.tsa.stattools import adfuller

from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter, Retry
import pickle
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# CACHÉ local en disco
import os, json, hashlib
from joblib import Memory
CACHE_DIR = ".cache_nasa"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(location=CACHE_DIR, verbose=0)

# ---------- ReportLab (PDF) ----------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm as RL_CM

# =====================
# CONFIG STREAMLIT
# =====================
st.set_page_config(
    page_title="Sistema de Monitoreo de Sequías Agrícolas",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================
# UTILIDADES (EDA)
# =====================
def _missing_report(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    miss = df.isna().sum()
    pct = (miss / total * 100).round(2)
    out = pd.DataFrame({
        'columna': miss.index,
        'faltantes': miss.values,
        '% faltantes': pct.values
    })
    return out.sort_values('% faltantes', ascending=False).reset_index(drop=True)

def _descriptive_stats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    num = df[cols].select_dtypes(include=[np.number]).copy()
    q1 = num.quantile(0.25)
    q3 = num.quantile(0.75)
    desc = pd.DataFrame({
        'count': num.count().round(0),
        'mean': num.mean().round(2),
        'std': num.std(ddof=1).round(2),
        'min': num.min().round(2),
        'q1': q1.round(2),
        'median': num.median().round(2),
        'q3': q3.round(2),
        'max': num.max().round(2),
    })
    desc['range'] = (desc['max'] - desc['min']).round(2)
    desc['IQR'] = (desc['q3'] - desc['q1']).round(2)
    with np.errstate(all='ignore'):
        desc['skew'] = num.skew().round(2)
        desc['kurtosis'] = num.kurtosis().round(2)
    return desc.reset_index().rename(columns={'index': 'variable'})

def _spearman_to_target(df, features, target='sequia'):
    rows = []
    for f in features:
        try:
            rho, p = stats.spearmanr(df[f], df[target])
        except Exception:
            rho, p = np.nan, np.nan
        rows.append({'feature': f, 'spearman_rho': rho, 'abs_rho': abs(rho), 'p_value': p})
    return pd.DataFrame(rows).sort_values('abs_rho', ascending=False)

def _prune_by_intercorr(df, features, target='sequia', max_intercorr=0.85):
    feats = features.copy()
    if len(feats) < 2:
        return feats
    cm = df[feats].corr().abs()
    while True:
        np.fill_diagonal(cm.values, 0.0)
        max_val = cm.values.max()
        if max_val <= max_intercorr or len(feats) <= 1:
            break
        i, j = np.where(cm.values == max_val)
        i, j = int(i[0]), int(j[0])
        f1, f2 = cm.index[i], cm.columns[j]
        rho1 = abs(stats.spearmanr(df[f1], df[target]).correlation)
        rho2 = abs(stats.spearmanr(df[f2], df[target]).correlation)
        drop = f1 if rho1 < rho2 else f2
        feats.remove(drop)
        cm = df[feats].corr().abs()
    return feats

def seleccionar_caracteristicas(df, candidate_features, target='sequia',
                                min_abs_corr=0.15, max_intercorr=0.85, top_k=None):
    corr_tbl = _spearman_to_target(df, candidate_features, target)
    pre = corr_tbl[corr_tbl['abs_rho'] >= min_abs_corr].copy()
    if pre.empty:
        pre = corr_tbl.head(6).copy()
    if top_k is not None and top_k > 0:
        pre = pre.head(top_k)
    prelim_feats = pre['feature'].tolist()
    pruned_feats = _prune_by_intercorr(df, prelim_feats, target=target, max_intercorr=max_intercorr)
    cm_selected = df[pruned_feats].corr() if len(pruned_feats) >= 2 else pd.DataFrame()
    return pruned_feats, corr_tbl, cm_selected

# =====================
# CONSTANTES + DESCARGA
# =====================
PARAMETROS_CLIMATICOS = {
    "precipitacion": "PRECTOTCORR",
    "temp_max": "T2M_MAX",
    "temp_min": "T2M_MIN",
    "humedad": "RH2M",
    "radiacion": "ALLSKY_SFC_SW_DWN",
    "presion": "PS",
    "viento": "WS2M",
    "evapotranspiracion": "EVPTRNS"
}

SENTINELAS_NASA = {
    "comunes": [-999, -999.0],
    "presion": [-1022.72, -1015.59],
}

REGIONES_ESTRATEGICAS = {
    "Valle de Jequetepeque (La Libertad)": {
        "coords": (-7.32, -79.57),
        "cultivos": ["arroz", "maíz"],
        "umbral_sequia": 1.8,
        "color": "#1f77b4"
    },
    "Valle de Chicama (La Libertad)": {
        "coords": (-7.84, -79.18),
        "cultivos": ["caña de azúcar"],
        "umbral_sequia": 2.1,
        "color": "#ff7f0e"
    },
    "Valle de Virú (La Libertad)": {
        "coords": (-8.41, -78.75),
        "cultivos": ["espárrago", "palta"],
        "umbral_sequia": 1.5,
        "color": "#2ca02c"
    },
    "Valle de Santa (Ancash)": {
        "coords": (-9.07, -78.57),
        "cultivos": ["maíz", "papa"],
        "umbral_sequia": 1.6,
        "color": "#d62728"
    }
}

# ---------- NUEVO: Features extendidas ----------
FEATURES_BASE = [
    'precipitacion', 'temp_max', 'temp_min', 'humedad',
    'balance_hidrico', 'evapotranspiracion', 'deficit_hidrico',
    'precipitacion_acum_30d', 'temp_promedio', 'sequia_30d', 'estacion',
    'precipitacion_diff', 'temp_max_diff', 'temp_min_diff'
]
FEATURES_EXT = [
    'precipitacion_acum_7d','precipitacion_acum_15d',
    'temp_promedio_7d','temp_promedio_15d','temp_promedio_30d',
    'anomalia_precipitacion_z','anomalia_temp_z',
    'spi30'
]
FEATURES_ALL = FEATURES_BASE + FEATURES_EXT

# ---------- Sesión HTTP con reintentos ----------
def _requests_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1.2, status_forcelist=[429, 500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

# ---------- Caché de respuesta NASA ----------
def _cache_key(lat, lon, start, end, params):
    k = json.dumps({'lat': lat, 'lon': lon, 'start': start, 'end': end, 'p': params}, sort_keys=True)
    return hashlib.md5(k.encode()).hexdigest()

def _save_cache(key, df):
    path = os.path.join(CACHE_DIR, f"{key}.parquet")
    try:
        df.to_parquet(path, index=True)
    except Exception:
        df.to_pickle(path)
    return path

def _load_cache(key):
    path = os.path.join(CACHE_DIR, f"{key}.parquet")
    if not os.path.exists(path):
        pkl = os.path.join(CACHE_DIR, f"{key}.parquet")
        if not os.path.exists(pkl):
            return None
    try:
        return pd.read_parquet(path)
    except Exception:
        try:
            return pd.read_pickle(path)
        except Exception:
            return None

@st.cache_data(ttl=3600, show_spinner="Obteniendo datos climáticos de la NASA POWER…")
def obtener_datos_nasa(lat, lon, fecha_ini, fecha_fin, umbral_sequia, interpolate_missing=True, use_cache=True):
    params = ",".join(PARAMETROS_CLIMATICOS.values())
    key = _cache_key(lat, lon, fecha_ini, fecha_fin, params)
    if use_cache:
        cached = _load_cache(key)
        if cached is not None and not cached.empty:
            return cached.copy()

    url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
           f"parameters={params}&latitude={lat}&longitude={lon}&"
           f"start={fecha_ini}&end={fecha_fin}&community=AG")
    try:
        s = _requests_session()
        r = s.get(url, timeout=120)
        r.raise_for_status()
        data = r.json()
        if not data.get("properties", {}).get("parameter"):
            return None

        frames = []
        for nombre, codigo in PARAMETROS_CLIMATICOS.items():
            if codigo in data['properties']['parameter']:
                serie = pd.DataFrame.from_dict(
                    data['properties']['parameter'][codigo],
                    orient='index', columns=[nombre]
                )
                serie.index = pd.to_datetime(serie.index)
                frames.append(serie)

        df = pd.concat(frames, axis=1).sort_index()
        df = df.loc[~df.index.duplicated(keep="first")]

        # Limpieza sentinelas
        cols_comunes = [c for c in ['precipitacion','radiacion','temp_max','temp_min',
                                    'viento','humedad','evapotranspiracion','presion']
                        if c in df.columns]
        for c in cols_comunes:
            df[c] = df[c].replace(SENTINELAS_NASA["comunes"], np.nan)
        if 'presion' in df.columns:
            df['presion'] = df['presion'].replace(SENTINELAS_NASA["presion"], np.nan)

        # Imputación temporal priorizando estacionalidad
        if interpolate_missing and len(df) > 1:
            df[cols_comunes] = df[cols_comunes].ffill().bfill()
            df[cols_comunes] = df[cols_comunes].interpolate(method='time', limit_direction='both')

        # Saneos básicos
        if 'precipitacion' in df.columns:
            df['precipitacion'] = df['precipitacion'].clip(lower=0).round(2)
        if 'humedad' in df.columns:
            df['humedad'] = df['humedad'].clip(0, 100)

        # Derivados base
        if {'precipitacion','temp_max','temp_min'}.issubset(df.columns):
            df['balance_hidrico'] = (
                df['precipitacion'].rolling(30, min_periods=1).mean()
                - (df['temp_max'] - df['temp_min']).rolling(30, min_periods=1).mean() * 0.5
            )

        if 'evapotranspiracion' not in df.columns or df['evapotranspiracion'].isna().all():
            if {'temp_max','temp_min'}.issubset(df.columns):
                df['evapotranspiracion'] = (
                    0.0023 * ((df['temp_max'] + df['temp_min'])/2 + 17.8) *
                    (df['temp_max'] - df['temp_min'])**0.5
                )

        if {'evapotranspiracion','precipitacion'}.issubset(df.columns):
            df['deficit_hidrico'] = df['evapotranspiracion'].shift(1) - df['precipitacion'].shift(1)

        df['sequia_hoy'] = (df['precipitacion'] < umbral_sequia).astype(int) if 'precipitacion' in df.columns else np.nan
        df['sequia'] = df['sequia_hoy'].shift(-1)  # target t+1
        df['umbral_sequia'] = umbral_sequia
        df['mes'] = df.index.month
        df['estacion'] = df['mes'].apply(lambda m: 1 if m in [12,1,2] else (2 if m in [3,4,5] else (3 if m in [6,7,8] else 4)))
        if 'precipitacion' in df.columns:
            df['precipitacion_acum_30d'] = df['precipitacion'].rolling(30, min_periods=1).sum()
            df['precipitacion_acum_7d'] = df['precipitacion'].rolling(7, min_periods=1).sum()
            df['precipitacion_acum_15d'] = df['precipitacion'].rolling(15, min_periods=1).sum()
        if {'temp_max','temp_min'}.issubset(df.columns):
            df['temp_promedio'] = (df['temp_max'] + df['temp_min']) / 2
            df['temp_promedio_7d']  = df['temp_promedio'].rolling(7, min_periods=1).mean()
            df['temp_promedio_15d'] = df['temp_promedio'].rolling(15, min_periods=1).mean()
            df['temp_promedio_30d'] = df['temp_promedio'].rolling(30, min_periods=1).mean()

        for col in ['precipitacion', 'temp_max', 'temp_min']:
            if col in df.columns:
                try:
                    p = adfuller(df[col].dropna())[1]
                except Exception:
                    p = 1.0
                df[f'{col}_diff'] = df[col].diff() if p > 0.05 else df[col]

        if 'sequia' in df.columns:
            df['sequia_30d'] = df['sequia'].rolling(30, min_periods=1).mean()

        # NUEVO: anomalías z estacionales (por mes)
        if 'precipitacion' in df.columns:
            df['anomalia_precipitacion_z'] = df.groupby('mes')['precipitacion'].transform(lambda s: zscore(s, nan_policy='omit'))
        if 'temp_promedio' in df.columns:
            df['anomalia_temp_z'] = df.groupby('mes')['temp_promedio'].transform(lambda s: zscore(s, nan_policy='omit'))

        # NUEVO: SPI 30 días (aprox gamma/z)
        if 'precipitacion' in df.columns:
            roll = df['precipitacion'].rolling(30, min_periods=1).sum().replace(0, 1e-6)
            try:
                spi = pd.Series(index=df.index, dtype=float)
                for m in range(1,13):
                    idx = df['mes']==m
                    x = roll[idx].dropna()
                    if len(x) > 20:
                        if x.std() < 1e-6:
                            spi.loc[idx] = 0.0
                            continue
                        a, loc, scale = gamma.fit(x, floc=0)
                        cdf = gamma.cdf(roll[idx], a=a, loc=loc, scale=scale).clip(1e-6,1-1e-6)
                        spi.loc[idx] = stats.norm.ppf(cdf)
                    else:
                        spi.loc[idx] = zscore(roll[idx], nan_policy='omit')
                df['spi30'] = spi
            except Exception:
                df['spi30'] = zscore(roll, nan_policy='omit')

        # Quitar SOLO filas sin etiqueta
        if 'sequia' in df.columns and df['sequia'].isna().any():
            df = df.loc[~df['sequia'].isna()].copy()

        if use_cache and (df is not None) and not df.empty:
            _save_cache(key, df)

        return df

    except Exception as e:
        st.error(f"Error descargando NASA POWER: {e}")
        return None

# =====================
# PRE/POST-PROCESO ADICIONAL
# =====================
def tratar_outliers_por_cuantiles(df: pd.DataFrame, cols: list, low_q=0.001, high_q=0.999):
    dfc = df.copy()
    for c in cols:
        if c in dfc.columns and pd.api.types.is_numeric_dtype(dfc[c]):
            lo, hi = dfc[c].quantile([low_q, high_q])
            dfc[c] = dfc[c].clip(lo, hi)
    return dfc

def crear_pipeline_preprocesamiento(features_num):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('selector', SelectKBest(f_classif, k=min(12, len(features_num))))
    ])
    preprocessor = ColumnTransformer(
        transformers=[('num', numeric_transformer, features_num)],
        verbose_feature_names_out=False
    )
    return preprocessor

def build_model_config():
    return {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced']
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42, class_weight='balanced'),
            'params': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=42, verbose=-1),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 63],
                'max_depth': [3, 5, -1],
                'min_child_samples': [20, 50],
                'feature_fraction': [0.8, 1.0]
            }
        },
        'Híbrido (Stacking RF+XGB+SVM)': {
            'model': StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
                    ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5,
                                          subsample=0.9, colsample_bytree=0.9,
                                          random_state=42, eval_metric='logloss', use_label_encoder=False)),
                    ('svm', SVC(C=1.0, kernel='rbf', gamma='scale', probability=True,
                                class_weight='balanced', random_state=42))
                ],
                final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
                stack_method='predict_proba',
                n_jobs=-1,
                passthrough=False
            ),
            'params': {}
        }
    }

MODEL_CONFIG = build_model_config()

# =====================
# SMOTE ADAPTATIVO (nuevo)
# =====================
def _smote_adaptativo(y, random_state=42, base_k=5):
    """
    Devuelve una instancia de SMOTE con k_neighbors ajustado dinámicamente
    según el tamaño de la clase minoritaria. Si no es posible, devuelve None.
    """
    try:
        from collections import Counter
        cnt = Counter(pd.Series(y).astype(int))
        min_count = min(cnt.values())
        if min_count <= 1:
            return None
        k_safe = min(base_k, min_count - 1)
        if k_safe < 1:
            return None
        return SMOTE(random_state=random_state, k_neighbors=k_safe)
    except Exception:
        return None



# === Helpers robustos para SMOTE (limpieza y fallback) ===
def _mask_limpio_xy(X, y):
    Xnum = X.replace([np.inf, -np.inf], np.nan)
    m = Xnum.notna().all(axis=1) & pd.Series(y).notna()
    return Xnum.loc[m], pd.Series(y).loc[m]

def _safe_fit_resample(sampler, X, y):
    """Intenta SMOTE; si falla por validación, castea a numpy y reintenta."""
    try:
        return sampler.fit_resample(X, y)
    except AttributeError as e:
        # Algunos entornos rompen _validate_data; probamos con numpy
        X_np = np.asarray(X, dtype=float)
        y_np = np.asarray(y).astype(int)
        return sampler.fit_resample(X_np, y_np)
    except ValueError as e:
        st.warning(f"No se pudo aplicar SMOTE: {e}")
        return X, y
# =====================
# ENTRENAMIENTO / EVAL
# =====================
def _has_both_classes(y):
    v = set(pd.Series(y).astype(int).unique())
    return (0 in v) and (1 in v)

def _calibrar_modelo_si_aplica(best_estimator, X_train, y_train, method='isotonic'):
    try:
        cal = CalibratedClassifierCV(best_estimator, method=method, cv=3)
        cal.fit(X_train, y_train)
        return cal
    except Exception as e:
        st.info(f"No se pudo calibrar (se continúa sin calibración): {e}")
        return best_estimator

def _umbral_optimo(y_true, y_proba, modo='max_f1', recall_min=0.85):
    P, R, T = precision_recall_curve(y_true, y_proba)
    if modo == 'max_f1':
        f1_vals = (2*P*R/(P+R+1e-12))
        i = int(np.nanargmax(f1_vals))
        thr = 0.5 if i >= len(T) else T[i]
        return float(thr)
    else:
        pairs = [(t, r) for t, r in zip(np.append(T,1.0), R)]
        candidates = [t for t, r in pairs if r >= recall_min]
        if not candidates:
            return 0.5
        return float(max(candidates))

def _optuna_search(pipe, param_grid, X_train, y_train, scoring='f1', n_trials=30):
    try:
        import optuna
        from sklearn.model_selection import cross_val_score
        tscv = TimeSeriesSplit(n_splits=5, gap=30)

        def objective(trial):
            params = {}
            for k, v in param_grid.items():
                k2 = k.replace('model__','')
                if isinstance(v, list) and all(isinstance(e,(int,float)) for e in v):
                    lo, hi = min(v), max(v)
                    if isinstance(v[0], int) and isinstance(v[-1], int):
                        params[k] = trial.suggest_int(k2, int(lo), int(hi))
                    else:
                        params[k] = trial.suggest_float(k2, float(lo), float(hi), log=False)
                elif isinstance(v, list):
                    params[k] = trial.suggest_categorical(k2, v)
            pipe.set_params(**{f'model__{k.replace("model__","")}': val for k, val in params.items()})
            scores = cross_val_score(pipe, X_train, y_train, cv=tscv, scoring=scoring, n_jobs=-1)
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params, study.best_value
    except Exception as e:
        st.info(f"Optuna no disponible / falló: {e}")
        return None, None

def optimizar_modelo(X_train, y_train, model_name, features_num, scoring='f1', use_optuna=False, n_trials=30):
    cfg = MODEL_CONFIG[model_name]
    pipe = Pipeline([
        ('preprocessor', crear_pipeline_preprocesamiento(features_num)),
        ('model', cfg['model'])
    ])
    params = {f'model__{k}': v for k, v in cfg['params'].items()} if cfg['params'] else None
    tscv = TimeSeriesSplit(n_splits=5, gap=30)

    import time
    t0 = time.time()
    best_params, best_score, best_est = {}, np.nan, pipe

    if params:
        if use_optuna:
            bp, bs = _optuna_search(pipe, params, X_train, y_train, scoring=scoring, n_trials=n_trials)
            if bp:
                final_params = {f'model__{k}': v for k, v in bp.items()}
                pipe.set_params(**final_params)
                pipe.fit(X_train, y_train)
                best_est = pipe
                best_params = final_params
                best_score = bs if bs is not None else np.nan
            else:
                gs = GridSearchCV(pipe, params, cv=tscv, scoring=scoring, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                best_est = gs.best_estimator_
                best_params = gs.best_params_
                best_score = gs.best_score_
        else:
            gs = GridSearchCV(pipe, params, cv=tscv, scoring=scoring, n_jobs=-1, verbose=0)
            gs.fit(X_train, y_train)
            best_est = gs.best_estimator_
            best_params = gs.best_params_
            best_score = gs.best_score_
    else:
        pipe.fit(X_train, y_train)
        best_est = pipe
        scores = []
        for tr, val in tscv.split(X_train):
            best_est.fit(X_train.iloc[tr], y_train.iloc[tr])
            ypv = best_est.predict(X_train.iloc[val])
            scores.append(f1_score(y_train.iloc[val], ypv, zero_division=0))
        best_score = float(np.mean(scores)) if scores else np.nan

    elapsed = time.time() - t0
    return {
        'mejor_modelo': best_est,
        'mejores_parametros': best_params,
        'mejor_puntaje_cv': best_score,
        'tiempo_entrenamiento_seg': round(elapsed, 2)
    }

def optimizar_modelos(X_train, y_train, modelos_seleccionados, features_num, scoring='f1', use_optuna=False, n_trials=30):
    resultados, best_models = {}, {}
    progress = st.progress(0.0)
    step = 1.0 / max(len(modelos_seleccionados), 1)
    for i, name in enumerate(modelos_seleccionados):
        st.info(f"Optimizando {name}…")
        try:
            r = optimizar_modelo(X_train, y_train, name, features_num, scoring=scoring, use_optuna=use_optuna, n_trials=n_trials)
            resultados[name] = r
            best_models[name] = r['mejor_modelo']
        except Exception as e:
            st.error(f"Error optimizando {name}: {e}")
        progress.progress(min((i+1)*step, 1.0))
    return resultados, best_models

def _pred_with_threshold(model, X, thr=0.5):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X)
        proba = (s - s.min()) / (s.max() - s.min() + 1e-12)
    else:
        y_hat = model.predict(X).astype(int)
        return y_hat, y_hat.astype(float)
    y_hat = (proba >= thr).astype(int)
    return y_hat, proba

# === Estadísticas extendidas de matriz de confusión ===
def _confusion_stats(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
    total = max(TN+FP+FN+TP, 1)
    acc = (TN+TP)/total
    prec = TP/max(TP+FP, 1e-12)
    rec = TP/max(TP+FN, 1e-12)
    spec = TN/max(TN+FP, 1e-12)
    f1 = (2*prec*rec)/max((prec+rec), 1e-12)
    prev = (TP+FN)/total
    fpr = FP/max(FP+TN, 1e-12)
    fnr = FN/max(FN+TP, 1e-12)
    tnr = spec
    tpr = rec
    return {
        'cm': cm, 'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 'total': total,
        'accuracy': acc, 'precision': prec, 'recall': rec, 'specificity': spec,
        'f1': f1, 'prevalence': prev, 'FPR': fpr, 'FNR': fnr, 'TNR': tnr, 'TPR': tpr
    }

def evaluar_modelos(resultados, best_models, X_test, y_test,
                    thr_strategy=None, recall_min=0.85, calibrate=False, cal_method='isotonic',
                    X_cal=None, y_cal=None):
    rows = []
    details = {}

    def safe_auc(y_true, y_score):
        if not _has_both_classes(y_true):
            return np.nan
        try:
            return roc_auc_score(y_true, y_score)
        except Exception:
            return np.nan

    def safe_pr_auc(y_true, y_score):
        if not _has_both_classes(y_true):
            return np.nan
        try:
            P, R, _ = precision_recall_curve(y_true, y_score)
            return auc(R, P)
        except Exception:
            return np.nan

    for name, model in best_models.items():
        try:
            cur_model = model
            if calibrate and (X_cal is not None) and _has_both_classes(y_cal):
                cur_model = _calibrar_modelo_si_aplica(model, X_cal, y_cal, method=cal_method)

            y_proba_full = cur_model.predict_proba(X_test)[:,1] if hasattr(cur_model,'predict_proba') else None
            thr = 0.5
            if (thr_strategy is not None) and (y_proba_full is not None) and _has_both_classes(y_test):
                if thr_strategy == 'max_f1':
                    thr = _umbral_optimo(y_test, y_proba_full, modo='max_f1')
                elif thr_strategy == 'recall_min':
                    thr = _umbral_optimo(y_test, y_proba_full, modo='recall_min', recall_min=recall_min)

            y_pred, y_proba = _pred_with_threshold(cur_model, X_test, thr=thr)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)
            spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
            mcc  = matthews_corrcoef(y_test, y_pred) if _has_both_classes(y_test) else np.nan
            bacc = balanced_accuracy_score(y_test, y_pred) if _has_both_classes(y_test) else np.nan
            brier = brier_score_loss(y_test, y_proba) if _has_both_classes(y_test) else np.nan
            rocauc = safe_auc(y_test, y_proba)
            prauc  = safe_pr_auc(y_test, y_proba)
            kappa  = cohen_kappa_score(y_test, y_pred)

            rows.append({
                'Modelo': name,
                'Threshold': float(thr),
                'Calibrado': bool(calibrate),
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1-Score': f1,
                'ROC AUC': rocauc,
                'PR AUC': prauc,
                'Specificity': spec,
                'MCC': mcc,
                'Brier Score': brier,
                'Balanced Accuracy': bacc,
                'Kappa': kappa,
                'Mejores Parámetros': str(resultados[name]['mejores_parametros']),
                'Tiempo (s)': resultados[name]['tiempo_entrenamiento_seg'],
                'Score CV (F1)': resultados[name]['mejor_puntaje_cv']
            })

            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            resultados[name]['matriz_confusion'] = cm
            resultados[name]['y_pred'] = y_pred
            resultados[name]['y_proba'] = y_proba
            resultados[name]['residuos_prob'] = (y_test.values.astype(float) - y_proba)
            resultados[name]['threshold'] = float(thr)
            resultados[name]['calibrado'] = bool(calibrate)
            resultados[name]['confusion_stats'] = _confusion_stats(y_test, y_pred)
            details[name] = {'thr': float(thr), 'calibrado': bool(calibrate)}

        except Exception as e:
            st.error(f"Error evaluando {name}: {e}")

    return pd.DataFrame(rows), details

# =====================
# PRUEBAS ESTADÍSTICAS (MODELOS)
# =====================
def test_friedman(df_metricas, metricas=('F1-Score', 'PR AUC', 'Kappa')):
    out = {}
    if df_metricas is None or df_metricas.empty:
        return out
    for m in metricas:
        vals, names = [], []
        for n in df_metricas['Modelo'].unique():
            v = df_metricas.loc[df_metricas['Modelo'] == n, m].values
            if len(v) > 0 and not np.isnan(v[0]):
                vals.append(v)
                names.append(n)
        if len(vals) < 2:
            out[m] = {'Estadístico': np.nan, 'p-valor': np.nan, 'Mensaje': 'Se necesitan ≥2 modelos válidos'}
            continue
        if np.std(np.concatenate(vals)) < 1e-8:
            out[m] = {'Estadístico': np.nan, 'p-valor': np.nan, 'Mensaje': 'Sin variabilidad suficiente'}
            continue
        try:
            stat, p = friedmanchisquare(*vals)
            out[m] = {'Estadístico': stat, 'p-valor': p, 'Modelos': names}
        except Exception as e:
            out[m] = {'Estadístico': np.nan, 'p-valor': np.nan, 'Error': str(e)}
    return out

def posthoc_nemenyi(df_metricas, metrica='F1-Score'):
    try:
        import scikit_posthocs as sp
        wide = df_metricas.pivot_table(index='Modelo', values=metrica, aggfunc='mean')
        W = pd.concat([wide, wide.add_suffix('_rep')], axis=1)
        ph = sp.posthoc_nemenyi_friedman(W.T)
        return ph, None
    except Exception as e:
        return None, f"Post‑hoc no disponible (instala scikit-posthocs) o no aplica: {e}"

# =====================
# PRUEBAS ESTADÍSTICAS DESCRIPTIVAS
# =====================
def _compute_descriptive_tests(df: pd.DataFrame, num_cols: list):
    df_levene_rows, df_kw_rows, dunn_results = [], [], {}
    grp_flag = df['sequia_hoy'].astype(int)
    grp_est = df['estacion'].astype(int)

    for c in num_cols:
        try:
            g0 = df.loc[grp_flag == 0, c].dropna()
            g1 = df.loc[grp_flag == 1, c].dropna()
            if len(g0) > 2 and len(g1) > 2:
                W, p = levene(g0, g1, center='median')
                df_levene_rows.append({'variable': c, 'W': W, 'p_levene': p})
        except Exception:
            pass

        try:
            groups = [df.loc[grp_est == k, c].dropna() for k in sorted(grp_est.unique())]
            groups = [g for g in groups if len(g) > 2]
            if len(groups) >= 2:
                H, p = kruskal(*groups)
                df_kw_rows.append({'variable': c, 'H': H, 'p_kw': p})
                if p < 0.05:
                    try:
                        import scikit_posthocs as sp
                        vals = df[c].values
                        labs = grp_est.values
                        mask = ~pd.isna(vals) & ~pd.isna(labs)
                        ph = sp.posthoc_dunn(vals[mask], labs[mask], p_adjust='bonferroni')
                        ph.index = [f"Est.{int(i)}" for i in ph.index]
                        ph.columns = [f"Est.{int(i)}" for i in ph.columns]
                        dunn_results[c] = ph.round(4)
                    except Exception:
                        dunn_results[c] = None
            else:
                dunn_results[c] = None
        except Exception:
            dunn_results[c] = None

    df_levene = pd.DataFrame(df_levene_rows).sort_values('p_levene', ascending=True).reset_index(drop=True)
    df_kw = pd.DataFrame(df_kw_rows).sort_values('p_kw', ascending=True).reset_index(drop=True)
    return df_levene, df_kw, dunn_results

# =====================
# VIZ BÁSICAS
# =====================
def viz_missing_and_descriptives(df, num_cols):
    with st.expander("📋 Reporte de datos faltantes", expanded=True):
        miss = _missing_report(df)
        st.dataframe(miss, use_container_width=True)

    with st.expander("📊 Estadísticos descriptivos", expanded=True):
        desc = _descriptive_stats(df, num_cols)
        st.dataframe(desc, use_container_width=True)

    with st.expander("📉 Histogramas", expanded=False):
        for c in num_cols:
            st.plotly_chart(px.histogram(df, x=c, nbins=50, title=f'Histograma: {c}'), use_container_width=True)

    with st.expander("📦 Diagramas de Caja (Boxplots)", expanded=False):
        for c in num_cols:
            st.plotly_chart(px.box(df, y=c, points='outliers', title=f'Boxplot: {c}'), use_container_width=True)

def viz_correlations(df, features, target='sequia'):
    with st.expander("🧩 Selección por correlación con el target", expanded=True):
        selected, corr_tbl, cm_sel = seleccionar_caracteristicas(
            df, features, target=target, min_abs_corr=0.15, max_intercorr=0.85, top_k=None
        )
        st.markdown("**Ranking Spearman |ρ| vs. target**")
        st.dataframe(corr_tbl[['feature', 'spearman_rho', 'abs_rho', 'p_value']].round(4), use_container_width=True)

    if len(features) >= 2:
        with st.expander("🔥 Mapa de calor (TODAS las features candidatas)", expanded=True):
            fig_all = px.imshow(df[features].corr().round(2),
                                text_auto=True, title="Correlaciones entre features")
            st.plotly_chart(fig_all, use_container_width=True)

    if not cm_sel.empty:
        with st.expander("🔥 Mapa de calor (subset seleccionado tras pruning)", expanded=True):
            fig_sel = px.imshow(cm_sel.round(2), text_auto=True, title="Correlaciones (subset seleccionado)")
            st.plotly_chart(fig_sel, use_container_width=True)

    st.success(f"✅ Features seleccionadas ({len(selected)}): {', '.join(selected)}")
    return selected

def viz_model_comparison(df_metricas):
    with st.expander("📊 Comparación Visual de Modelos", expanded=True):
        ycols = ['Precision', 'Recall', 'F1-Score', 'PR AUC', 'Kappa']
        fig = px.bar(df_metricas, x='Modelo', y=ycols, barmode='group',
                     title='Métricas clave por modelo', labels={'value': 'Puntaje', 'variable': 'Métrica'})
        st.plotly_chart(fig, use_container_width=True)

def viz_pred_vs_real(resultados, y_test):
    with st.expander("🎯 Predicción vs Real (probabilidad de sequía)", expanded=False):
        for name, res in resultados.items():
            y_proba = res.get('y_proba', None)
            if y_proba is None:
                continue
            dfp = pd.DataFrame({'Real': y_test.values.astype(int), 'Probabilidad': y_proba})
            fig = px.scatter(dfp, x=dfp.index, y='Probabilidad',
                             color=dfp['Real'].map({0: 'No sequía', 1: 'Sequía'}),
                             title=f'{name}: Probabilidad estimada vs etiqueta real',
                             labels={'x': 'Índice', 'y': 'Prob.'})
            st.plotly_chart(fig, use_container_width=True)

def viz_residual_tests(resultados):
    with st.expander("🧪 Pruebas de Normalidad sobre Residuos Probabilísticos (y − p)", expanded=False):
        for name, res in resultados.items():
            resid = res.get('residuos_prob', None)
            if resid is None:
                continue
            r = pd.Series(resid)
            if len(r) <= 5000:
                stat, p = shapiro(r)
                testname = "Shapiro-Wilk"
            else:
                stat, p = normaltest(r)
                testname = "D'Agostino-Pearson"
            st.markdown(f"**{name}** — {testname}: estadístico={stat:.3f}, p={p:.4f}")
        st.caption("p<0.05 sugiere desviación de normalidad (esperable en residuos de clasificación).")

# =====================
# PDF HELPERS
# =====================
def _fmt(x, n=2):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "N/A"
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)

def _mk_table(data, col_widths=None, repeat_rows=1, font_size=8.8):
    t = Table(data, colWidths=col_widths, repeatRows=repeat_rows)
    t.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F3F4F6")),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), font_size),
        ('LEFTPADDING', (0,0), (-1,-1), 2),
        ('RIGHTPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
    ]))
    return t

def _plot_confusion_image(cm):
    fig, ax = plt.subplots(figsize=(2.8, 2.8), dpi=150)
    im = ax.imshow(cm, cmap='Blues')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha='center', va='center', color='black', fontsize=9)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['No sequía', 'Sequía']); ax.set_yticklabels(['No sequía', 'Sequía'])
    ax.set_xlabel('Predicho'); ax.set_ylabel('Real')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return RLImage(buf, width=7*RL_CM, height=7*RL_CM)

def _plot_importance_image(df_imp):
    fig = px.bar(df_imp.head(12), x='feature', y='importance', title='Importancia de características')
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=260)
    buf = BytesIO()
    try:
        fig.write_image(buf, format='png', scale=2)
    except Exception:
        # fallback si kaleido no está
        import matplotlib.pyplot as _plt
        _plt.figure(figsize=(6,3))
        _plt.bar(df_imp.head(12)['feature'], df_imp.head(12)['importance'])
        _plt.xticks(rotation=45, ha='right')
        _plt.tight_layout()
        _plt.savefig(buf, format='png', bbox_inches='tight')
        _plt.close()
    buf.seek(0)
    return RLImage(buf, width=16*RL_CM, height=5.5*RL_CM)

def _interpret_confusion(cmat):
    try:
        TN, FP = int(cmat[0, 0]), int(cmat[0, 1])
        FN, TP = int(cmat[1, 0]), int(cmat[1, 1])
    except Exception:
        return ["No se pudo interpretar la matriz de confusión (formato inesperado)."]

    total = TN + FP + FN + TP if (TN+FP+FN+TP) > 0 else 1
    accuracy = (TN + TP) / total
    precision = TP / (TP + FP) if (TP + FP) > 0 else float('nan')
    recall = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
    specificity = TN / (TN + FP) if (TN + FP) > 0 else float('nan')
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else float('nan')
    prevalence = (TP + FN) / total

    lines = []
    lines.append(
        f"Se evaluaron {total:,} observaciones: {TN:,} TN, {FP:,} FP, {FN:,} FN, {TP:,} TP."
    )
    lines.append(
        f"Accuracy: {accuracy:.3f} · Precision: {precision:.3f} · Recall: {recall:.3f} · "
        f"Especificidad: {specificity:.3f} · F1: {f1:.3f} · Prevalencia: {prevalence:.3f}."
    )
    if FN > FP:
        lines.append("Predominan FN (sequías reales no detectadas) → bajar umbral y/o usar class_weight/SMOTE.")
    elif FP > FN:
        lines.append("Predominan FP (falsas alarmas) → subir umbral o calibrar probabilidades.")
    else:
        lines.append("FP y FN relativamente balanceados → el umbral parece razonable, ajustar con curvas PR/ROC.")

    if not np.isnan(recall) and not np.isnan(specificity):
        gap = recall - specificity
        if gap < -0.15:
            lines.append("Especificidad >> Sensibilidad → se evitan falsas alarmas, pero se pierden sequías. Prioriza Recall.")
        elif gap > 0.15:
            lines.append("Sensibilidad >> Especificidad → detecta muchas sequías pero con más falsas alarmas. Ajusta umbral.")

    return lines

def _calc_feature_importances_for_pdf(best_model, X_test, y_test, features):
    try:
        r = permutation_importance(best_model, X_test, y_test, scoring="f1",
                                   n_repeats=10, random_state=42, n_jobs=-1)
        mask = best_model.named_steps['preprocessor']\
                         .named_transformers_['num']\
                         .named_steps['selector'].get_support()
        base_names = features if len(features) >= sum(mask) else X_test.columns.tolist()
        selected = [n for n, keep in zip(base_names, mask) if keep]
        imp_df = pd.DataFrame({'feature': selected, 'importance': r.importances_mean})\
                   .sort_values('importance', ascending=False)
        return imp_df
    except Exception:
        inner = getattr(best_model, 'named_steps', {})
        mdl = inner.get('model', None) if isinstance(inner, dict) else None
        fi = getattr(mdl, 'feature_importances_', None) if mdl is not None else None
        if fi is not None:
            mask = best_model.named_steps['preprocessor']\
                             .named_transformers_['num']\
                             .named_steps['selector'].get_support()
            base_names = X_test.columns.tolist()
            selected = [n for n, keep in zip(base_names, mask) if keep]
            return pd.DataFrame({'feature': selected, 'importance': fi})\
                     .sort_values('importance', ascending=False)
        return None

def _for_report_sanitize(df_rep: pd.DataFrame) -> pd.DataFrame:
    dfc = df_rep.copy()
    sentinelas = [-999, -999.0, -1022.72, -1015.59]
    dfc.replace(sentinelas, np.nan, inplace=True)
    return dfc

def generar_reporte_pdf(
    df, region, cultivo, fecha_ini, fecha_fin,
    df_metricas=None, resultados=None, best_models=None,
    features=None, X_test=None, y_test=None,
    df_levene=None, df_kw=None, dunn_results=None,
    thr_strategy_text=None, recall_min_req=None
):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", fontSize=13, leading=15, spaceAfter=10, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="H2", fontSize=11, leading=13, spaceAfter=8, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="Body", fontSize=9.2, leading=11))

    pdf_buf = BytesIO()
    doc = SimpleDocTemplate(
        pdf_buf,
        pagesize=landscape(A4),  # HORIZONTAL
        leftMargin=1.4*RL_CM, rightMargin=1.4*RL_CM, topMargin=1.2*RL_CM, bottomMargin=1.2*RL_CM
    )
    story = []

    story += [Paragraph("Reporte Ejecutivo – Sistema de Monitoreo de Sequías Agrícolas", styles['H1'])]
    story += [Paragraph(f"Región: {region} | Cultivo: {cultivo}", styles['Body'])]
    story += [Paragraph(f"Rango de análisis: {fecha_ini} a {fecha_fin}", styles['Body'])]
    if thr_strategy_text:
        story += [Paragraph(f"Estrategia de umbral: {thr_strategy_text}" + \
                            (f" (Recall mínimo exigido = {recall_min_req:.2f})" if recall_min_req else ""),
                            styles['Body'])]
    story += [Spacer(1, 8)]

    story += [Paragraph("Resumen ejecutivo", styles['H2'])]
    story += [Paragraph(
        "Este informe resume el estado del dataset, los resultados de los modelos de Machine Learning y hallazgos estadísticos clave. "
        "Se enfatiza la detección de sequías (minimizar falsos negativos) y la interpretabilidad de resultados.", styles['Body']
    )]
    story += [Spacer(1, 6)]

    story += [Paragraph("Descripción del dataset", styles['H2'])]
    story += [Paragraph(f"Número de observaciones: {df.shape[0]:,}. Variables: {len(df.columns)}.", styles['Body'])]
    pct_sequia = 100 * df['sequia_hoy'].mean()
    story += [Paragraph(f"Proporción de días con sequía (hoy): {pct_sequia:.1f}%.", styles['Body'])]
    story += [Spacer(1, 6)]

    story += [Paragraph("Análisis estadístico descriptivo", styles['H2'])]
    df_desc = _descriptive_stats(_for_report_sanitize(df), [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
    data_desc = [["Variable","count","mean","std","min","q1","median","q3","max","range","IQR","skew","kurtosis"]]
    for _, row in df_desc.iterrows():
        data_desc.append([row['variable']] + [str(row[k]) for k in ["count","mean","std","min","q1","median","q3","max","range","IQR","skew","kurtosis"]])
    story += [_mk_table(data_desc, col_widths=[5.0*RL_CM]+[2.1*RL_CM]*12, repeat_rows=1, font_size=8.3)]
    story += [Spacer(1, 6)]

    if df_metricas is not None and not df_metricas.empty:
        story += [Paragraph("Resultados de modelos de Machine Learning", styles['H2'])]
        heads = ["Modelo","Thr","Cal.","F1","Precision","Recall","Kappa","PR AUC","ROC AUC","Brier","BA","Tiempo(s)","CV(F1)"]
        data_m = [heads]
        for _, r in df_metricas.sort_values('F1-Score', ascending=False).iterrows():
            data_m.append([
                r['Modelo'], _fmt(r.get('Threshold',0.5),3), "Sí" if r.get('Calibrado',False) else "No",
                _fmt(r['F1-Score'],4), _fmt(r['Precision'],4), _fmt(r['Recall'],4),
                _fmt(r['Kappa'],4), _fmt(r['PR AUC'],4), _fmt(r['ROC AUC'],4),
                _fmt(r['Brier Score'],4), _fmt(r['Balanced Accuracy'],4), _fmt(r['Tiempo (s)'],2),
                _fmt(r.get('Score CV (F1)', np.nan),4)
            ])
        story += [_mk_table(data_m,
                            col_widths=[5.2*RL_CM,1.6*RL_CM,1.6*RL_CM,1.7*RL_CM,1.9*RL_CM,1.9*RL_CM,1.7*RL_CM,
                                        1.9*RL_CM,1.9*RL_CM,1.9*RL_CM,1.9*RL_CM,2.0*RL_CM,2.0*RL_CM],
                            repeat_rows=1, font_size=8.2)]
        story += [Spacer(1, 8)]

    if df_metricas is not None and not df_metricas.empty and resultados is not None:
        best_name = df_metricas.loc[df_metricas['F1-Score'].idxmax(), 'Modelo']
        story += [Paragraph(f"Análisis del mejor modelo: {best_name}", styles['H2'])]

        mrow = df_metricas[df_metricas['Modelo']==best_name].iloc[0]
        detalle = (
            f"<b>Thr</b>: {_fmt(mrow.get('Threshold',0.5),3)} | <b>Calibrado</b>: {'Sí' if mrow.get('Calibrado',False) else 'No'} | "
            f"<b>F1</b>: {_fmt(mrow['F1-Score'],4)} | <b>Precisión</b>: {_fmt(mrow['Precision'],4)} | "
            f"<b>Recall</b>: {_fmt(mrow['Recall'],4)} | <b>Kappa</b>: {_fmt(mrow['Kappa'],4)} | "
            f"<b>PR AUC</b>: {_fmt(mrow['PR AUC'],4)} | <b>ROC AUC</b>: {_fmt(mrow['ROC AUC'],4)}"
        )
        story += [Paragraph(detalle, styles['Body'])]
        story += [Spacer(1, 4)]

        cmat = resultados[best_name].get('matriz_confusion', None)
        if cmat is not None:
            story += [_mk_table([["Matriz de confusión (tabla)","",""],
                                 ["", "Pred: No sequía", "Pred: Sequía"],
                                 ["Real: No sequía", str(cmat[0,0]), str(cmat[0,1])],
                                 ["Real: Sequía", str(cmat[1,0]), str(cmat[1,1])]],
                                col_widths=[4.2*RL_CM,4.2*RL_CM,4.2*RL_CM], repeat_rows=1)]
            story += [Spacer(1, 4)]
            story += [_plot_confusion_image(cmat)]
            story += [Spacer(1, 4)]
            story += [Paragraph("Interpretación automática de la matriz de confusión", styles['Body'])]
            for line in _interpret_confusion(cmat):
                story += [Paragraph(line, styles['Body'])]
            story += [Spacer(1, 6)]

        # Tasas detalladas (TPR/FPR/FNR/TNR/Prevalencia)
        stats_cf = resultados[best_name].get('confusion_stats', {})
        if stats_cf:
            tasas_data = [
                ["Métrica","Valor"],
                ["TPR / Recall", _fmt(stats_cf.get('TPR'),3)],
                ["TNR / Especificidad", _fmt(stats_cf.get('TNR'),3)],
                ["FPR", _fmt(stats_cf.get('FPR'),3)],
                ["FNR", _fmt(stats_cf.get('FNR'),3)],
                ["Precisión", _fmt(stats_cf.get('precision'),3)],
                ["Accuracy", _fmt(stats_cf.get('accuracy'),3)],
                ["Prevalencia", _fmt(stats_cf.get('prevalence'),3)],
            ]
            story += [_mk_table(tasas_data, col_widths=[5.0*RL_CM,3.0*RL_CM], repeat_rows=1, font_size=8.6)]
            story += [Spacer(1, 6)]

        if best_models and X_test is not None and y_test is not None and features is not None:
            best_model = best_models[best_name]
            imp_df = _calc_feature_importances_for_pdf(best_model, X_test, y_test, features)
            if imp_df is not None and not imp_df.empty:
                story += [Paragraph("Características más importantes", styles['Body'])]
                data_imp = [["Feature","Importancia"]]
                for _, rr in imp_df.head(10).iterrows():
                    data_imp.append([str(rr['feature']), _fmt(float(rr['importance']),4)])
                story += [_mk_table(data_imp, col_widths=[8*RL_CM,4.6*RL_CM], repeat_rows=1)]
                story += [Spacer(1, 4)]
                story += [_plot_importance_image(imp_df)]
                story += [Spacer(1, 8)]

        if resultados.get(best_name, {}).get('residuos_prob', None) is not None:
            r = pd.Series(resultados[best_name]['residuos_prob'])
            if len(r) <= 5000:
                stat, p = shapiro(r); testname = "Shapiro‑Wilk"
            else:
                stat, p = normaltest(r); testname = "D'Agostino‑Pearson"
            story += [Paragraph(f"Normalidad de residuos (y−p): {testname} → estadístico={_fmt(stat,3)}, p={_fmt(p,4)}", styles['Body'])]
            story += [Spacer(1, 6)]

    if df_levene is not None or df_kw is not None:
        story += [Paragraph("Pruebas estadísticas descriptivas", styles['H2'])]

        if df_levene is not None and not df_levene.empty:
            story += [Paragraph("Levene (homocedasticidad) entre Sequía vs No-sequía", styles['Body'])]
            data_lv = [["Variable","W","p"]]
            for _, r in df_levene.iterrows():
                data_lv.append([r['variable'], _fmt(r['W'],3), _fmt(r['p_levene'],4)])
            story += [_mk_table(data_lv, col_widths=[6*RL_CM,3*RL_CM,3*RL_CM], repeat_rows=1)]
            story += [Spacer(1, 4)]

        if df_kw is not None and not df_kw.empty:
            story += [Paragraph("Kruskal–Wallis por estaciones (1..4)", styles['Body'])]
            data_kw = [["Variable","H","p"]]
            for _, r in df_kw.iterrows():
                data_kw.append([r['variable'], _fmt(r['H'],3), _fmt(r['p_kw'],4)])
            story += [_mk_table(data_kw, col_widths=[6*RL_CM,3*RL_CM,3*RL_CM], repeat_rows=1)]
            story += [Spacer(1, 4)]

            try:
                ph_candidates = [c for c in df_kw['variable'].tolist() if dunn_results.get(c, None) is not None]
                if len(ph_candidates) > 0:
                    var_show = ph_candidates[0]
                    story += [Paragraph(f"Post‑hoc Dunn (Bonferroni) para '{var_show}'", styles['Body'])]
                    ph = dunn_results[var_show]
                    cols = [""] + ph.columns.tolist()
                    data_ph = [cols]
                    for idx in ph.index:
                        data_ph.append([idx] + [str(v) for v in ph.loc[idx].values])
                    story += [_mk_table(data_ph, col_widths=[3*RL_CM]+[2.2*RL_CM]*len(ph.columns), repeat_rows=1)]
                    story += [Spacer(1, 6)]
            except Exception:
                pass

        story += [Paragraph("Interpretación:", styles['Body'])]
        story += [Paragraph("• p<0.05 en Levene → varianzas distintas entre Sequía vs No-sequía.", styles['Body'])]
        story += [Paragraph("• p<0.05 en Kruskal → diferencias de distribución entre estaciones; ver Dunn para pares distintos.", styles['Body'])]
        story += [Spacer(1, 6)]

    story += [Paragraph("Conclusiones y recomendaciones", styles['H2'])]
    story += [Paragraph(
        "Se recomienda priorizar el Recall para reducir falsos negativos (sequías no detectadas). "
        "Ajustar el umbral de decisión, considerar class_weight/SMOTE y calibración de probabilidades. "
        "Utilizar las pruebas estadísticas descriptivas para respaldar diferencias entre estaciones y refinar estrategias de manejo.", styles['Body']
    )]

    doc.build(story)
    pdf_buf.seek(0)
    return pdf_buf

# =====================
# APP (UI)
# =====================
def main():
    st.title("🌾 Sistema de Monitoreo de Sequías Agrícolas en Perú")
    st.caption("Predicción diaria basada en NASA POWER y Machine Learning (con validación temporal, pruebas estadísticas y mejoras de calibración/umbral)")

    # Sidebar
    st.sidebar.title("⚙️ Configuración")
    region = st.sidebar.selectbox("Seleccionar región:", list(REGIONES_ESTRATEGICAS.keys()), index=0)
    cultivo = st.sidebar.selectbox("Seleccionar cultivo:", REGIONES_ESTRATEGICAS[region]["cultivos"], index=0)

    fecha_ini = st.sidebar.date_input(
        "Fecha inicial:", datetime(2020, 1, 1),
        min_value=datetime(2000, 1, 1),
        max_value=datetime.today() - timedelta(days=30)
    )
    fecha_fin = st.sidebar.date_input(
        "Fecha final:", datetime.today() - timedelta(days=1),
        min_value=datetime(2000, 1, 1),
        max_value=datetime.today()
    )

    st.sidebar.markdown("### 🧠 Modelos")
    modelos_disponibles = list(MODEL_CONFIG.keys())
    por_defecto = ['Random Forest', 'XGBoost', 'SVM', 'Híbrido (Stacking RF+XGB+SVM)']
    modelos_seleccionados = [
        m for m in modelos_disponibles
        if st.sidebar.checkbox(m, value=(m in por_defecto), key=f"mdl_{m}")
    ]

    st.sidebar.markdown("### 🔧 Opciones avanzadas")
    balancear = st.sidebar.checkbox("Balancear clases (SMOTE)", True)
    interpolar_sentinelas = st.sidebar.checkbox("Interpolar tras limpiar sentinelas NASA", True)
    usar_cache = st.sidebar.checkbox("Usar caché local NASA", True)
    test_size = st.sidebar.slider("Tamaño conjunto de prueba", 0.1, 0.4, 0.2, 0.05)
    random_state = st.sidebar.number_input("Semilla aleatoria", min_value=0, value=42)

    st.sidebar.markdown("### 🧪 Optimización")
    usar_optuna = st.sidebar.checkbox("Usar Optuna (si está instalado)", False)
    n_trials = st.sidebar.slider("Nº de ensayos (Optuna)", 10, 80, 30, 5)

    st.sidebar.markdown("### 🎯 Objetivo de decisión")
    estrategia_thr = st.sidebar.selectbox("Estrategia de umbral", ["0.5 (por defecto)", "Maximizar F1", "Forzar Recall mínimo"], index=1)
    recall_min_req = st.sidebar.slider("Recall mínimo exigido", 0.50, 0.99, 0.85, 0.01)

    st.sidebar.markdown("### 🎛️ Calibración de probabilidades")
    calibrar = st.sidebar.checkbox("Calibrar probabilidades", True)
    metodo_cal = st.sidebar.selectbox("Método de calibración", ["isotonic", "sigmoid"], index=0)

    # Carga de datos
    with st.spinner("Cargando datos climáticos NASA POWER…"):
        coord = REGIONES_ESTRATEGICAS[region]["coords"]
        umbral = REGIONES_ESTRATEGICAS[region]["umbral_sequia"]
        df_raw = obtener_datos_nasa(
            coord[0], coord[1],
            fecha_ini.strftime("%Y%m%d"),
            fecha_fin.strftime("%Y%m%d"),
            umbral,
            interpolate_missing=interpolar_sentinelas,
            use_cache=usar_cache
        )

    if df_raw is None or df_raw.empty:
        st.error("No se pudieron cargar datos. Cambia el rango de fechas/ubicación y vuelve a intentar.")
        return

    # Outliers por cuantiles (ligero) para robustez EDA/ML
    num_cols_all = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
    df = tratar_outliers_por_cuantiles(df_raw, num_cols_all, low_q=0.001, high_q=0.999)

    st.sidebar.success(f"Datos cargados: {df.shape[0]:,} registros  |  Caché: {'ON' if usar_cache else 'OFF'}")

    with st.sidebar.expander("ℹ️ ¿Qué mide cada métrica?", expanded=False):
        st.markdown("""
- **Precision**: Fracción de predicciones de sequía que realmente son sequía.
- **Recall**: Fracción de sequías reales detectadas (**clave** en riesgo).
- **F1-Score**: Media armónica entre Precision y Recall.
- **PR AUC**: Área bajo Precision–Recall (mejor que ROC en desbalance).
- **ROC AUC**: Separación global entre clases.
- **Balanced Accuracy**: Promedio de sensibilidad y especificidad.
- **MCC**: Correlación predicción‑realidad (robusto a desbalance).
- **Brier Score**: Calidad/calibración de probabilidades (menor es mejor).
- **Kappa (Cohen)**: Acuerdo modelo–realidad más allá del azar.
        """)

    # Tabs
    tabs = st.tabs([
        "📊 EDA",
        "🧩 Selección de características",
        "🤖 Modelado",
        "📈 Resultados",
        "🧪 Estadística",
        #"🔎 Interpretabilidad (opcional)",
        "💾 Exportar"
    ])

    # ---- EDA ----
    with tabs[0]:
        st.header("📊 Análisis Exploratorio de Datos (EDA reforzado)")

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Muestras", f"{df.shape[0]:,}")
        pct_sequia_hoy = df['sequia_hoy'].mean() * 100
        with c2: st.metric("Días con sequía (hoy)", f"{int(df['sequia_hoy'].sum()):,} ({pct_sequia_hoy:.1f}%)")
        with c3: st.metric("Precipitación Promedio", f"{df['precipitacion'].mean():.2f} mm")
        with c4: st.metric("Temp. Máx. Prom.", f"{df['temp_max'].mean():.1f} °C")

        if pct_sequia_hoy > 70 or pct_sequia_hoy < 30:
            st.warning("Datos desbalanceados. Considera usar SMOTE/class_weight y priorizar PR‑AUC, Recall y la matriz de confusión.")

        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        viz_missing_and_descriptives(df, num_cols)

        with st.expander("🧪 Pruebas estadísticas descriptivas (Levene, Kruskal–Wallis, Dunn)", expanded=False):
            df_levene, df_kw, dunn_results = _compute_descriptive_tests(df, num_cols)
            st.markdown("**Levene (Sequía vs No-sequía)** — p<0.05 sugiere varianzas distintas")
            st.dataframe(df_levene.round(4), use_container_width=True)

            st.markdown("**Kruskal–Wallis por Estaciones** — p<0.05 sugiere distribuciones distintas entre estaciones")
            st.dataframe(df_kw.round(4), use_container_width=True)

            vars_ph = [c for c in (df_kw['variable'].tolist() if not df_kw.empty else []) if dunn_results.get(c) is not None]
            if len(vars_ph) > 0:
                var_sel = st.selectbox("Ver Post‑hoc Dunn para la variable:", vars_ph, index=0)
                st.dataframe(dunn_results[var_sel], use_container_width=True)
                st.caption("Celdas = p‑valores ajustados Bonferroni. Valores pequeños → pares de estaciones significativamente distintos.")
            else:
                st.info("Post‑hoc Dunn no disponible (no significativo o falta paquete `scikit-posthocs`).")

            st.session_state['df_levene'] = df_levene
            st.session_state['df_kw'] = df_kw
            st.session_state['dunn_results'] = dunn_results

        with st.expander("⏳ Series temporales", expanded=False):
            vars_show = st.multiselect("Variables a graficar",
                                       ['precipitacion', 'temp_max', 'temp_min', 'humedad',
                                        'balance_hidrico','spi30','anomalia_precipitacion_z'],
                                       default=['precipitacion', 'temp_max'])
            if vars_show:
                fig = go.Figure()
                for v in vars_show:
                    fig.add_trace(go.Scatter(x=df.index, y=df[v], mode='lines', name=v))
                fig.update_layout(hovermode='x unified', title='Evolución temporal')
                st.plotly_chart(fig, use_container_width=True)

    # ---- Selección de características ----
    with tabs[1]:
        st.header("🧩 Selección de características guiada por correlación")
        selected = viz_correlations(df, FEATURES_ALL, target='sequia')
        st.session_state['features_selected'] = selected

    # ---- Modelado ----
    with tabs[2]:
        st.header("🤖 Modelado Predictivo")
        if not modelos_seleccionados:
            st.warning("Selecciona al menos un modelo en la barra lateral.")
            st.stop()

        features = st.session_state.get('features_selected', FEATURES_ALL[:])
        if not features:
            features = FEATURES_ALL[:]

        X, y = df[features], df['sequia']
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # dividir parte de calibración desde el final del train (10% del train)
        if len(X_train) > 100:
            cal_size = max(int(0.1*len(X_train)), 50)
            X_trn, X_cal = X_train.iloc[:-cal_size], X_train.iloc[-cal_size:]
            y_trn, y_cal = y_train.iloc[:-cal_size], y_train.iloc[-cal_size:]
        else:
            X_trn, y_trn, X_cal, y_cal = X_train, y_train, None, None

        # garantizar ambas clases en test
        if not _has_both_classes(y_test):
            found = False
            for extra in [0.05, 0.1, 0.15, 0.2]:
                new_ts = min(0.5, test_size + extra)
                si2 = int(len(X) * (1 - new_ts))
                X_train, X_test = X.iloc[:si2], X.iloc[si2:]
                y_train, y_test = y.iloc[:si2], y.iloc[si2:]
                if _has_both_classes(y_test):
                    found = True
                    st.info(f"Se ajustó el split para garantizar ambas clases en test (test_size={new_ts:.2f}).")
                    break
            if not found:
                st.warning("El conjunto de prueba no contiene ambas clases. Algunas métricas se mostrarán como N/A.")

        # Balanceo opcional con SMOTE ADAPTATIVO
        if balancear and _has_both_classes(y_train):
            sm_tr = _smote_adaptativo(y_train, random_state=random_state, base_k=5)
            if sm_tr is not None:
                X_train, y_train = sm_tr.fit_resample(X_train, y_train)
            else:
                st.warning("No se aplicó SMOTE en entrenamiento: clase minoritaria demasiado pequeña.")

            if X_cal is not None and _has_both_classes(y_cal):
                sm_cal = _smote_adaptativo(y_cal, random_state=random_state, base_k=5)
                if sm_cal is not None:
                    X_cal, y_cal = sm_cal.fit_resample(X_cal, y_cal)
                else:
                    st.warning("No se aplicó SMOTE en calibración: clase minoritaria demasiado pequeña.")
        elif balancear:
            st.info("No se aplica SMOTE: el conjunto de entrenamiento no contiene ambas clases.")

        # Entrenamiento
        scoring_choice = 'f1'
        thr_str_map = {"0.5 (por defecto)": None, "Maximizar F1": "max_f1", "Forzar Recall mínimo": "recall_min"}
        thr_strategy = thr_str_map.get(estrategia_thr, None)

        if st.button("🚀 Entrenar modelos", type="primary"):
            with st.spinner(f"Optimizando {len(modelos_seleccionados)} modelo(s)…"):
                resultados, best_models = optimizar_modelos(
                    X_train, y_train, modelos_seleccionados, features,
                    scoring=scoring_choice, use_optuna=usar_optuna, n_trials=n_trials
                )
                df_metricas, details = evaluar_modelos(
                    resultados, best_models, X_test, y_test,
                    thr_strategy=thr_strategy, recall_min=recall_min_req,
                    calibrate=calibrar, cal_method=metodo_cal,
                    X_cal=X_trn if calibrar else None, y_cal=y_trn if calibrar else None
                )
                friedman_results = test_friedman(df_metricas)

            st.session_state.update({
                'resultados': resultados,
                'best_models': best_models,
                'df_metricas': df_metricas,
                'friedman_results': friedman_results,
                'X_test': X_test, 'y_test': y_test,
                'features': features,
                'thr_strategy': thr_strategy,
                'thr_strategy_text': estrategia_thr,
                'recall_min_req': recall_min_req
            })

            st.success("¡Modelos entrenados!")
            st.subheader("📋 Resultados preliminares")
            st.dataframe(df_metricas.sort_values('F1-Score', ascending=False).round(4), use_container_width=True)

            if friedman_results:
                with st.expander("📉 Comparación Estadística (Test de Friedman)", expanded=True):
                    for metrica, r in friedman_results.items():
                        stat = r.get('Estadístico', np.nan)
                        p = r.get('p-valor', np.nan)
                        st.markdown(f"**{metrica}** — χ²: **{stat if pd.notna(stat) else 'N/A'}** · p: **{p if pd.notna(p) else 'N/A'}**")
                    st.caption("Interpretación: Un p-valor < 0.05 indica diferencias significativas entre los modelos.")

            viz_model_comparison(df_metricas)
            viz_pred_vs_real(resultados, y_test)
            viz_residual_tests(resultados)

    # ---- Resultados ----
    with tabs[3]:
        st.header("📈 Resultados Detallados")
        if 'resultados' not in st.session_state:
            st.info("Primero entrena los modelos en la pestaña **🤖 Modelado**.")
        else:
            res = st.session_state['resultados']
            best_models = st.session_state['best_models']
            dfm = st.session_state['df_metricas']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']

            best_name = dfm.loc[dfm['F1-Score'].idxmax(), 'Modelo']
            st.success(f"🏆 Mejor Modelo: **{best_name}**")

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("F1-Score", f"{dfm.loc[dfm['Modelo']==best_name,'F1-Score'].values[0]:.4f}")
            with c2: st.metric("Precision", f"{dfm.loc[dfm['Modelo']==best_name,'Precision'].values[0]:.4f}")
            with c3: st.metric("Recall", f"{dfm.loc[dfm['Modelo']==best_name,'Recall'].values[0]:.4f}")
            with c4: st.metric("Kappa", f"{dfm.loc[dfm['Modelo']==best_name,'Kappa'].values[0]:.4f}")
            st.caption(f"Threshold aplicado: {dfm.loc[dfm['Modelo']==best_name,'Threshold'].values[0]:.3f} · Calibrado: {'Sí' if bool(dfm.loc[dfm['Modelo']==best_name,'Calibrado'].values[0]) else 'No'}")

            with st.expander("📊 Matrices de Confusión (2x2)", expanded=True):
                cols = st.columns(2)
                for i, name in enumerate(res.keys()):
                    cm = res[name]['matriz_confusion']
                    figm = px.imshow(cm, text_auto=True,
                                     labels=dict(x="Predicho", y="Real", color="Casos"),
                                     x=['No Sequía','Sequía'], y=['No Sequía','Sequía'],
                                     title=f"{name}")
                    cols[i % 2].plotly_chart(figm, use_container_width=True)

            with st.expander("📈 Curvas ROC y Precision‑Recall", expanded=True):
                fig_roc, fig_pr = go.Figure(), go.Figure()
                for name in res.keys():
                    y_proba = res[name]['y_proba']
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC={roc_auc:.2f})'))
                    P, R, _ = precision_recall_curve(y_test, y_proba)
                    pr_auc = auc(R, P)
                    fig_pr.add_trace(go.Scatter(x=R, y=P, mode='lines', name=f'{name} (AUC={pr_auc:.2f})'))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Aleatorio', line=dict(dash='dash')))
                fig_roc.update_layout(title='ROC', xaxis_title='Falsos Positivos', yaxis_title='Verdaderos Positivos')
                fig_pr.update_layout(title='Precision‑Recall', xaxis_title='Recall', yaxis_title='Precision')
                c1, c2 = st.columns(2)
                c1.plotly_chart(fig_roc, use_container_width=True)
                c2.plotly_chart(fig_pr, use_container_width=True)

            with st.expander("🔍 Importancia de Características", expanded=True):
                best_model = best_models[best_name]
                try:
                    r = permutation_importance(best_model, X_test, y_test, scoring="f1",
                                               n_repeats=10, random_state=42, n_jobs=-1)
                    mask = best_model.named_steps['preprocessor']\
                                     .named_transformers_['num']\
                                     .named_steps['selector'].get_support()
                    base_names = st.session_state.get('features', X_test.columns.tolist())
                    if len(base_names) < sum(mask):
                        base_names = X_test.columns.tolist()
                    selected = [n for n, keep in zip(base_names, mask) if keep]
                    imp = pd.DataFrame({"Característica": selected, "Importancia": r.importances_mean})\
                            .sort_values("Importancia", ascending=False)
                    fig_imp = px.bar(imp, x="Característica", y="Importancia",
                                     title="Permutation Importance (impacto en F1)")
                    st.plotly_chart(fig_imp, use_container_width=True)
                except Exception as e:
                    st.warning(f"No fue posible calcular importancias por permutación: {e}")
                    inner = getattr(best_model, 'named_steps', {})
                    mdl = inner.get('model', None) if isinstance(inner, dict) else None
                    fi = getattr(mdl, 'feature_importances_', None) if mdl is not None else None
                    if fi is not None:
                        mask = best_model.named_steps['preprocessor']\
                                         .named_transformers_['num']\
                                         .named_steps['selector'].get_support()
                        base_names = X_test.columns.tolist()
                        selected = [n for n, keep in zip(base_names, mask) if keep]
                        gini = pd.DataFrame({"Característica": selected, "Importancia": fi})\
                                .sort_values("Importancia", ascending=False)
                        fig_g = px.bar(gini, x="Característica", y="Importancia",
                                       title="Importancia de Características (Gini)")
                        st.plotly_chart(fig_g)
                    else:
                        st.info("El modelo no expone importancias y no fue posible calcularlas.")

    # ---- Estadística ----
    with tabs[4]:
        st.header("🧪 Pruebas Estadísticas")
        if 'df_metricas' not in st.session_state:
            st.info("Primero entrena los modelos en la pestaña **🤖 Modelado**.")
        else:
            dfm = st.session_state['df_metricas']
            fried = st.session_state.get('friedman_results', {})
            if fried:
                with st.expander("Test de Friedman (rangos)", expanded=True):
                    for m, r in fried.items():
                        st.markdown(f"**{m}** — χ²: **{r.get('Estadístico','N/A')}** · p: **{r.get('p-valor','N/A')}**")
                    st.caption("p < 0.05 → diferencias significativas.")
            else:
                st.info("No hay suficientes modelos válidos/variabilidad para Friedman.")

            with st.expander("Post‑hoc (Nemenyi) — opcional", expanded=True):
                metrica_ph = st.selectbox("Métrica para post‑hoc:", ['F1-Score', 'PR AUC', 'Kappa'], index=0)
                ph, msg = posthoc_nemenyi(dfm, metrica=metrica_ph)
                if ph is not None:
                    st.dataframe(ph.round(4), use_container_width=True)
                    st.caption("Valores pequeños sugieren pares de modelos significativamente distintos.")
                else:
                    st.info(msg or "No fue posible ejecutar el post‑hoc.")

    # ---- Interpretabilidad (opcional) ----
    with tabs[5]:
        st.header("🔎 Interpretabilidad (SHAP / PDP)")
        if 'resultados' not in st.session_state:
            st.info("Primero entrena los modelos.")
        else:
            best_name = st.session_state['df_metricas'].loc[st.session_state['df_metricas']['F1-Score'].idxmax(), 'Modelo']
            best_model = st.session_state['best_models'][best_name]
            X_test = st.session_state['X_test']
            st.markdown(f"**Modelo seleccionado:** {best_name}")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Mostrar SHAP (si disponible)"):
                    try:
                        import shap
                        Xt = best_model.named_steps['preprocessor'].transform(X_test)
                        shap_values = shap.Explainer(best_model.named_steps['model'])(Xt)
                        st.info("Se generaron valores SHAP (resumen no embebido por limitaciones de plotting aquí).")
                        st.caption("Para visualización avanzada, exporta el modelo y usa un notebook.")
                    except Exception as e:
                        st.warning(f"SHAP no disponible o no compatible en este entorno: {e}")

            with col2:
                if st.button("Mostrar PDP (Partial Dependence) de 2 variables (si disponible)"):
                    try:
                        import matplotlib
                        matplotlib.use("Agg")
                        feats = st.session_state.get('features', X_test.columns.tolist())
                        target_feats = st.multiselect("Escoge variables para PDP (máx. 2)", feats, default=feats[:2], key="pdp_feats")
                        if len(target_feats) > 0:
                            fig = plt.figure(figsize=(6,4), dpi=120)
                            PartialDependenceDisplay.from_estimator(best_model, X_test, target_feats)
                            st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"No fue posible generar PDP: {e}")

    # ---- Exportar ----
    with tabs[6]:
        st.header("💾 Exportar Modelos y Resultados")
        if 'resultados' not in st.session_state:
            st.info("Primero entrena los modelos en la pestaña **🤖 Modelado**.")
        else:
            res = st.session_state['resultados']
            best_models = st.session_state['best_models']
            dfm = st.session_state['df_metricas']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            features = st.session_state['features']

            best_name = dfm.loc[dfm['F1-Score'].idxmax(), 'Modelo']
            best_model = best_models[best_name]

            st.subheader(f"📦 Exportar Mejor Modelo ({best_name})")
            paquete = {
                'model': best_model,
                'features': features,
                'metrics': dfm[dfm['Modelo'] == best_name].to_dict('records')[0],
                'fecha_entrenamiento': datetime.now(),
                'region': region,
                'cultivo': cultivo
            }
            buf = BytesIO(); pickle.dump(paquete, buf); buf.seek(0)
            st.download_button(
                "⬇️ Descargar Modelo (.pkl)",
                data=buf,
                file_name=f"modelo_sequias_{best_name}_{datetime.now().strftime('%Y%m%d')}.pkl",
                mime="application/octet-stream"
            )

            st.subheader("📊 Exportar Resultados (Excel)")
            excel_buf = BytesIO()
            with pd.ExcelWriter(excel_buf, engine='xlsxwriter') as w:
                dfm.round(4).to_excel(w, sheet_name='Métricas', index=False)
                params_rows = []
                for n, r2 in res.items():
                    pr = {'Modelo': n, 'Tiempo (s)': r2.get('tiempo_entrenamiento_seg', np.nan),
                          'Threshold': r2.get('threshold', 0.5), 'Calibrado': r2.get('calibrado', False)}
                    pr.update({k.replace('model__', ''): v for k, v in r2.get('mejores_parametros', {}).items()})
                    params_rows.append(pr)
                pd.DataFrame(params_rows).to_excel(w, sheet_name='Parámetros', index=False)

                y_pred = res[best_name]['y_pred']
                y_proba = res[best_name]['y_proba']
                pred = pd.DataFrame({'Real': y_test, 'Predicho': y_pred, 'Probabilidad': y_proba})
                pred.to_excel(w, sheet_name='Predicciones', index=False)
            excel_buf.seek(0)
            st.download_button(
                "📄 Descargar Resultados (.xlsx)",
                data=excel_buf,
                file_name=f"resultados_sequias_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.subheader("📑 Reporte Ejecutivo (PDF)")
            if st.button("📑 Generar Reporte Ejecutivo (PDF)"):
                df_levene = st.session_state.get('df_levene', None)
                df_kw = st.session_state.get('df_kw', None)
                dunn_results = st.session_state.get('dunn_results', None)
                thr_strategy_text = st.session_state.get('thr_strategy_text', None)
                recall_min_req = st.session_state.get('recall_min_req', None)

                pdf_buf = generar_reporte_pdf(
                    df=df,
                    region=region,
                    cultivo=cultivo,
                    fecha_ini=fecha_ini.strftime("%Y-%m-%d"),
                    fecha_fin=fecha_fin.strftime("%Y-%m-%d"),
                    df_metricas=dfm,
                    resultados=res,
                    best_models=best_models,
                    features=features,
                    X_test=X_test, y_test=y_test,
                    df_levene=df_levene,
                    df_kw=df_kw,
                    dunn_results=dunn_results,
                    thr_strategy_text=thr_strategy_text,
                    recall_min_req=recall_min_req if thr_strategy_text == "Forzar Recall mínimo" else None
                )
                st.download_button(
                    "⬇️ Descargar PDF",
                    data=pdf_buf,
                    file_name=f"reporte_ejecutivo_{region.replace(' ','_')}_{cultivo}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )

    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align:center;color:#888;padding:10px'>
            <b>🌾 Sistema de Monitoreo de Sequías Agrícolas en Perú</b><br>
            <small>Última actualización: {datetime.now().strftime('%Y-%m-%d')} · Versión 5.1
            (reintentos + caché NASA, outliers, features extendidas, SPI30, calibración, umbral objetivo,
            SMOTE adaptativo, Optuna opcional, PDF apaisado con tablas detalladas)</small>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
