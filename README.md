# 🌾 Sistema de Monitoreo de Sequías Agrícolas (Streamlit)

App Streamlit para análisis y predicción de sequías con datos **NASA POWER** y modelos de ML.
Archivo principal: `sequias.py`.

---

## 📦 Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

> Incluye paquetes como `xgboost`, `lightgbm`, `imbalanced-learn`, `reportlab` y `kaleido` (para exportar gráficos).

---

## 🚀 Ejecutar localmente

```bash
streamlit run sequias.py
```

La app se abrirá en tu navegador (por defecto en `http://localhost:8501/`).

---

## ☁️ Despliegue 1: Streamlit Cloud (recomendado)

1. **Sube a GitHub** este repositorio con:
   - `sequias.py`
   - `requirements.txt`
   - (opcional) `README.md`

2. Ve a **https://streamlit.io/cloud** → **New app**.
3. Elige tu repositorio, rama (ej. `main`) y archivo principal: `sequias.py`.
4. Pulsa **Deploy**. Al terminar tendrás una URL pública como:  
   `https://tuusuario-<nombre>-streamlit.app`

### Notas
- La app usa una **caché local** para acelerar consultas (`.cache_nasa`). En Streamlit Cloud el sistema de archivos no es persistente a largo plazo, por lo que la caché puede regenerarse.
- Si agregas archivos de datos, súbelos también al repo o ajusta el código para descargarlos dinámicamente.
- Los paquetes pesados (`xgboost`, `lightgbm`) pueden tardar un poco en instalarse la primera vez. 

---

## 🤗 Despliegue 2: Hugging Face Spaces (alternativa)

1. Crea un Space en **https://huggingface.co/spaces** → **Create new Space**.
2. Tipo de Space: **Streamlit**.
3. Sube `sequias.py` y `requirements.txt` (o conecta el repo de GitHub).
4. El Space construirá la app automáticamente y te dará una URL pública.

---

## 🔧 Variables y configuración

- No se requieren llaves API para NASA POWER (datos públicos).  
- Si en el futuro agregas credenciales, puedes gestionarlas como **Secrets** en Streamlit Cloud (⚙️ → Secrets) o como **Variables** del Space en HF.

---

## 🧪 Pruebas rápidas

- Cambia la **región**, **cultivo** y **rango de fechas** desde la barra lateral.
- Presiona **“🚀 Entrenar modelos”** en la pestaña **🤖 Modelado** para ver métricas, matrices de confusión, curvas ROC/PR y exportaciones (Excel/PDF/PKL).

---

## 🗂️ Estructura sugerida del repo

```text
/
├─ sequias.py
├─ requirements.txt
├─ README.md
└─ .gitignore             # opcional
```

**.gitignore** (opcional):
```gitignore
__pycache__/
.cache_nasa/
.streamlit/
*.pkl
*.xlsx
*.pdf
```

---

## ❓Soporte

Si algo falla en el build (por ejemplo, una versión de paquete), prueba a **actualizar** o **fijar** versiones en `requirements.txt`.
