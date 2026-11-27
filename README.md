# natural-language-scheduler

Asistente de agenda en lenguaje natural — práctica final de PLN 2026/27. El plan de trabajo por fases está en [PLAN.md](PLAN.md).

## Requisitos

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (o `pip`)
- ~10 GB libres en disco: unos 3 GB para las dependencias (torch, transformers) y unos 5 GB para los modelos (Qwen2.5-1.5B-Instruct ≈ 3 GB, BETO ≈ 0,4 GB, MiniLM ≈ 0,5 GB, spaCy ≈ 40 MB)

## Instalación

```bash
uv venv --python 3.12 --managed-python .venv
```

(`--managed-python` usa un Python descargado por uv en lugar del de Anaconda/sistema.)

```bash
source .venv/bin/activate
```

```bash
uv pip install -r requirements.txt
```

```bash
uv pip install https://github.com/explosion/spacy-models/releases/download/es_core_news_md-3.8.0/es_core_news_md-3.8.0-py3-none-any.whl
```

(Un entorno creado con uv no trae `pip`, así que `python -m spacy download es_core_news_md` falla; con `pip` normal sí funciona.)

```bash
python -m ipykernel install --user --name pln-agenda --display-name "PLN agenda (.venv)"
```

Las dependencias directas están en `requirements.in`. `requirements.txt` tiene las versiones fijadas y se regenera con `uv pip compile requirements.in -o requirements.txt`.

## Estructura

| Ruta | Contenido |
|---|---|
| `train.jsonl`, `agenda.jsonl`, `evaluar.py` | Datos y script de evaluación oficiales (no modificar) |
| `plantillas/` | Plantillas originales del profesor (no modificar) |
| `1_corpus.ipynb` … `4_asistente.ipynb` | Notebooks de trabajo (copias de las plantillas) |
| `utilidades.py` | Funciones comunes (carga, guardado, anotación, preprocesado) |
| `tarea_a.py`, `tarea_b.py`, `tarea_c.py` | Enfoques elegidos, exportados para el asistente y `generar_predicciones` |
| `datos/` | Frases propias y particiones |
| `modelos/` | Modelos entrenados y embeddings cacheados |
| `resultados/` | `resultados_*.json` y figuras para la memoria |
| `memoria/` | Fuente de la memoria |

## Orden de ejecución

1. `1_corpus.ipynb`: genera `mi_train.jsonl` y `mi_validacion.jsonl` (una sola vez)
2. `2A_intencion.ipynb`, `2B_extraccion.ipynb`, `2C_consulta.ipynb`: generan `resultados/resultados_*.json` y `tarea_*.py`
3. `3_evaluacion.ipynb`: tabla comparativa, experimentos E1–E4 y `generar_predicciones`
4. `4_asistente.ipynb`: asistente y celda de demostración

Para evaluar unas predicciones desde la terminal:

```bash
python evaluar.py mi_validacion.jsonl pred_validacion.jsonl
```
