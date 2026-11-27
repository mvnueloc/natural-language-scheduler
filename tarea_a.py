"""Tarea A · Intención. Enfoques del notebook 2A reutilizables desde los notebooks 3 y 4.

- intencion_reglas(texto)                  → A1 (reglas ordenadas por prioridad)
- crear_modelo(enfoque, **params)          → pipeline sin entrenar de A2, A3, A4 o A5
- entrenar_y_predecir(enfoque, X, y, Xv)   → lo usan E1 y E2 (notebook 3)
- predecir_intencion(texto)                → enfoque elegido, cargado de modelos/ (lo usan el asistente y generar_predicciones)

Las reglas de A1 y los hiperparámetros se han diseñado mirando solo mi_train.jsonl (nunca la validación).
"""
import os, re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from utilidades import normalizar, sin_tildes, tokenizar

SEMILLA = 0
RUTA_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos", "tarea_a.joblib")

# Configuraciones de preprocesado (las mismas que la ablación E2)
PREPRO = {
    "original": dict(minusculas=False),
    "normalizado": dict(quitar_tildes=True, abreviaturas=True, numeros=True),
    "sin_stopwords": dict(quitar_tildes=True, abreviaturas=True, numeros=True, quitar_stopwords=True),
    "abreviaturas": dict(abreviaturas=True, numeros=True),    # sin quitar tildes (para los vectores de spaCy, A5)
    "stemming": dict(quitar_tildes=True, abreviaturas=True, numeros=True, quitar_stopwords=True, raiz="stem"),
}


# ---------------------------------------------------------------------------
# A1 · Reglas
# ---------------------------------------------------------------------------
# Se aplican sobre el texto en minúsculas, sin tildes y con las abreviaturas de mensajería expandidas.
# Orden de prioridad: otro (fuera de dominio claro) → borrar → completar → modificar → buscar → consultar
# → crear (tarea o evento) → otro. Cada regla mira solo la frase; no hay pesos ni entrenamiento.

def _r(patron):
    return re.compile(patron)

# Fuera de dominio que se parece al dominio (fechas, «cuándo», «a qué hora»…): se comprueba lo primero.
R_OTRO = _r(r"\btiempo (hace|hara|va a hacer)\b|\bllover|\bllueve|\bjuega el\b|\bgano el\b|\bpartido de ayer\b"
            r"|\bcapital de\b|\bcuanto (es|se tarda|cuesta)\b|\babre el\b|\bcierra el\b|^que hora es\b|\bpuente de\b"
            r"|\bchiste\b|\bmusica\b|\btraduce\b|\bexplicame\b|\brobot\b|\brecomiendame\b|\bque le regalo\b"
            r"|\bque puedes hacer\b|\bno me apetece\b|\bquien (es|ganó|gano)\b")

# borrar: verbos de eliminación (no negados) o «se ha cancelado/suspendido».
R_NEG_BORRAR = _r(r"\bno (borres|quites|elimines|canceles|anules)\b")
R_BORRAR = _r(r"\b(borr|elimin|quit|cancel|anul)\w*|\bolvidate de\b|\bse ha (cancelado|suspendido)\b"
              r"|\b(han|ha) (cancelado|suspendido|anulado)\b")

# completar: «ya …» (sin «ya no»), participios de hecho, «he + participio», tachar/marcar como hecho.
R_COMPLETAR = _r(r"\bya\b(?! no\b)|\btach\w*|\bmarca\w* como (hech|complet|termin)|\blisto\b|^hecho\b|\bhecho lo\b"
                 r"|(?<!no )\bhe \w+(ado|ido|cho)\b|\bhemos \w+(ado|ido)\b")
R_PARTICIPIO = _r(r"(?<!tener )(?<!tenga )\b(pagad|hech|terminad|devuelt|entregad|lavad|subid|renovad|comprad"
                  r"|enviad|acabad|mandad|leid)[oa]s?\b")

# modificar: verbos de cambio de fecha/hora/lugar. Se quita antes «pasado mañana», que no es «pasar».
R_MODIFICAR = _r(r"\b(cambi|muev|movid|movel|retras|adelant|aplaz|traslad)\w*|\bpasa\w* (lo|el|la|al|a las|para)\b"
                 r"|\b(han|ha|lo han|la han|se) pasad[oa]\b|\bse pasa\b|\bahora es\b|\bal final es\b"
                 r"|\bempieza (mas tarde|antes|a las)\b|\bdejal[oa] (para|pa)\b|\bmejor el\b|\bno es el \w+, es\b")

# buscar: pregunta por un dato de UNA entrada concreta (cuándo, dónde, a qué hora, qué día…).
R_BUSCAR = _r(r"\b(cuando|donde|a que hora|que dia|hasta cuando|para cuando)\b|\btenia\b|\bera\b")
R_NO_BUSCAR = _r(r"\blo primero\b|\bla primera cosa\b|\bque tengo\b")

# consultar: pregunta por lo que hay en un día o periodo, o pide un listado/resumen.
R_CONSULTAR = _r(r"\bque (tengo|hay|planes|tareas|entregas|cosas|me queda)\b|\btengo algo\b|\bestoy libre\b"
                 r"|\bense[nñ]\w*|\bdime\b|\bresum\w*|\blista\b|\bcomo tengo\b|\bme queda algo\b|\bagenda de\b"
                 r"|^que tengo\b|\bhay algo\b|\blo primero\b|\bla primera cosa\b")

# crear: órdenes explícitas de apuntar y marcas de tarea frente a evento.
R_CREAR = _r(r"^(\w+,? )?(apunta|apuntame|anade|crea|mete|pon|recuerdame|recordarme|nueva tarea|no te olvides"
             r"|acuerdate)\b|\bapunta(me)? (que|en|una?|el|la)\b")
R_TAREA = _r(r"\btengo que\b|\bhay que\b|\bpendiente|\btarea\b|\brecuerdame\b|\brecordarme\b|\bacuerdate\b"
             r"|\bno te olvides\b|\bantes del\b|\burgente\b|\bprisa\b|\bsin falta\b|\bcuando puedas\b"
             r"|\bpara el (dia )?\d|\bpara el (lunes|martes|miercoles|jueves|viernes|sabado|domingo)\b"
             r"|^(comprar|llamar|devolver|preparar|hacer|entregar|enviar|mandar|estudiar|imprimir|pagar|renovar"
             r"|escribir|leer|repasar|subir)\b")
R_HORA = _r(r"\ba las? \d|\ba las (una|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce)\b|\bmediodia\b"
            r"|\ba la hora de\b|\d:\d\d")
R_EVENTO = _r(r"\b(reunion|cita|cena|ceno|comida|quedada|quedo|examen|tutoria|clase|seminario|concierto|partido"
              r"|fiesta|cine|evento|charla|entrevista|videollamada|entreno|entrenamiento|dentista|medico|fisio"
              r"|desayuno|cumple\w*|revision|jornada|ensayo)\b")
R_FECHA = _r(r"\b(hoy|manana|lunes|martes|miercoles|jueves|viernes|sabado|domingo|semana|finde|fin de semana)\b"
             r"|\b(el|dia|del) \d+\b|\d+ de (enero|febrero|marzo|abril|mayo|junio)")


def _preparar(texto):
    t = normalizar(texto, quitar_tildes=True, abreviaturas=True)
    t = re.sub(r"\s+([,.;:!?¿¡])", r"\1", t)              # «hola , que» → «hola, que» tras tokenizar
    t = re.sub(r"[¿¡!?\"]", " ", t)
    t = re.sub(r"\bpasado manana\b", "PASADOMANANA manana", t)   # que «pasado» no dispare modificar
    return re.sub(r"\s+", " ", t).strip()


def intencion_reglas(texto):
    """A1: devuelve una de las 8 intenciones aplicando reglas ordenadas por prioridad."""
    t = _preparar(texto)
    if R_OTRO.search(t):
        return "otro"
    if R_BORRAR.search(R_NEG_BORRAR.sub(" ", t)):
        return "borrar"
    crear = bool(R_CREAR.search(t))
    if not crear and (R_COMPLETAR.search(t) or R_PARTICIPIO.search(t)):
        return "completar"
    if not crear and R_MODIFICAR.search(t):
        return "modificar"
    if not crear and R_BUSCAR.search(t) and not R_NO_BUSCAR.search(t):
        return "buscar"
    if not crear and R_CONSULTAR.search(t):
        return "consultar"
    # crear: primero la hora (un evento tiene hora), luego las marcas de tarea, luego las de evento
    if crear or R_TAREA.search(t) or R_EVENTO.search(t) or R_HORA.search(t):
        if R_HORA.search(t) or (R_EVENTO.search(t) and not R_TAREA.search(t)) or re.search(r"\bevento\b", t):
            return "crear_evento"
        if R_TAREA.search(t) or crear:
            return "crear_tarea"
    if R_FECHA.search(t) and R_EVENTO.search(t):
        return "crear_evento"
    return "otro"


# ---------------------------------------------------------------------------
# Preprocesado como paso del pipeline (así el modelo guardado incluye su normalización)
# ---------------------------------------------------------------------------

class Preprocesar(BaseEstimator, TransformerMixin):
    """Aplica normalizar() con una de las configuraciones de PREPRO."""

    def __init__(self, config="normalizado"):
        self.config = config

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [normalizar(t, **PREPRO[self.config]) for t in X]


# ---------------------------------------------------------------------------
# A4 · Word2Vec propio  /  A5 · vectores preentrenados de spaCy: media de vectores por frase
# ---------------------------------------------------------------------------

def _palabras(texto):
    return [w for w in tokenizar(texto) if re.match(r"\w", w)]


class MediaWord2Vec(BaseEstimator, TransformerMixin):
    """Entrena Word2Vec (gensim) con las frases de entrenamiento y representa cada frase por la media de sus vectores."""

    def __init__(self, dim=100, ventana=5, epocas=50, sg=1, min_count=1, semilla=SEMILLA):
        self.dim, self.ventana, self.epocas, self.sg, self.min_count, self.semilla = dim, ventana, epocas, sg, min_count, semilla

    def fit(self, X, y=None):
        from gensim.models import Word2Vec
        frases = [_palabras(t) for t in X]
        # workers=1 + semilla fija → entrenamiento reproducible
        self.w2v_ = Word2Vec(frases, vector_size=self.dim, window=self.ventana, min_count=self.min_count,
                             sg=self.sg, epochs=self.epocas, seed=self.semilla, workers=1).wv
        return self

    def transform(self, X):
        out = np.zeros((len(X), self.dim), dtype=np.float32)
        for i, t in enumerate(X):
            vs = [self.w2v_[w] for w in _palabras(t) if w in self.w2v_]
            if vs:
                out[i] = np.mean(vs, axis=0)
        return out


class MediaSpacy(BaseEstimator, TransformerMixin):
    """Media de los vectores preentrenados de es_core_news_md (300 dim.). solo_contenido: ignora stopwords y signos."""

    def __init__(self, solo_contenido=False):
        self.solo_contenido = solo_contenido

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from utilidades import _nlp
        nlp = _nlp()
        out = np.zeros((len(X), nlp.vocab.vectors_length), dtype=np.float32)
        for i, doc in enumerate(nlp.pipe(X, disable=["parser", "ner", "lemmatizer"])):
            toks = [t for t in doc if t.has_vector and not t.is_punct
                    and not (self.solo_contenido and t.is_stop and t.lower_ not in {"no", "ya", "que", "qué", "cuándo"})]
            if toks:
                out[i] = np.mean([t.vector for t in toks], axis=0)
        return out


# ---------------------------------------------------------------------------
# Pipelines de A2–A5 (hiperparámetros por defecto = los elegidos por validación cruzada en mi_train, notebook 2A)
# ---------------------------------------------------------------------------

DEFECTO = {
    "A2": dict(prepro="normalizado", ngramas=(1, 1), alpha=0.1),
    "A3": dict(prepro="normalizado", ngramas=(1, 2), chars=(2, 4), C=10.0),
    "A4": dict(prepro="normalizado", dim=100, epocas=50, C=1.0),
    "A5": dict(prepro="abreviaturas", solo_contenido=False, C=0.1),
}


def crear_modelo(enfoque, **params):
    p = {**DEFECTO[enfoque], **params}
    pre = ("prepro", Preprocesar(p["prepro"]))
    if enfoque == "A2":
        return Pipeline([pre, ("bow", CountVectorizer(lowercase=False, ngram_range=p["ngramas"], token_pattern=r"(?u)\b\w+\b")),
                         ("nb", MultinomialNB(alpha=p["alpha"]))])
    if enfoque == "A3":
        rasgos = FeatureUnion([
            ("palabras", TfidfVectorizer(lowercase=False, ngram_range=p["ngramas"], sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
            ("chars", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=p["chars"], sublinear_tf=True)),
        ])
        return Pipeline([pre, ("tfidf", rasgos),
                         ("rl", LogisticRegression(C=p["C"], max_iter=5000, random_state=SEMILLA))])
    if enfoque == "A4":
        return Pipeline([pre, ("w2v", MediaWord2Vec(dim=p["dim"], epocas=p["epocas"])), ("esc", StandardScaler()),
                         ("rl", LogisticRegression(C=p["C"], max_iter=5000, random_state=SEMILLA))])
    if enfoque == "A5":
        # spaCy espera texto natural: por defecto solo minúsculas, abreviaturas y números, sin quitar tildes
        return Pipeline([pre, ("spacy", MediaSpacy(p["solo_contenido"])), ("esc", StandardScaler()),
                         ("rl", LogisticRegression(C=p["C"], max_iter=5000, random_state=SEMILLA))])
    raise NotImplementedError(f"{enfoque}: se implementa en la fase 3 (A6/A7)")


def entrenar_y_predecir(enfoque, textos_train, etiquetas_train, textos_val, prepro=None):
    """Entrena `enfoque` (A2–A5; A6/A7 en la fase 3) y predice las intenciones de textos_val.

    prepro: None (el del enfoque por defecto) o una clave de PREPRO (para la ablación E2)."""
    params = {} if prepro is None else {"prepro": prepro}
    modelo = crear_modelo(enfoque, **params).fit(list(textos_train), list(etiquetas_train))
    return list(modelo.predict(list(textos_val)))


# ---------------------------------------------------------------------------
# Enfoque elegido (provisional en la fase 2: el mejor clásico; se revisa en la fase 3 con A6–A8)
# ---------------------------------------------------------------------------
_modelo = None


def predecir_intencion(texto):
    global _modelo
    if _modelo is None:
        import joblib
        _modelo = joblib.load(RUTA_MODELO)
    return str(_modelo.predict([texto])[0])
