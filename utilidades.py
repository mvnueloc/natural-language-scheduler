"""Funciones comunes de la práctica (las del anexo del enunciado). Añadid aquí lo que uséis en varios notebooks."""
import json, os, re, unicodedata, datetime as dt

HOY = dt.date(2026, 2, 9)          # lunes 9 de febrero de 2026
INTENCIONES = ["crear_evento", "crear_tarea", "consultar", "buscar", "completar", "modificar", "borrar", "otro"]
TIPOS = ["FECHA", "HORA", "LUGAR", "PERSONA", "ASIGNATURA", "PRIORIDAD"]
# Tipos de dificultad que usamos en el campo propio "dificultad" (evaluar.py lo ignora; sirve para E3)
DIFICULTADES = ["ortografia", "abreviatura", "mayusculas", "negacion", "fecha_relativa",
                "numeros_letras", "larga", "fuera_dominio", "parafrasis", "contexto"]


def cargar(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def guardar(filas, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        for x in filas:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def anotar(id_, texto, intencion, entidades, ref=None, dificil=False, dificultad=None):
    """entidades: lista de (trozo, tipo) o (trozo, tipo, valor). Calcula inicio y fin.

    dificultad (opcional): uno de DIFICULTADES; si se indica, la frase se marca como difícil."""
    ents = []
    for trozo, tipo, *valor in entidades:
        i = texto.index(trozo)
        e = {"tipo": tipo, "texto": trozo, "inicio": i, "fin": i + len(trozo)}
        if valor:
            e["valor"] = valor[0]
        ents.append(e)
    x = {"id": id_, "texto": texto, "intencion": intencion, "entidades": ents, "ref": ref,
         "dificil": bool(dificil or dificultad)}
    if dificultad:
        assert dificultad in DIFICULTADES, dificultad
        x["dificultad"] = dificultad
    return x


def comprobar(corpus):
    """Comprueba que inicio/fin de cada entidad corresponden a su texto."""
    for x in corpus:
        for e in x["entidades"]:
            assert x["texto"][e["inicio"]:e["fin"]] == e["texto"], (x["id"], e)
    return True


def trozos_repetidos(corpus):
    """Entidades cuyo trozo aparece más de una vez en la frase (anotar() coge la primera aparición: revisar a mano)."""
    return [(x["id"], e["texto"], x["texto"]) for x in corpus for e in x["entidades"]
            if x["texto"].count(e["texto"]) > 1]


def particiones():
    """Devuelve (entrenamiento, validación). Usa los ficheros del notebook 1; si aún no existen, divide train.jsonl."""
    if os.path.exists("mi_train.jsonl") and os.path.exists("mi_validacion.jsonl"):
        return cargar("mi_train.jsonl"), cargar("mi_validacion.jsonl")
    from sklearn.model_selection import train_test_split
    corpus = cargar("train.jsonl")
    print("Aviso: aún no existen mi_train.jsonl y mi_validacion.jsonl (notebook 1); uso una división provisional de train.jsonl")
    return train_test_split(corpus, test_size=0.2, random_state=0, stratify=[x["intencion"] for x in corpus])


# ---------------------------------------------------------------------------
# Preprocesado (notebook 1, apartado 1.4). Lo usan E2, C1 y el asistente.
# ---------------------------------------------------------------------------

# Abreviaturas y grafías de mensajería → forma estándar (claves en minúsculas y sin tildes).
# No incluimos «a» ni «e»: son palabras normales del español («a las 5», «Elena e Inés»).
ABREVIATURAS = {
    "q": "que", "k": "que", "ke": "que", "xq": "porque", "pq": "porque", "porq": "porque",
    "x": "por", "xa": "para", "pa": "para", "d": "de", "cn": "con", "tb": "también", "tmb": "también",
    "mñn": "mañana", "mñna": "mañana", "mna": "mañana", "mnn": "mañana",
    "tng": "tengo", "tngo": "tengo", "tgo": "tengo", "tngs": "tienes",
    "dnd": "dónde", "qndo": "cuándo", "cdo": "cuándo", "ns": "no sé", "nse": "no sé",
    "porfa": "por favor", "xfa": "por favor", "grx": "gracias", "msj": "mensaje",
    "finde": "fin de semana", "dsp": "después", "dps": "después", "toy": "estoy", "stas": "estás",
    "kita": "quita", "kiero": "quiero", "kambia": "cambia", "ay": "hay", "ta": "está",
    "biblio": "biblioteca", "poli": "polideportivo", "bbdd": "bases de datos", "matri": "matrícula",
    "cumple": "cumpleaños", "entreno": "entrenamiento", "lab": "laboratorio", "profe": "profesor",
}

# Números escritos con letras → cifras. «una» solo en «la una» (si no, «una reunión» → «1 reunión»).
NUMEROS = {
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17,
    "dieciocho": 18, "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintidos": 22, "veintitres": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28,
    "veintinueve": 29, "treinta": 30,
}

# Stopwords de spaCy, salvo las que llevan información para la agenda: negaciones (cambian el sentido),
# interrogativos (consultar/buscar), referencias temporales (fechas relativas) y marcas de completado.
CONSERVAR = {
    "no", "ni", "nunca", "nada", "sin", "ya", "todavía", "aún", "hoy", "mañana", "tarde", "noche",
    "antes", "después", "pasado", "próximo", "próxima", "dentro", "día", "días", "semana", "mes",
    "qué", "cuándo", "dónde", "cómo", "cuál", "quién", "hecho", "primero", "primera",
    "hay", "tengo", "tenía", "era", "muy",
}

_cache = {}


def sin_tildes(t):
    """Quita tildes y diéresis pero conserva la ñ («mañana» no debe volverse «manana»)."""
    t = t.replace("ñ", "\0").replace("Ñ", "\1")
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return t.replace("\0", "ñ").replace("\1", "Ñ")


def tokenizar(t):
    """Palabras (con ñ y tildes), horas tipo «20:30» y signos de puntuación por separado."""
    return re.findall(r"\d+[:.]\d+|\w+|[^\w\s]", t)


def stopwords():
    """Stopwords de spaCy (con y sin tildes) menos las de CONSERVAR y los números con letras («a las cinco»)."""
    if "sw" not in _cache:
        from spacy.lang.es.stop_words import STOP_WORDS
        conservar = CONSERVAR | {sin_tildes(w) for w in CONSERVAR} | set(NUMEROS) | {"una"}
        sw = {w for w in STOP_WORDS if w not in conservar}
        _cache["sw"] = sw | {sin_tildes(w) for w in sw}
    return _cache["sw"]


def _stemmer():
    if "stem" not in _cache:
        from nltk.stem import SnowballStemmer
        _cache["stem"] = SnowballStemmer("spanish")
    return _cache["stem"]


def _nlp():
    if "nlp" not in _cache:
        import spacy
        _cache["nlp"] = spacy.load("es_core_news_md")
    return _cache["nlp"]


def normalizar(texto, minusculas=True, quitar_tildes=False, abreviaturas=False, numeros=False,
               quitar_stopwords=False, raiz=None):
    """Normalización configurable (se usa en E2, C1 y el asistente).

    minusculas      : pasa a minúsculas.
    quitar_tildes   : quita tildes/diéresis (conserva la ñ).
    abreviaturas    : expande abreviaturas de mensajería («q tngo mñn» → «que tengo mañana»).
    numeros         : números con letras → cifras («a las cinco» → «a las 5», «la una» → «la 1»).
    quitar_stopwords: quita stopwords de spaCy salvo las de CONSERVAR (no, ya, hoy, qué, cuándo...).
    raiz            : None, "stem" (Snowball español) o "lema" (lematización de spaCy).
    Si solo se piden minúsculas/tildes se devuelve el texto tal cual; si no, los tokens unidos por espacios.
    """
    t = texto.lower() if minusculas else texto
    if not (abreviaturas or numeros or quitar_stopwords or raiz):
        return sin_tildes(t) if quitar_tildes else t

    toks = tokenizar(t)
    if abreviaturas:
        nuevos = []
        for w in toks:
            exp = ABREVIATURAS.get(sin_tildes(w.lower()))
            if exp is None:
                nuevos.append(w)
            else:
                nuevos.extend((exp.upper() if w.isupper() and len(w) > 1 else exp).split())
        toks = nuevos
    if numeros:
        toks = [str(NUMEROS[sin_tildes(w.lower())]) if sin_tildes(w.lower()) in NUMEROS
                else "1" if w.lower() == "una" and i > 0 and toks[i - 1].lower() == "la"
                else w
                for i, w in enumerate(toks)]
    if raiz == "lema":
        toks = [tok.lemma_.lower() if minusculas else tok.lemma_ for tok in _nlp()(" ".join(toks))]
    if quitar_stopwords:
        sw = stopwords()
        toks = [w for w in toks if w.lower() not in sw]
    if raiz == "stem":
        toks = [_stemmer().stem(w) if w.isalpha() else w for w in toks]
    if quitar_tildes:
        toks = [sin_tildes(w) for w in toks]
    return " ".join(toks)
