"""Funciones comunes de la práctica (las del anexo del enunciado). Añadid aquí lo que uséis en varios notebooks."""
import json, os, datetime as dt

HOY = dt.date(2026, 2, 9)          # lunes 9 de febrero de 2026
INTENCIONES = ["crear_evento", "crear_tarea", "consultar", "buscar", "completar", "modificar", "borrar", "otro"]
TIPOS = ["FECHA", "HORA", "LUGAR", "PERSONA", "ASIGNATURA", "PRIORIDAD"]


def cargar(ruta):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(linea) for linea in f if linea.strip()]


def guardar(filas, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        for x in filas:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def anotar(id_, texto, intencion, entidades, ref=None, dificil=False):
    """entidades: lista de (trozo, tipo) o (trozo, tipo, valor). Calcula inicio y fin."""
    ents = []
    for trozo, tipo, *valor in entidades:
        i = texto.index(trozo)
        e = {"tipo": tipo, "texto": trozo, "inicio": i, "fin": i + len(trozo)}
        if valor:
            e["valor"] = valor[0]
        ents.append(e)
    return {"id": id_, "texto": texto, "intencion": intencion, "entidades": ents, "ref": ref, "dificil": dificil}


def comprobar(corpus):
    """Comprueba que inicio/fin de cada entidad corresponden a su texto."""
    for x in corpus:
        for e in x["entidades"]:
            assert x["texto"][e["inicio"]:e["fin"]] == e["texto"], (x["id"], e)
    return True


def particiones():
    """Devuelve (entrenamiento, validación). Usa los ficheros del notebook 1; si aún no existen, divide train.jsonl."""
    if os.path.exists("mi_train.jsonl") and os.path.exists("mi_validacion.jsonl"):
        return cargar("mi_train.jsonl"), cargar("mi_validacion.jsonl")
    from sklearn.model_selection import train_test_split
    corpus = cargar("train.jsonl")
    print("Aviso: aún no existen mi_train.jsonl y mi_validacion.jsonl (notebook 1); uso una división provisional de train.jsonl")
    return train_test_split(corpus, test_size=0.2, random_state=0, stratify=[x["intencion"] for x in corpus])
