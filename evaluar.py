"""Evaluación oficial de la práctica «Asistente de agenda».

Uso:
    python evaluar.py ORO.jsonl PREDICCIONES.jsonl

Cada línea de PREDICCIONES.jsonl es un JSON con el mismo "id" que en ORO y los campos que se quieran evaluar:
    {"id": "te001",
     "intencion": "crear_evento",                                    # tarea A
     "entidades": [{"tipo": "FECHA", "texto": "el viernes", "valor": "2026-02-13"}, ...],   # tarea B
     "ref_ranking": ["ag07", "ag15", "ag03"]}                       # tarea C (o "ref": "ag07")

Comparación de entidades (tarea B):
  - FECHA, HORA, ASIGNATURA, PRIORIDAD: por (tipo, valor normalizado).
  - PERSONA, LUGAR: por (tipo, texto) en minúsculas, sin tildes y sin artículos/preposiciones iniciales.
También se puede importar y usar la función evaluar(oro, pred) desde un notebook.
"""
import json, sys, unicodedata, re
from collections import Counter, defaultdict

INTENCIONES = ["crear_evento", "crear_tarea", "consultar", "buscar", "completar", "modificar", "borrar", "otro"]
TIPOS = ["FECHA", "HORA", "LUGAR", "PERSONA", "ASIGNATURA", "PRIORIDAD"]
CON_VALOR = {"FECHA", "HORA", "ASIGNATURA", "PRIORIDAD"}
INICIALES = {"el", "la", "los", "las", "en", "del", "de", "al", "a", "mi", "su"}


def _norm_texto(t):
    t = unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode()
    w = re.findall(r"\w+", t)
    while w and w[0] in INICIALES:
        w = w[1:]
    return " ".join(w)


def _clave(e):
    if e.get("tipo") in CON_VALOR:
        return (e.get("tipo"), str(e.get("valor", "")).strip())
    return (e.get("tipo"), _norm_texto(e.get("texto", "")))


def cargar(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluar(oro, pred, subconjunto=None):
    """oro, pred: listas de dicts. subconjunto: None, "normal" o "dificil". Devuelve un dict de métricas."""
    P = {p["id"]: p for p in pred}
    if subconjunto == "dificil":
        oro = [o for o in oro if o.get("dificil")]
    elif subconjunto == "normal":
        oro = [o for o in oro if not o.get("dificil")]
    res = {"n": len(oro)}

    # ---- Tarea A: intención
    if any("intencion" in p for p in pred):
        y = [o["intencion"] for o in oro]
        yp = [P.get(o["id"], {}).get("intencion") for o in oro]
        res["A_accuracy"] = sum(a == b for a, b in zip(y, yp)) / len(y)
        f1s = {}
        for c in INTENCIONES:
            tp = sum(a == c and b == c for a, b in zip(y, yp))
            fp = sum(a != c and b == c for a, b in zip(y, yp))
            fn = sum(a == c and b != c for a, b in zip(y, yp))
            if tp + fn:
                f1s[c] = _f1(tp, fp, fn)[2]
        res["A_f1_macro"] = sum(f1s.values()) / len(f1s)
        res["A_f1_por_clase"] = f1s

    # ---- Tarea B: entidades
    if any("entidades" in p for p in pred):
        cnt = defaultdict(lambda: [0, 0, 0])
        for o in oro:
            g = Counter(_clave(e) for e in o["entidades"])
            s = Counter(_clave(e) for e in P.get(o["id"], {}).get("entidades", []))
            for k in set(g) | set(s):
                tp = min(g[k], s[k])
                cnt[k[0]][0] += tp
                cnt[k[0]][1] += s[k] - tp
                cnt[k[0]][2] += g[k] - tp
        tot = [sum(v[i] for v in cnt.values()) for i in range(3)]
        res["B_f1_micro"] = _f1(*tot)[2]
        res["B_f1_por_tipo"] = {t: _f1(*cnt[t])[2] for t in TIPOS if t in cnt}

    # ---- Tarea C: entrada de la agenda referida
    con_ref = [o for o in oro if o.get("ref")]
    if con_ref and any("ref" in p or "ref_ranking" in p for p in pred):
        top1, rr, top3 = 0, 0.0, 0
        for o in con_ref:
            p = P.get(o["id"], {})
            rank = p.get("ref_ranking") or ([p["ref"]] if p.get("ref") else [])
            if rank[:1] == [o["ref"]]:
                top1 += 1
            if o["ref"] in rank[:3]:
                top3 += 1
            if o["ref"] in rank:
                rr += 1 / (rank.index(o["ref"]) + 1)
        n = len(con_ref)
        res["C_acierto@1"], res["C_acierto@3"], res["C_MRR"] = top1 / n, top3 / n, rr / n
    return res


def imprimir(res, titulo=""):
    if titulo:
        print(f"== {titulo} (n={res['n']})")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k:<14} {v:.3f}")
        elif isinstance(v, dict):
            print(f"  {k:<14} " + "  ".join(f"{a}={b:.2f}" for a, b in v.items()))


if __name__ == "__main__":
    oro, pred = cargar(sys.argv[1]), cargar(sys.argv[2])
    faltan = {o["id"] for o in oro} - {p["id"] for p in pred}
    if faltan:
        print(f"Aviso: faltan {len(faltan)} predicciones (cuentan como fallo)")
    imprimir(evaluar(oro, pred), "Todo el conjunto")
    if any(o.get("dificil") for o in oro):
        imprimir(evaluar(oro, pred, "normal"), "Frases normales")
        imprimir(evaluar(oro, pred, "dificil"), "Frases difíciles")
