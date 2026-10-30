"""Smoke test du bloc Setup : embedding, Ollama gen/util, sorties structurées, Claude, LangSmith.

Chaque vérification tourne même si une autre échoue ; le résumé final dit ce qui reste à régler.
"""
import sys
import time
from typing import Literal

from pydantic import BaseModel

import config as C
from llm import make_llms


class Route(BaseModel):                       # même schéma que graph.Route (graph.py arrive au bloc Graph)
    route: Literal["droit_travail", "hors_perimetre", "avis_juridique"]


ROUTE_CHECKS = {"Quel est le délai de préavis en cas de démission ?": "droit_travail",
                "Mon patron refuse mes congés, est-ce que je peux partir quand même ?": "avis_juridique"}


def check_embedding():
    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(model_name=C.EMBED_MODEL, model_kwargs={"device": C.DEVICE},
                                encode_kwargs={"normalize_embeddings": True})
    dim = len(emb.embed_query("préavis de démission"))
    assert dim == C.EMBED_DIM, f"dimension {dim}, attendu {C.EMBED_DIM}"
    return f"dim {dim}"


def say_ok(llm):
    return repr(llm.invoke("Réponds OK.").content.strip()[:40])   # .content doit être une chaîne : graph.py s'en sert


def check_routes(llm):
    router = llm.with_structured_output(Route)
    got = {q: router.invoke(q).route for q in ROUTE_CHECKS}          # lève si l'objet Route est invalide
    wrong = [f"{exp} → {got[q]}" for q, exp in ROUTE_CHECKS.items() if got[q] != exp]
    return "Route valide" + (f", classement à revoir : {', '.join(wrong)}" if wrong else ", classement attendu")


def check_ollama_memory():
    import ollama                                 # dépendance de langchain-ollama
    norm = lambda name: name if ":" in name else f"{name}:latest"
    loaded = {m.model: m for m in ollama.ps().models}
    sizes = ", ".join(f"{name} {m.size / 1e9:.1f} Go ({m.size_vram / max(m.size, 1):.0%} GPU)"
                      for name, m in loaded.items())
    missing = [m for m in (C.LOCAL_GEN_MODEL, C.LOCAL_UTIL_MODEL) if norm(m) not in loaded]
    assert not missing, f"déchargé : {missing} ; chargés : {sizes or 'aucun'}"
    return sizes


def check_langsmith():
    from langsmith import Client
    next(Client().list_datasets(limit=1), None)   # lève une erreur si la clé LangSmith est invalide
    return "clé valide ; les appels LLM ci-dessus sont tracés dans le projet si LANGSMITH_TRACING=true"


def run(name: str, fn, *args) -> bool:
    t0 = time.perf_counter()
    try:
        detail, ok = fn(*args) or "", True
    except Exception as e:                        # on veut le résumé complet, pas la première erreur
        first_line = str(e).splitlines()[0][:200] if str(e) else ""
        detail, ok = f"{type(e).__name__}: {first_line}", False
    print(f"[{'OK' if ok else 'ÉCHEC':5}] {name:32} {time.perf_counter() - t0:6.1f} s  {detail}", flush=True)
    return ok


def main() -> int:
    print(f"Berya smoke · backend={C.BACKEND} · device={C.DEVICE} · prompts={C.PROMPTS_VERSION}\n")
    local_gen, local_util = make_llms("local")
    _, claude_util = make_llms("claude")
    checks = [
        (f"embedding {C.EMBED_MODEL}", check_embedding),
        (f"Ollama gen {C.LOCAL_GEN_MODEL}", say_ok, local_gen),
        (f"Ollama util {C.LOCAL_UTIL_MODEL}", say_ok, local_util),
        ("Ollama util → Route", check_routes, local_util),
        ("Ollama : gen et util en mémoire", check_ollama_memory),
        (f"Claude util {C.CLAUDE_UTIL_MODEL}", say_ok, claude_util),
        ("Claude util → Route", check_routes, claude_util),
        ("LangSmith", check_langsmith),
    ]
    failed = [c[0] for c in checks if not run(*c)]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} vérifications OK"
          + (f" · à régler : {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
