"""Berya : seule source de vérité pour les modèles, tailles, k, chemins et backend."""
import os

from dotenv import load_dotenv

load_dotenv()  # .env lu ici pour que BERYA_BACKEND soit vu par tous les scripts, quel que soit l'ordre d'import

BACKEND = os.getenv("BERYA_BACKEND", "local")    # local | claude
DEVICE  = os.getenv("BERYA_DEVICE", "mps")       # mps sur le M3 Pro ; cpu ou cuda ailleurs

EMBED_MODEL      = "BAAI/bge-m3"                 # 1024 dims ; en changer oblige à réindexer
EMBED_DIM        = 1024
RERANK_MODEL     = "BAAI/bge-reranker-v2-m3"

# Cas A : Ollama
LOCAL_GEN_MODEL    = "mistral-small3.2"
LOCAL_UTIL_MODEL   = "qwen3:8b"
LOCAL_GEN_NUM_CTX  = 16384
LOCAL_UTIL_NUM_CTX = 8192

# Cas B : Claude. Identifiants de la page Models au 30/10/2026 (claude-sonnet-4-5 est retiré le 30/11/2026).
# Pas de temperature : l'API (version 2026-10-21) la refuse sur tous les modèles.
# Réflexion coupée sur gen et util, comme reasoning=False côté Ollama, pour comparer A et B à réglage égal.
CLAUDE_GEN_MODEL     = "claude-sonnet-5-5"
CLAUDE_UTIL_MODEL    = "claude-haiku-5-5"
JUDGE_MODEL          = "claude-haiku-5-5"
CLAUDE_GEN_THINKING  = {"type": "between_tools"}   # Sonnet 5.5 refuse "disabled" ; between_tools = pas de réflexion
CLAUDE_UTIL_THINKING = {"type": "disabled"}        # Haiku 5.5 : garde le tool calling forcé de with_structured_output
GEN_MAX_TOKENS       = 1500
UTIL_MAX_TOKENS      = 1024                        # 300 tronquait le champ explanation des sorties structurées
JUDGE_MAX_TOKENS     = 1024

GEN_NAME  = LOCAL_GEN_MODEL  if BACKEND == "local" else CLAUDE_GEN_MODEL
UTIL_NAME = LOCAL_UTIL_MODEL if BACKEND == "local" else CLAUDE_UTIL_MODEL

MAX_ARTICLE_TOKENS = 1024                        # au-delà, l'article est découpé en parties
K_BM25, K_VEC, K_FUSED, K_FINAL = 20, 20, 20, 5
MAX_REWRITES    = 2
PROMPTS_VERSION = "v1"
CACHE_TTL_S     = 7 * 24 * 3600
CORPUS_VERSION  = "code-travail-2025-07-11"
CORPUS_DATE     = "11 juillet 2025"              # forme lisible, reprise dans ANSWER_PROMPT
INDEX_DIR       = "index"


if __name__ == "__main__":
    for k, v in sorted((k, v) for k, v in globals().items() if k.isupper()):
        print(f"{k:22} = {v!r}")
