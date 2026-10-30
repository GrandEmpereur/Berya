# Berya

Assistant de recherche juridique sur le Code du travail : retrieval hybride (BM25 + vecteurs + reranker) et graph
LangGraph (routage, grading, réécriture, contrôle d'ancrage, cache), avec deux backends LLM interchangeables :
**local** (Ollama, cas A) et **claude** (API Anthropic, cas B). Outil de recherche, pas un avis juridique.

Le cahier des charges est [docs/cadrage-berya.md](docs/cadrage-berya.md) ; les règles du repo sont dans [CLAUDE.md](CLAUDE.md).

## Avancement

| Bloc | État |
| --- | --- |
| Setup (`config.py`, `llm.py`, `prompts.py`, `smoke.py`) | Code écrit, `smoke.py` à valider sur le M3 Pro |
| Ingestion, Index + fusion, Jeu d'éval, Reranker + mesure, Baseline + LangSmith | À faire (samedi) |
| Graph, Eval agentic | À faire (dimanche) |
| Variante Claude, cas d'étude, vidéo | À faire (semaine 2) |

## Installation (MacBook Pro M3 Pro, 36 Go)

```bash
brew install ollama && brew services start ollama
ollama pull mistral-small3.2 && ollama pull qwen3:8b
uv sync                         # Python 3.12, versions figées par uv.lock
cp .env.example .env            # puis renseigner ANTHROPIC_API_KEY et LANGSMITH_API_KEY
uv run python smoke.py
```

`BERYA_DEVICE` vaut `mps` par défaut ; mettre `cpu` pour tourner hors Mac.

## Commandes

```bash
uv run python smoke.py          # embedding, Ollama gen/util, sorties structurées, Claude, LangSmith
uv run python config.py         # affiche la configuration active
uv run python prompts.py        # vérifie les 7 prompts (variables, échappement)
```

À venir : `ingest.py`, `retriever.py "L1234-9"`, `graph.py "question"`, `eval.py`,
`BERYA_BACKEND=claude uv run python graph.py "question"`.

## Versions figées

| Élément | Valeur |
| --- | --- |
| Corpus | `louisbrulenaudet/code-travail`, version du 11 juillet 2025 (`CORPUS_VERSION = code-travail-2025-07-11`) |
| Embedding | `BAAI/bge-m3`, 1 024 dimensions (en changer oblige à réindexer) |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| LLM cas A | `mistral-small3.2` (gen), `qwen3:8b` (util), via Ollama |
| LLM cas B | `claude-sonnet-5-5` (gen), `claude-haiku-5-5` (util et juge des evals) |
| Prompts | `v1` |
| Bibliothèques | `uv.lock` : langchain 1.4.7, langchain-core 1.7.3, langgraph 1.2.18, langsmith 0.14.16, langchain-anthropic 1.7.7 (SDK anthropic 2.3.0), langchain-ollama 1.1.0, sentence-transformers 6.2.0, chromadb 1.5.9, torch 2.15.0 |

## Écarts assumés par rapport au cadrage

- **Modèles Claude** : `claude-sonnet-5-5` et `claude-haiku-5-5` au lieu de `claude-sonnet-4-5` / `claude-haiku-4-5`
  (le cadrage demande d'aligner sur la page Models ; Sonnet 4.5 est retiré le 30 novembre 2026).
- **Pas de `temperature=0` côté Claude** : le SDK anthropic 2.x envoie la version d'API `2026-10-21`, qui refuse
  `temperature` sur tous les modèles, et langchain-anthropic la rejette déjà sur Sonnet 5.5. Le cas B n'est donc pas
  déterministe ; le cas A garde `temperature=0`.
- **Réflexion coupée côté Claude**, comme `reasoning=False` côté Ollama : `between_tools` pour Sonnet 5.5 (qui refuse
  `disabled`), `disabled` pour Haiku 5.5. Cela garde `.content` en chaîne (utilisé par `graph.py`) et le tool calling
  forcé de `with_structured_output`.
- **`max_tokens` de util porté de 300 à 1 024** : 300 risquait de tronquer le champ `explanation` des sorties
  structurées (`GroundGrade`, juges).
- **`llm.make_judge()`** fournit le juge des evals, pour que `eval.py` n'instancie pas `ChatAnthropic` en dur.
- **`config.py`** ajoute `DEVICE`, `EMBED_DIM`, `CORPUS_DATE`, les `num_ctx` et `max_tokens`, et charge `.env` lui-même.

## Structure

```text
berya/
├── CLAUDE.md          # règles du repo pour Claude Code
├── config.py          # seule source de vérité : modèles, tailles, k, chemins, backend
├── llm.py             # make_llms(backend) : local (Ollama) ou claude ; make_judge()
├── prompts.py         # 5 prompts du graph + 2 prompts de juge, en français, v1
├── smoke.py           # critère de sortie du bloc Setup
├── docs/              # cadrage-berya.md (+ architecture.png à déposer)
├── pyproject.toml, uv.lock, .python-version
└── .env.example
```
