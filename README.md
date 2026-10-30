# Berya

Assistant de recherche juridique sur le Code du travail : retrieval hybride (BM25 + vecteurs + reranker) et graph
LangGraph (routage, grading, réécriture, contrôle d'ancrage, cache), avec deux backends LLM interchangeables :
**local** (Ollama, cas A) et **claude** (API Anthropic, cas B). Outil de recherche, pas un avis juridique.

Le cahier des charges est [docs/cadrage-berya.md](docs/cadrage-berya.md) ; les règles du repo sont dans [CLAUDE.md](CLAUDE.md).

## Avancement

| Bloc | État |
| --- | --- |
| Setup (`config.py`, `llm.py`, `prompts.py`, `smoke.py`) | Code écrit, `smoke.py` à valider sur le M3 Pro |
| Données (datasets Hugging Face récupérés, révision figée) | Fait |
| Ingestion, Index + fusion, Jeu d'éval, Reranker + mesure, Baseline + LangSmith | À faire (samedi) |
| Graph, Eval agentic | À faire (dimanche) |
| Variante Claude, cas d'étude, vidéo | À faire (semaine 2) |

## Installation (MacBook Pro M3 Pro, 36 Go)

```bash
brew install ollama && brew services start ollama
ollama pull mistral-small3.2 && ollama pull qwen3:8b
uv sync                         # Python 3.14, versions figées par uv.lock
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

## Données

Le corpus n'est pas versionné dans le repo : il est chargé depuis Hugging Face (Apache 2.0) à une révision figée
dans `config.py`, ce qui garantit à tout le monde les mêmes articles, même si l'auteur republie le dataset.

```python
load_dataset(C.CORPUS_REPO, split="train", revision=C.CORPUS_REVISION)
```

Le premier chargement télécharge 10,5 Mo dans `~/.cache/huggingface/` ; ensuite tout tourne hors ligne
(`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`).

| Dataset | Révision HF | Lignes | États | Indexées (`VIGUEUR` + texte) |
| --- | --- | ---: | --- | ---: |
| `louisbrulenaudet/code-travail` | `0677dc295c32354a6b21a6b897c9d09269b61caa` | 9 774 | VIGUEUR 9 732 · ABROGE_DIFF 42 | 9 732 |
| `louisbrulenaudet/code-civil` | `6db84f4ac37e4bc64310d8e25c28ea00168e9176` | 2 893 | VIGUEUR 2 885 · ABROGE_DIFF 8 | non indexé |

- Aucun article `ABROGE` dans ce snapshot : les seuls écartés sont les `ABROGE_DIFF` (abrogation différée).
- Le Code civil est téléchargé pour l'extension multi-codes mais n'est pas indexé : routeur, prompts et jeu d'éval
  sont limités au Code du travail (la question « bail » du jeu d'éval doit rester hors périmètre).
- `fullSectionsTitre` contient des entités HTML (`&gt;` comme séparateur) : à décoder dans `ingest.py`.

## Versions figées

| Élément | Valeur |
| --- | --- |
| Corpus | `louisbrulenaudet/code-travail`, version du 11 juillet 2025 (`CORPUS_VERSION = code-travail-2025-07-11`), révision HF `0677dc295c32354a6b21a6b897c9d09269b61caa` (`CORPUS_REVISION`) |
| Embedding | `BAAI/bge-m3`, 1 024 dimensions (en changer oblige à réindexer) |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| LLM cas A | `mistral-small3.2` (gen), `qwen3:8b` (util), via Ollama |
| LLM cas B | `claude-sonnet-5-5` (gen), `claude-haiku-5-5` (util et juge des evals) |
| Prompts | `v1` |
| Python | 3.14 (`.python-version`, `requires-python >= 3.14`) |
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
- **Python 3.14** au lieu de 3.12 : toutes les dépendances (torch, chromadb, onnxruntime, pyarrow, tokenizers)
  publient des wheels macOS arm64 pour 3.14, aucune compilation depuis les sources.
- **Corpus figé à une révision** (`CORPUS_REPO`, `CORPUS_REVISION` dans `config.py`) ; le cadrage annonce
  « 9 800 articles », le snapshot compte 9 774 lignes dont 9 732 en vigueur.
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
