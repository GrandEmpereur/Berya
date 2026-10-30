# Berya

RAG juridique sur le Code du travail (dataset Hugging Face louisbrulenaudet/code-travail), qui tourne en local
sur un MacBook Pro M3 Pro 36 Go. Deux backends LLM interchangeables : local (Ollama) et claude (API Anthropic).
Le document de cadrage fait foi : docs/cadrage-berya.md (export Markdown du doc Claude, original : https://claude.ai/artifact/5N7LLQkMcB96qnbq5MnbKp)
(architecture, interfaces, planning, critères de sortie). Ne change pas l'architecture sans me le dire.

## Stack imposée
Python 3.14 + uv ; LangChain 1.x, LangGraph 1.x, LangSmith ; Hugging Face datasets + sentence-transformers ;
Chroma + rank_bm25 ; langchain-ollama (cas A) et langchain-anthropic (cas B). Pas d'autre framework.

## Règles
- config.py est la seule source de vérité : aucun nom de modèle, k, chemin ou seuil en dur ailleurs.
- L'article est l'unité d'indexation. Chaque Document porte : num, article_id, etat, date_debut, date_fin,
  section, url, corpus_version, chunk_id, n_parts.
- baseline(q) et ask(q, use_cache) renvoient le même dict : answer, documents, sources.
- Les prompts vivent dans prompts.py, en français, versionnés par PROMPTS_VERSION ; ANSWER_PROMPT impose
  de citer les articles par numéro, de ne jamais donner d'avis sur un cas, et de dire quand le texte ne répond pas.
- Les evals n'utilisent jamais le cache (use_cache=False). Le jeu d'éval ne sert jamais à écrire un prompt.
- Secrets dans .env (python-dotenv), jamais dans le code ; .env.example tenu à jour.
- Chaque script a un bloc if __name__ == "__main__" qui le fait tourner seul avec une sortie lisible.
- Après chaque étape : lance le script, colle la sortie, liste ce que je dois vérifier à la main.
- Pas de nouvelle dépendance sans une ligne de justification. Pas de téléchargement > 100 Mo sans me demander.
- Commits petits, messages en français, un par bloc du planning.

## Commandes
uv run python smoke.py · uv run python ingest.py · uv run python retriever.py "L1234-9" ·
uv run python graph.py "question" · uv run python eval.py · BERYA_BACKEND=claude uv run python graph.py "question"
