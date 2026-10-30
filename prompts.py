"""Prompts de Berya, en français, versionnés par config.PROMPTS_VERSION (qui fait partie de la clé de cache).

Toute modification d'un prompt = incrémenter PROMPTS_VERSION. Le jeu d'éval ne sert jamais à écrire un prompt.
Les variables entre accolades sont remplies par .format() dans graph.py, baseline.py et eval.py.
"""
import config as C

# ---------- Graph : LLM util ----------

ROUTE_PROMPT = """Tu tries les questions posées à un outil de recherche dans le Code du travail, utilisé par un cabinet d'avocats.
Classe la question dans une seule catégorie :

- droit_travail : question de recherche sur ce que dit le Code du travail (règles, conditions, délais, procédures, \
contenu d'un article), même si elle est formulée sur un exemple général (« un salarié », « l'employeur »).
- avis_juridique : la personne décrit sa propre situation ou un cas précis et demande quoi faire, si elle a raison \
ou quelle décision prendre.
- hors_perimetre : tout le reste (autre branche du droit, sujet non juridique, bavardage).

Exemples :
- « Quelles sont les causes réelles et sérieuses de licenciement ? » → droit_travail
- « Quelle est la durée de la période d'essai d'un cadre en CDI ? » → droit_travail
- « Que prévoit l'article L1234-9 ? » → droit_travail
- « Mon employeur me licencie, que faire ? » → avis_juridique
- « Mon salarié a encore oublié de badger, est-ce que je peux le sanctionner ? » → avis_juridique
- « Dois-je signer l'avenant que mon patron me propose ? » → avis_juridique
- « Combien faut-il d'associés pour créer une SAS ? » → hors_perimetre
- « Quelles sont les conditions d'un divorce par consentement mutuel ? » → hors_perimetre
- « Tu peux me raconter une blague ? » → hors_perimetre

Question : {q}"""

GRADE_PROMPT = """Tu évalues si un article du Code du travail est utile pour répondre à une question.
Réponds relevant = true si l'article contient une règle, une condition, un délai ou une définition qui aide à répondre, \
même partiellement. Réponds relevant = false s'il traite d'un autre sujet ou ne fait que le mentionner en passant.

Question : {q}

Article :
{doc}"""

REWRITE_PROMPT = """La recherche dans le Code du travail n'a rien trouvé de pertinent pour la question ci-dessous.
Reformule-la pour une recherche dans le texte du code : remplace le vocabulaire courant par les termes juridiques \
du Code du travail (par exemple « virer » devient « licencier », « arrêt maladie » devient « suspension du contrat \
de travail pour maladie »), explicite le sujet juridique et retire ce qui ne sert pas à la recherche.
Réponds uniquement par la question reformulée, en une phrase, sans guillemets ni commentaire.

Question : {q}"""

GROUND_PROMPT = """Tu vérifies qu'une réponse ne dit rien qui ne soit dans les articles fournis.
Pour chaque affirmation de la réponse, cherche le passage des articles qui la soutient. Ne tiens pas compte de la \
mention de la version du corpus, de l'avertissement final ni d'une phrase indiquant que les articles ne répondent pas.
Explique en deux ou trois phrases, puis réponds grounded = true seulement si toutes les affirmations sont soutenues.

Articles :
{docs}

Réponse :
{answer}"""

# ---------- Graph et baseline : LLM gen ----------

ANSWER_PROMPT = f"""Tu es un assistant de recherche juridique pour un cabinet d'avocats en droit social. \
Tu réponds à partir des seuls articles du Code du travail fournis ci-dessous, jamais de ta mémoire.

Règles :
1. Cite chaque article utilisé par son numéro complet (« article L1111-1 ») et reprends mot pour mot, entre guillemets « », \
le passage qui fonde chaque point de ta réponse.
2. Ne cite aucun article absent des articles fournis, même si un article fourni y renvoie.
3. Ne donne jamais d'avis sur un cas particulier : tu exposes ce que dit le texte, tu ne conclus pas sur la situation \
de la personne et tu ne lui dis pas quoi faire.
4. Si les articles fournis ne permettent pas de répondre, ou seulement en partie, dis-le explicitement \
(« Les articles retrouvés ne répondent pas à cette question. ») au lieu de compléter.
5. Termine par deux lignes : « Version du corpus : {C.CORPUS_DATE}. » puis « Aide à la recherche, pas un avis juridique. »

Articles :
{{context}}

Question : {{q}}"""

# ---------- Juges des evals (eval.py), un seul modèle pour les trois expériences ----------

CORRECTNESS_PROMPT = """Tu notes la réponse d'un assistant de recherche juridique par rapport à une réponse de référence.
La réponse est correcte si elle contient l'information essentielle de la référence et ne la contredit sur aucun point \
(règle, condition, délai, chiffre). Des précisions supplémentaires exactes ne la rendent pas fausse ; une omission de \
l'essentiel ou une contradiction, si.
Explique en deux ou trois phrases, puis réponds passed = true ou false.

Question : {q}

Réponse de référence : {ref}

Réponse à noter :
{ans}"""

GROUNDED_PROMPT = """Tu vérifies qu'une réponse est entièrement fondée sur les articles du Code du travail fournis.
Chaque affirmation juridique de la réponse doit être soutenue par un passage des articles. Ignore la mention de la \
version du corpus, l'avertissement « pas un avis juridique » et les phrases qui disent que les articles ne répondent pas.
Explique en deux ou trois phrases, puis réponds passed = true si toutes les affirmations sont soutenues, false sinon.

Articles :
{facts}

Réponse à vérifier :
{ans}"""

PROMPTS = {
    "ROUTE_PROMPT": (ROUTE_PROMPT, {"q"}),
    "GRADE_PROMPT": (GRADE_PROMPT, {"q", "doc"}),
    "REWRITE_PROMPT": (REWRITE_PROMPT, {"q"}),
    "GROUND_PROMPT": (GROUND_PROMPT, {"answer", "docs"}),
    "ANSWER_PROMPT": (ANSWER_PROMPT, {"q", "context"}),
    "CORRECTNESS_PROMPT": (CORRECTNESS_PROMPT, {"q", "ref", "ans"}),
    "GROUNDED_PROMPT": (GROUNDED_PROMPT, {"facts", "ans"}),
}


if __name__ == "__main__":
    from string import Formatter

    print(f"Prompts version {C.PROMPTS_VERSION}\n")
    for name, (tpl, expected) in PROMPTS.items():
        found = {f for _, f, _, _ in Formatter().parse(tpl) if f}
        tpl.format(**{k: "…" for k in expected})          # lève une erreur si une accolade est mal échappée
        status = "OK" if found == expected else f"ÉCHEC : attendu {sorted(expected)}, trouvé {sorted(found)}"
        print(f"{name:20} {len(tpl):5} car.  variables {sorted(found)}  {status}")
    print("\n--- ANSWER_PROMPT rempli ---")
    print(ANSWER_PROMPT.format(q="Un salarié en CDD peut-il rompre son contrat avant le terme ?",
                               context="[L1243-1 | en vigueur depuis … | https://…]\nArticle L1243-1 du Code du travail\n…"))
