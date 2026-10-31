"""Débrief d'une partie rédigé par un LLM local (Ollama), version 5.3.

Principe : le code analyse, le modèle rédige.
  1. `build_facts` calcule les chiffres de la partie (stats, position parmi les mid
     Master, moments clés de la courbe de probabilité).
  2. `build_brief` en tire des phrases toutes faites : jugements (faible / moyenne /
     fort) calculés par des seuils, conseils issus d'une table de règles, moments clés
     décrits avec leurs événements.
  3. Le modèle reçoit ces phrases et ne fait que les reformuler (il n'a rien à
     interpréter, donc rien à inventer).
  4. `check_text` contrôle le résultat : nombres absents, pourcentages mal rattachés,
     moments clés omis, champions cités à la mauvaise minute, minutes écrites en lettres,
     caractères chinois, formulations causales.
  5. En mode hybride, le modèle n'écrit que le résumé, sans connaître aucun événement ; s'il
     échoue au contrôle il reçoit la liste de ses erreurs et réessaie (3 fois), puis on
     utilise le résumé écrit par le code (`generate_summary`).

    python -m src.debrief --limit 3                 # les 3 dernières parties analysées
    python -m src.debrief --match-id EUW1_123456    # une partie précise
    python -m src.debrief --limit 1 --dry-run       # affiche les phrases et le prompt, sans LLM
    python -m src.debrief --limit 3 --no-llm        # débrief sans LLM (gabarit), sert de référence
    python -m src.debrief --recheck results/debriefs  # rejoue le contrôle sur des débriefs déjà écrits, sans LLM
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
from pathlib import Path

import pandas as pd
import requests

from .benchmarks import (
    DEFAULT_BENCH,
    MIN_REFERENCE_ROWS,
    ROLE_NAMES,
    STAT_COLS,
    build_benchmark,
    compare,
    load_benchmarks,
    reference_for,
    role_rows,
)
from .winprob import EARLY_MINUTE, collapse_duplicates, format_gaps

DEFAULT_MODEL = "qwen2.5:7b"
MIN_PROFILE_GAMES = 10
RATING_LOW, RATING_HIGH = 25, 75  # percentiles : <= 25 faible, >= 75 fort

ROLES_FR = {"TOP": "top", "JUNGLE": "jungle", "MIDDLE": "mid", "BOTTOM": "bot (ADC)",
            "UTILITY": "support"}
# nom de la statistique dans benchmarks.py -> (nom dans les faits, facteur, décimales)
STAT_FORMAT = {
    "cs_per_min": ("cs_par_min", 1, 1),
    "vision_per_min": ("vision_par_min", 1, 2),
    "kda": ("kda_ratio", 1, 1),
    "kill_participation": ("participation_kills_pct", 100, 0),
    "damage_share": ("part_degats_pct", 100, 0),
    "cs_at_10": ("cs_a_10_min", 1, 0),
    "gold_at_10": ("or_a_10_min", 1, 0),
    "xp_at_10": ("xp_a_10_min", 1, 0),
}
# nom dans les faits -> (libellé affiché, unité)
LABELS = {
    "cs_par_min": ("CS par minute", ""),
    "vision_par_min": ("Vision par minute", ""),
    "kda_ratio": ("KDA", ""),
    "participation_kills_pct": ("Participation aux kills", " %"),
    "part_degats_pct": ("Part des dégâts de l'équipe", " %"),
    "cs_a_10_min": ("CS à 10 minutes", ""),
    "or_a_10_min": ("Or total gagné à 10 minutes", ""),
    "xp_a_10_min": ("XP à 10 minutes", ""),
}
# Conseils écrits à la main, associés à la statistique la plus basse (pas générés par le modèle)
ADVICE = {
    "cs_par_min": "ne pas laisser les vagues de sbires, surtout pendant les déplacements vers les autres lanes.",
    "vision_par_min": "poser et nettoyer plus de balises, notamment avant les objectifs.",
    "kda_ratio": "réduire les morts évitables.",
    "participation_kills_pct": "rejoindre davantage les combats de l'équipe après avoir poussé la vague.",
    "part_degats_pct": "chercher à infliger plus de dégâts pendant les combats d'équipe.",
    "cs_a_10_min": "en début de partie, privilégier les derniers coups sur les sbires avant les échanges.",
    "or_a_10_min": "soigner la phase de lane : farm et échanges avantageux.",
    "xp_a_10_min": "soigner la phase de lane : rester en position pour récupérer l'expérience des vagues.",
}

# Ce qui change selon le poste. Un support n'a pas à farmer, et sa part des dégâts dépend surtout du champion.
IGNORED_STATS = {"UTILITY": {"cs_par_min", "cs_a_10_min", "part_degats_pct"}}
ADVICE_BY_ROLE = {
    "JUNGLE": {
        "cs_par_min": "enchaîner les camps sans temps mort, en optimisant le chemin dans la jungle.",
        "cs_a_10_min": "soigner le premier tour de jungle pour ne perdre aucun camp.",
        "or_a_10_min": "enchaîner les camps entre les actions, sans attendre une action qui ne vient pas.",
        "xp_a_10_min": "enchaîner les camps entre les actions, sans attendre une action qui ne vient pas.",
        "participation_kills_pct": "se placer près des lanes qui peuvent se battre et près des objectifs.",
    },
    "UTILITY": {
        "or_a_10_min": "soigner la phase de lane : échanges avantageux et quêtes de support terminées tôt.",
        "xp_a_10_min": "rester à portée de l'expérience des vagues, sauf pour un déplacement qui rapporte.",
        "participation_kills_pct": "se déplacer vers les combats quand la lane le permet.",
    },
}

SYSTEM_PROMPT = """Tu es un rédacteur. Tu transformes des FAITS déjà analysés en débrief de \
League of Legends, uniquement en français. Tu ne fais AUCUNE analyse toi-même.

Règles strictes :
- Reprends les phrases des FAITS et reformule-les seulement pour qu'elles soient fluides. \
Garde tous les nombres tels quels, rattachés à la même chose.
- N'ajoute aucune information, aucune cause, aucune explication et aucun conseil qui ne \
soit pas dans les FAITS.
- La probabilité de victoire est celle de l'équipe, jamais celle du joueur.
- Un événement coïncide avec un saut de probabilité : il ne le « cause » pas.
- N'écris une section que si elle existe dans les FAITS. Si « points_forts » ou \
« axes_de_progres » sont absents, n'écris ni points forts ni axes de progrès : recopie \
seulement la phrase « remarque ».
- N'emploie aucun mot anglais. Écris les minutes en chiffres (« 4e minute ») et ne calcule \
rien : pas de différence entre deux pourcentages.

Format : « Résumé » (2 phrases : la partie, puis la courbe), « Moments clés » (une puce par \
moment fourni, dans l'ordre, avec ses nombres), « Points forts » (une puce par phrase \
fournie), « Axes de progrès » (une puce par phrase fournie, avec son conseil), « Tendance \
habituelle » (si elle existe). 18 lignes maximum."""

# Mode hybride : le modèle n'écrit que le résumé, le reste vient du code
SUMMARY_PROMPT = """Tu es un rédacteur. À partir des FAITS, écris le résumé d'une partie de \
League of Legends : 2 ou 3 phrases, en français, sans liste, sans titre, sans mise en forme.

Règles strictes :
- Raconte uniquement l'évolution de la probabilité de victoire de l'ÉQUIPE : le départ, les \
variations aux minutes indiquées, le maximum et le minimum avec leur minute, la valeur à la \
dernière minute de la courbe. Cette probabilité est celle de l'équipe, jamais celle du joueur.
- Les mots « maximum », « pic », « minimum », « le plus bas » et « fin » ne s'emploient que \
pour les valeurs et les minutes que les FAITS désignent ainsi. Une variation citée n'est \
pas un maximum ou un minimum. Ne compare pas les variations entre elles (pas de « plus \
brusque »), et n'écris rien sur ce qui se passe entre deux minutes citées.
- Recopie les nombres tels quels. Écris les minutes en chiffres (« 4e minute »), jamais en \
lettres. Ne calcule rien : pas de différence, pas de moyenne.
- Tu ne connais ni les événements, ni les causes, ni les autres joueurs : n'en mentionne \
aucun et n'écris jamais « grâce à », « suite à » ou « à cause de ».
- Ne donne ni conseil ni jugement sur le joueur. N'emploie aucun mot anglais."""

# formulations qui trahissent une explication inventée
BANNED = ("à cause", "a causé", "ont causé", "grâce à", "achat", "stabilité financière",
          "parce que", "en raison de", "ce qui explique", "ce qui indique", "ce qui suggère",
          "ce qui montre", "suite à", "grâce aux", "dû à", "du fait")
_CJK = re.compile(r"[㐀-鿿]")


MAX_ATTEMPTS = 3
ANGLICISMS = ("takedown",)
# « quatre minutes », « douzième minute », « vingt-deuxne minute » : on veut des chiffres
_SPELLED_MINUTE = re.compile(
    r"\b(?:premi\w+|deux\w*|trois\w*|quatr\w*|quatorz\w*|cinq\w*|six\w*|sept\w*|huit\w*|neuf\w*"
    r"|dix\w*|onz\w*|douz\w*|treiz\w*|quinz\w*|seiz\w*|vingt\w*|trent\w*|quarant\w*)"
    r"(?:-\w+)*\s+minutes?\b", re.IGNORECASE)
_CHAMPION = re.compile(r"(\w+) \((?:alliés|ennemis|bleus|rouges)\)")
_MINUTE_REF = re.compile(r"(?<!\d)(\d{1,2})\s*(?:e|ème|er|re)?\s+minutes?\b|\bminutes?\s+(\d{1,2})\b",
                         re.IGNORECASE)
# un moment de la partie : « 30e minute », « la 30e », « 30 minutes », « minute 30 »
_ORDINAL_MINUTE = re.compile(r"(?<![\d,.])(\d{1,2})(?:\s*(?:e|ème|er|re)\b(?:\s+minutes?)?|\s+minutes?\b)"
                             r"|\bminutes?\s+(\d{1,2})\b", re.IGNORECASE)
# des durées, pas des moments : une statistique (« CS à 10 minutes ») et la longueur de la partie (« en 22 minutes »)
_NOT_A_MOMENT = re.compile(r"\b(?:CS|XP|Or total gagné) à \d+ minutes?\b|\ben \d+ minutes?\b", re.IGNORECASE)
# ce que le résumé du mode hybride a le droit de citer : le reste du brief ne lui est pas montré
SUMMARY_FACTS = ("partie", "courbe", "variations_courbe")


class OllamaError(RuntimeError):
    """Ollama est injoignable ou le modèle est absent."""


# ----------------------------------------------------------------------------- faits
def _round(value: float, decimals: int) -> float | int:
    return int(round(value)) if decimals == 0 else round(float(value), decimals)


def _percentile(value: float, reference: pd.Series) -> int:
    ref = reference.dropna()
    return int(round(100 * (ref < value).mean())) if len(ref) else 0


def _format(col: str, value: float) -> float | int:
    _, factor, decimals = STAT_FORMAT[col]
    return _round(value * factor, decimals)


def _basic_stats(me: dict, minutes: float) -> dict:
    cs = me.get("totalMinionsKilled", 0) + me.get("neutralMinionsKilled", 0)
    return {
        "cs_par_min": _round(cs / minutes, 1),
        "vision_par_min": _round(me.get("visionScore", 0) / minutes, 2),
    }


def build_facts(
    analysis: dict,
    match: dict,
    timeline: dict | None,
    puuid: str,
    bench: pd.DataFrame | None = None,
    profiles: dict[str, tuple[int, pd.DataFrame, str]] | None = None,
) -> dict | None:
    """Chiffres d'une partie du point de vue du joueur. None si le joueur n'y figure pas.

    `profiles` : par poste, (nombre de parties du joueur, sa comparaison à la référence, nom de la référence)."""
    info = match["info"]
    me = next((p for p in info["participants"] if p.get("puuid") == puuid), None)
    if me is None:
        return None
    minutes = max(info.get("gameDuration", 0), 1) / 60
    role = me.get("teamPosition")

    facts: dict = {
        "partie": {
            "champion": me.get("championName"),
            "role": ROLES_FR.get(role, "inconnu"),
            "resultat": analysis["perspective"]["result"],
            "duree_min": int(round(minutes)),
        },
        "poste": role,
        "stats": {"kda": f"{me.get('kills', 0)}/{me.get('deaths', 0)}/{me.get('assists', 0)}"},
    }

    row = next((r for r in role_rows(match, timeline) if r["puuid"] == puuid), None)
    if row is not None:
        for col, (name, _, _) in STAT_FORMAT.items():
            if row.get(col) is not None and not pd.isna(row[col]):
                facts["stats"][name] = _format(col, row[col])
        reference, label = reference_for(bench, role, me.get("championName")) if bench is not None else ([], "")
        if len(reference) >= MIN_REFERENCE_ROWS:
            facts["reference"] = label
            facts["comparaison"] = {
                STAT_FORMAT[col][0]: {
                    "valeur": _format(col, row[col]),
                    "mediane": _format(col, reference[col].median()),
                    "percentile": _percentile(row[col], reference[col]),
                }
                for col in STAT_COLS
                if row.get(col) is not None and not pd.isna(row[col]) and reference[col].notna().any()
            }
    else:
        facts["stats"].update(_basic_stats(me, minutes))

    if profiles and role in profiles:
        n_games, table, label = profiles[role]
        facts["profil_habituel"] = {
            "nb_parties": n_games,
            "poste": ROLE_NAMES[role],
            "reference": label,
            "percentiles": {STAT_FORMAT[r["stat"]][0]: int(r["percentile"])
                            for _, r in table.iterrows() if r["stat"] in STAT_FORMAT},
        }

    probs = [c["win_prob"] for c in analysis["curve"]]
    if probs:
        minutes_curve = [c["minute"] for c in analysis["curve"]]
        pcts = [round(p * 100) for p in probs]
        facts["courbe"] = {
            "probabilite_depart_pct": pcts[0],
            "plus_haut_pct": max(pcts),
            "minute_plus_haut": minutes_curve[probs.index(max(probs))],
            "plus_bas_pct": min(pcts),
            "minute_plus_bas": minutes_curve[probs.index(min(probs))],
            "probabilite_finale_pct": pcts[-1],
            "derniere_minute": minutes_curve[-1],
            # toutes les minutes qui affichent la même valeur arrondie : le texte peut citer l'une d'elles
            "minutes_au_plus_haut": [m for m, v in zip(minutes_curve, pcts, strict=True) if v == max(pcts)],
            "minutes_au_plus_bas": [m for m, v in zip(minutes_curve, pcts, strict=True) if v == min(pcts)],
            "premiere_minute": minutes_curve[0],
        }
    facts["moments_cles"] = [moment_facts(tp) for tp in analysis["turning_points"]]
    return facts


def moment_facts(tp: dict) -> dict:
    """Un moment clé de l'analyse, tel que les phrases l'utilisent."""
    return {
        "minute": tp["minute"],
        "avant_pct": round(tp["win_prob_before"] * 100),
        "apres_pct": round(tp["win_prob_after"] * 100),
        "evenements": collapse_duplicates(tp["events"])[:4],
        "ecarts": tp.get("ecarts", {}),  # absents des analyses produites par une ancienne version
        "contributions": tp.get("contributions", {}),
    }


# ----------------------------------------------------------------------------- phrases
def _ordinal(minute: int) -> str:
    return "1re" if minute == 1 else f"{minute}e"


def _fr(x: float | int) -> str:
    return str(x).replace(".", ",")


def _rating(percentile: int) -> str:
    if percentile <= RATING_LOW:
        return "faible"
    return "fort" if percentile >= RATING_HIGH else "dans la moyenne"


def _stat_sentence(name: str, d: dict, reference: str) -> str:
    label, unit = LABELS[name]
    return (f"{label} : {_fr(d['valeur'])}{unit} (médiane des {reference} : {_fr(d['mediane'])}{unit}), "
            f"mieux que {d['percentile']} % des {reference} : {_rating(d['percentile'])}.")


def _moment_sentence(m: dict) -> str:
    events = (" ; ".join(m["evenements"]) if m["evenements"]
              else "aucun événement majeur (l'écart d'or ou d'XP évolue)")
    sentence = (f"{m['minute']}e minute : la probabilité de victoire de l'équipe passe de "
                f"{m['avant_pct']} % à {m['apres_pct']} %. Événements : {events}.")
    if m.get("ecarts"):
        sentence += f" Écarts sur la minute : {format_gaps(m['ecarts'])}."
    shares = [f"{family} {_fr(points)} points" for family, points in m.get("contributions", {}).items() if points]
    if shares:
        sentence += f" Part du saut : {', '.join(shares)}."
    if m["minute"] < EARLY_MINUTE:
        sentence += " Début de partie : estimation peu fiable."
    return sentence


def build_brief(facts: dict) -> dict:
    """Phrases toutes faites que le modèle doit seulement reformuler."""
    g = facts["partie"]
    brief: dict = {
        "partie": (f"{g['champion']} ({g['role']}), {g['resultat']} en {g['duree_min']} minutes, "
                   f"score {facts['stats']['kda']} (kills/morts/assistances)."),
    }
    c = facts.get("courbe")
    if c:
        brief["courbe"] = (
            f"La probabilité de victoire de l'équipe est partie de {c['probabilite_depart_pct']} % "
            f"et a varié entre {c['plus_bas_pct']} % et {c['plus_haut_pct']} %. "
            f"Son maximum ({c['plus_haut_pct']} %) est à la {_ordinal(c['minute_plus_haut'])} minute, "
            f"son minimum ({c['plus_bas_pct']} %) à la {_ordinal(c['minute_plus_bas'])} minute, "
            f"et elle est à {c['probabilite_finale_pct']} % à la dernière minute de la courbe "
            f"({_ordinal(c['derniere_minute'])}).")
        # valeurs de référence pour le contrôle ; jamais montrées au modèle (clé commençant par « _ »)
        brief["_courbe"] = {k: c[k] for k in (
            "probabilite_depart_pct", "plus_haut_pct", "plus_bas_pct", "probabilite_finale_pct",
            "derniere_minute", "premiere_minute", "minutes_au_plus_haut", "minutes_au_plus_bas")}
    brief["moments_cles"] = [_moment_sentence(m) for m in facts["moments_cles"]]
    # la forme de la courbe sans aucun événement : c'est tout ce que le modèle voit en mode hybride
    brief["variations_courbe"] = [
        f"{m['minute']}e minute : la probabilité de victoire de l'équipe passe de "
        f"{m['avant_pct']} % à {m['apres_pct']} %." for m in facts["moments_cles"]]

    role = facts.get("poste")
    ignored = IGNORED_STATS.get(role, set())
    advice = {**ADVICE, **ADVICE_BY_ROLE.get(role, {})}
    reference = facts.get("reference", "")
    comp = {k: v for k, v in facts.get("comparaison", {}).items() if k in LABELS and k not in ignored}
    if comp:
        ranked = sorted(comp.items(), key=lambda kv: kv[1]["percentile"])
        strong = [(k, v) for k, v in ranked[::-1] if _rating(v["percentile"]) == "fort"][:2]
        if strong:
            brief["points_forts"] = [_stat_sentence(k, v, reference) for k, v in strong]
        else:  # on n'appelle pas « point fort » une statistique simplement dans la moyenne
            k, v = ranked[-1]
            brief["points_forts"] = [f"Aucune statistique ne se démarque nettement au-dessus des {reference}. "
                                     "La plus haute : " + _stat_sentence(k, v, reference)]
        axes = []
        for k, v in ranked[:2]:
            sentence = _stat_sentence(k, v, reference)
            if _rating(v["percentile"]) != "faible":
                sentence = sentence[:-1] + " (parmi les plus basses de cette partie, sans être faible)."
            axes.append(f"{sentence} Conseil : {advice[k]}")
        brief["axes_de_progres"] = axes
    else:
        brief["remarque"] = "Pas de repères de comparaison disponibles pour cette partie."
        basics = {k: v for k, v in facts["stats"].items() if k in ("cs_par_min", "vision_par_min")}
        if basics:
            brief["stats"] = [f"{LABELS[k][0]} : {_fr(v)}." for k, v in basics.items()]

    habit = facts.get("profil_habituel")
    if habit:
        kept = [(k, p) for k, p in habit["percentiles"].items() if k in LABELS and k not in ignored]
        low = sorted(kept, key=lambda kv: kv[1])[:2]
        if low:
            brief["tendance_habituelle"] = [
                f"Sur vos {habit['nb_parties']} parties {habit['poste']}, {LABELS[k][0]} : "
                f"mieux que {p} % des {habit['reference']} ({_rating(p)})." for k, p in low
            ]
    return brief


def render_template(brief: dict, summary: str | None = None) -> str:
    """Les phrases du brief mises en page. Avec `summary`, le résumé est celui du modèle (mode
    hybride) ; sans, c'est le débrief sans LLM, qui sert de référence."""
    summary = summary or " ".join(brief[k] for k in ("partie", "courbe") if k in brief)
    parts = ["**Résumé**\n" + summary]
    parts.append("**Moments clés**\n" + "\n".join(f"- {s}" for s in brief["moments_cles"]))
    for key, title in (("points_forts", "Points forts"), ("axes_de_progres", "Axes de progrès"),
                       ("tendance_habituelle", "Tendance habituelle")):
        if key in brief:
            parts.append(f"**{title}**\n" + "\n".join(f"- {s}" for s in brief[key]))
    if "remarque" in brief:
        parts.append(brief["remarque"])
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- LLM
def build_messages(brief: dict) -> list[dict]:
    data = json.dumps({k: v for k, v in brief.items()
                       if k != "variations_courbe" and not k.startswith("_")},
                      ensure_ascii=False, indent=1)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"FAITS :\n```json\n{data}\n```\n\nRédige le débrief."},
    ]


def build_summary_messages(brief: dict) -> list[dict]:
    """Mode hybride : le modèle ne voit que la partie et la forme de la courbe, sans événement
    ni nom de champion : il ne peut donc se tromper ni de cause ni de joueur."""
    subset = {k: brief[k] for k in SUMMARY_FACTS if k in brief}
    data = json.dumps(subset, ensure_ascii=False, indent=1)
    return [
        {"role": "system", "content": SUMMARY_PROMPT},
        {"role": "user", "content": f"FAITS :\n```json\n{data}\n```\n\nÉcris le résumé."},
    ]


def chat(messages: list[dict], model: str = DEFAULT_MODEL, host: str | None = None,
         timeout: int = 600, seed: int = 42) -> str:
    host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    host = host.replace("0.0.0.0", "localhost")
    if not host.startswith("http"):
        host = "http://" + host
    session = requests.Session()
    session.trust_env = False  # Ollama est local : on ignore les proxys éventuels
    payload = {
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 4096, "seed": seed},
    }
    try:
        r = session.post(f"{host}/api/chat", json=payload, timeout=timeout)
    except requests.ConnectionError as e:
        raise OllamaError(
            f"Ollama ne répond pas sur {host}. Lancez `ollama serve` dans un autre terminal."
        ) from e
    except requests.Timeout as e:
        raise OllamaError(f"Ollama n'a pas répondu en {timeout} s : le modèle est peut-être trop lent ici.") from e
    if r.status_code == 404:
        raise OllamaError(f"Modèle introuvable : lancez `ollama pull {model}`.")
    if not r.ok:
        raise OllamaError(f"Ollama a renvoyé l'erreur {r.status_code}.")
    return r.json()["message"]["content"].strip()


# ----------------------------------------------------------------------------- vérification
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_PERCENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")


_TRANSITION = re.compile(r"(\d+(?:[.,]\d+)?)\s*%\s*(?:à|->|→|vers)\s*(\d+(?:[.,]\d+)?)\s*%")


def _transitions(text: str) -> set[tuple[float, float]]:
    return {(float(a.replace(",", ".")), float(b.replace(",", "."))) for a, b in _TRANSITION.findall(text)}


def _numbers(text: str) -> list[float]:
    return [float(x.replace(",", ".")) for x in _NUMBER.findall(text)]


def _percents(text: str) -> set[float]:
    return {float(x.replace(",", ".")) for x in _PERCENT.findall(text)}


def _strip_list_numbers(text: str) -> str:
    return re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)  # numérotation éventuelle des listes


def _public_json(brief: dict) -> str:
    """Le brief tel que le modèle le voit : sans les valeurs de contrôle (clés « _... »)."""
    return json.dumps({k: v for k, v in brief.items() if not k.startswith("_")}, ensure_ascii=False)


def check_numbers(text: str, facts: dict) -> list[float]:
    """Nombres du texte absents des faits (à un arrondi près). Liste vide = rien d'inventé."""
    allowed = _numbers(_public_json(facts))
    unknown = []
    for x in _numbers(_strip_list_numbers(text)):
        if not any(abs(x - a) < 0.051 or (x == int(x) and abs(x - a) <= 0.5) for a in allowed):
            unknown.append(x)
    return sorted(set(unknown))


def _fmt_number(x: float) -> str:
    return str(int(x)) if x == int(x) else str(x)


def check_text(text: str, brief: dict, *, require_moments: bool = True,
               strict_names: bool = False, visible: dict | None = None) -> list[str]:
    """Problèmes détectés automatiquement. Liste vide = rien à signaler (pas une preuve de justesse).

    `visible` : les faits montrés au modèle, si ce n'est pas tout le brief. Un nombre présent ailleurs dans le
    brief, qu'il n'a jamais vu, ne peut venir que d'une invention tombée juste."""
    issues: list[str] = []
    shown = brief if visible is None else visible
    unknown = check_numbers(text, shown)
    if unknown:
        issues.append("nombres absents des faits : " + ", ".join(_fmt_number(x) for x in unknown))

    # un nombre qui existe dans les faits, mais pas comme pourcentage, ne peut pas en devenir un
    allowed_pct = _percents(_public_json(shown))
    misplaced = sorted(p for p in _percents(_strip_list_numbers(text))
                       if p not in allowed_pct and p not in unknown)
    if misplaced:
        issues.append("pourcentages absents des faits : " + ", ".join(_fmt_number(x) for x in misplaced))

    # « de A % à B % » doit reprendre un saut fourni tel quel, pas mélanger deux sauts
    allowed_pairs = _transitions(_public_json(shown))
    wrong_pairs = sorted(t for t in _transitions(text) if t not in allowed_pairs)
    if wrong_pairs:
        issues.append("transitions absentes des faits : "
                      + ", ".join(f"{_fmt_number(a)} % à {_fmt_number(b)} %" for a, b in wrong_pairs))

    missing = []
    for sentence in (brief.get("moments_cles", []) if require_moments else []):
        minute = re.match(r"(\d+)e minute", sentence)
        # le nombre ne doit pas être collé à d'autres chiffres, mais « 9e » ou « minute 9 » comptent
        if minute and not re.search(rf"(?<!\d){minute.group(1)}(?!\d)", text):
            missing.append(minute.group(1))
    if missing:
        issues.append("moments clés omis (minute " + ", ".join(missing) + ")")

    for key, pattern, title in (("points_forts", r"points? forts?", "Points forts"),
                                ("axes_de_progres", r"axes? de progrès", "Axes de progrès"),
                                ("tendance_habituelle", r"tendance habituelle", "Tendance habituelle")):
        if key not in brief and re.search(pattern, text, re.IGNORECASE):
            issues.append(f"section « {title} » écrite alors que les faits n'en contiennent pas")

    if _CJK.search(text):
        issues.append("caractères non français (chinois) détectés")
    found = [w for w in BANNED if w in text.lower()]
    if found:
        issues.append("formulation causale ou hors faits : " + ", ".join(found))

    spelled = sorted({m.group(0).lower() for m in _SPELLED_MINUTE.finditer(text)})
    if spelled:
        issues.append("minutes écrites en lettres (écrire 4e minute) : " + ", ".join(spelled))
    english = [w for w in ANGLICISMS if w in text.lower()]
    if english:
        issues.append("mot anglais : " + ", ".join(english))
    issues.extend(check_names(text, brief, strict_names))
    issues.extend(check_claims(text, brief))
    return issues


# ----------------------------------------------------------------------------- affirmations
# Le contrôle des nombres ne voit pas qu'un « pic à 92 % » est faux si 92 % existe ailleurs dans
# les faits. Ici on vérifie les mots « maximum / minimum / fin » contre les vraies valeurs.
_MAX_WORDS = re.compile(r"\b(?:maximum|pic|(?:le|son|sa)\s+plus\s+haut|plus haut niveau|sommet|culmin\w*"
                        r"|apogée|point haut)\b", re.I)
_MIN_WORDS = re.compile(r"\b(?:minimum|(?:le|son|sa)\s+plus\s+bas|plus bas niveau|point bas|creux)\b",
                        re.I)
_END_WORDS = re.compile(r"\b(?:termin\w*|finit|finir|fini|finalement|à la fin)\b", re.I)
_UNVERIFIABLE = ("jusqu'à la fin", "jusqu’à la fin", "restant bas", "restant haut", "reste bas",
                 "reste haut", "plus brusque", "plus brutal", "plus fortement", "plus rapidement",
                 "plus lentement", "plus nettement", "répétition", "maintenu", "maintenait",
                 "maintient", "oscill",
                 # relecture de la v5.2 : le résumé ne connaît que quelques minutes de la courbe,
                 # il ne peut rien affirmer sur ce qui se passe entre deux d'entre elles
                 "reste à", "resté à", "restée à", "restant à", "rester à", "restait à", "stabilis", "demeur")
_UNTIL_MINUTE = re.compile(r"jusqu['’]à la \d+\w*", re.IGNORECASE)
_CLAUSE_SPLIT = re.compile(r"[,;]|\s(?:puis|avant de|avant|mais)\s")
_PCT_POS = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")


def _moment_jumps(brief: dict) -> dict[int, tuple[int, int]]:
    """Pour chaque moment clé fourni : minute -> (valeur avant le saut, valeur après)."""
    jumps = {}
    for sentence in brief.get("variations_courbe", []) + brief.get("moments_cles", []):
        m = re.match(r"(\d+)e minute : .*? de (\d+) % à (\d+) %", sentence)
        if m:
            jumps[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))
    return jumps


def _known_values(jumps: dict[int, tuple[int, int]], ref: dict) -> set[tuple[int, int]]:
    """(minute, pourcentage) que les faits établissent : un saut à la minute N va de la valeur de la minute
    N-1 à celle de la minute N, et s'y ajoutent le départ, la fin, le maximum et le minimum."""
    known = {(ref["premiere_minute"], ref["probabilite_depart_pct"]),
             (ref["derniere_minute"], ref["probabilite_finale_pct"])}
    known |= {(m, ref["plus_haut_pct"]) for m in ref["minutes_au_plus_haut"]}
    known |= {(m, ref["plus_bas_pct"]) for m in ref["minutes_au_plus_bas"]}
    for minute, (before, after) in jumps.items():
        known |= {(minute - 1, before), (minute, after)}
    return known


def _nearest(sentence: str, pos: int, matches: list[tuple[int, float]], reach: int = 70):
    """Valeur la plus proche de `pos` : d'abord celles qui suivent, sinon celles qui précèdent."""
    after = [(a - pos, v) for a, v in matches if a >= pos and a - pos <= reach]
    if after:
        return min(after)[1]
    before = [(pos - a, v) for a, v in matches if a < pos and pos - a <= reach]
    return min(before)[1] if before else None


def _clause_minute(sentence: str, pos: int) -> int | None:
    """Minute citée dans la même proposition que le mot en `pos`. À défaut, celle de la proposition
    précédente, si elle ne contient qu'une minute (« À la 15e minute, elle atteint son pic... »)."""
    start = 0
    previous = ""
    for sep in _CLAUSE_SPLIT.finditer(sentence):
        if sep.start() >= pos:
            break
        previous, start = sentence[start:sep.start()], sep.end()
    end = _CLAUSE_SPLIT.search(sentence, pos)
    clause = sentence[start:end.start() if end else len(sentence)]
    found = _MINUTE_REF.findall(clause)
    if not found and not _PCT_POS.search(previous) and re.match(r"\s*(?:à|en)\s+la\s+\d", previous, re.I):
        found = _MINUTE_REF.findall(previous)
    if not found:
        return None
    a, b = found[0]
    return int(a or b)


def check_claims(text: str, brief: dict) -> list[str]:
    ref = brief.get("_courbe")
    issues: list[str] = []
    lowered = text.lower()
    issues += [f"affirmation non vérifiable : « {w} »" for w in _UNVERIFIABLE if w in lowered]
    issues += [f"affirmation non vérifiable : « {m.group(0)} »" for m in _UNTIL_MINUTE.finditer(text)]
    if not ref:
        return issues
    jumps = _moment_jumps(brief)
    known = _known_values(jumps, ref)
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        percents = [(m.start(), float(m.group(1).replace(",", "."))) for m in _PCT_POS.finditer(sentence)]
        flagged = False
        for words, label, agree, value_key, minutes_key in (
                (_MAX_WORDS, "maximum", "annoncé", "plus_haut_pct", "minutes_au_plus_haut"),
                (_MIN_WORDS, "minimum", "annoncé", "plus_bas_pct", "minutes_au_plus_bas"),
                (_END_WORDS, "valeur finale", "annoncée", "probabilite_finale_pct", None)):
            pronoun = "il" if label != "valeur finale" else "elle"
            for w in words.finditer(sentence):
                pct = _nearest(sentence, w.end(), percents)
                if pct is not None and abs(pct - ref[value_key]) > 0.5:
                    issues.append(f"{label} {agree} à {_fmt_number(pct)} % alors qu'{pronoun} est à "
                                  f"{ref[value_key]} %")
                    flagged = True
                if minutes_key:
                    minute = _clause_minute(sentence, w.start())
                    if minute is not None and minute not in ref[minutes_key]:
                        issues.append(f"{label} situé à la {minute}e minute alors qu'il est à la "
                                      f"{ref[minutes_key][0]}e")
                        flagged = True
        if flagged:
            continue
        # « de A % à B % » doit être cité à la minute du saut
        for t in _TRANSITION.finditer(sentence):
            minute = _clause_minute(sentence, t.start())
            pair = (round(float(t.group(1).replace(",", "."))), round(float(t.group(2).replace(",", "."))))
            actual = [m for m, jump in jumps.items() if jump == pair]
            if minute is not None and actual and minute not in actual:
                issues.append(f"transition de {pair[0]} % à {pair[1]} % citée à la {minute}e minute : "
                              f"les faits la placent à la {actual[0]}e")
        # une valeur associée à une minute doit être celle que les faits y placent (ex. « 0 % à la 15e
        # minute ») : pour chaque minute citée, le pourcentage le plus proche dans la même proposition
        for clause in _CLAUSE_SPLIT.split(sentence):
            clause = _NOT_A_MOMENT.sub("", _TRANSITION.sub("", clause))
            c_pcts = [(m.start(), float(m.group(1).replace(",", "."))) for m in _PCT_POS.finditer(clause)]
            for m in _ORDINAL_MINUTE.finditer(clause):
                minute = int(m.group(1) or m.group(2))
                near = [(abs(a - m.start()), v) for a, v in c_pcts if abs(a - m.start()) <= 40]
                if near and (minute, round(min(near)[1])) not in known:
                    value = min(near)[1]
                    extreme = "extrême " if round(value) in (ref["plus_bas_pct"], ref["plus_haut_pct"]) else ""
                    issues.append(f"{_fmt_number(value)} % associé à la {minute}e minute : "
                                  f"valeur {extreme}que les faits ne placent pas là")
    # une même minute ne peut pas porter deux valeurs (« 30 % à la 7e minute, 22 % à la 7e minute »)
    seen: dict[int, set[float]] = {}
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    for clause in (c for sentence in sentences for c in _CLAUSE_SPLIT.split(sentence)):
        # un saut décrit deux minutes : il ne contredit pas une valeur citée
        clause = _NOT_A_MOMENT.sub("", _TRANSITION.sub("", clause))
        c_pcts = [(m.start(), float(m.group(1).replace(",", "."))) for m in _PCT_POS.finditer(clause)]
        for m in _ORDINAL_MINUTE.finditer(clause):
            near = [(abs(a - m.start()), v) for a, v in c_pcts if abs(a - m.start()) <= 40]
            if near:
                seen.setdefault(int(m.group(1) or m.group(2)), set()).add(min(near)[1])
    for minute, values in sorted(seen.items()):
        if len(values) > 1:
            issues.append(f"la {minute}e minute est citée avec deux valeurs : "
                          + " et ".join(f"{_fmt_number(v)} %" for v in sorted(values, reverse=True)))
    return list(dict.fromkeys(issues))


def _moment_names(brief: dict) -> dict[int, set[str]]:
    """Champions cités dans les événements de chaque moment clé fourni."""
    out: dict[int, set[str]] = {}
    for sentence in brief.get("moments_cles", []):
        m = re.match(r"(\d+)e minute", sentence)
        if m:
            out[int(m.group(1))] = set(_CHAMPION.findall(sentence))
    return out


def check_names(text: str, brief: dict, strict: bool = False) -> list[str]:
    """Un champion cité dans une phrase qui parle de la minute N doit figurer dans les événements
    de la minute N. Avec `strict` (résumé du mode hybride, où aucun événement n'est fourni), seul
    le champion du joueur peut être cité. Ne vérifie PAS qui a tué qui."""
    by_minute = _moment_names(brief)
    player_match = re.match(r"(\w+) \(", brief.get("partie", ""))
    player = player_match.group(1) if player_match else None
    known = set().union(*by_minute.values()) - {player} if by_minute else set()
    wrong: dict[str, set[int | None]] = {}
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        cited = {n for n in known if re.search(rf"\b{re.escape(n)}\b", sentence)}
        if not cited:
            continue
        minutes = {int(a or b) for a, b in _MINUTE_REF.findall(sentence)} & by_minute.keys()
        allowed = set().union(*(by_minute[m] for m in minutes)) if minutes else (
            set() if strict else known)
        for name in cited - allowed:
            wrong.setdefault(name, set()).update(minutes or {None})
    issues = []
    for name, minutes in sorted(wrong.items()):
        where = ", ".join(f"{m}e minute" for m in sorted(x for x in minutes if x is not None))
        issues.append(f"champion cité hors de ses événements : {name}" + (f" ({where})" if where else ""))
    return issues


def check_summary(text: str, brief: dict) -> list[str]:
    """Contrôle du résumé du mode hybride : le contrôle général, plus ni liste, ni titre, ni
    section, ni plus de 3 phrases, ni champion autre que celui du joueur."""
    if not text.strip():
        return ["résumé vide"]
    visible = {k: brief[k] for k in SUMMARY_FACTS if k in brief}
    issues = check_text(text, brief, require_moments=False, strict_names=True, visible=visible)
    if re.search(r"(?m)^\s*(?:[-•*]|#|\d+[.)])\s|\*\*", text):
        issues.append("mise en forme interdite (liste, titre ou gras)")
    if re.search(r"points? forts?|axes? de progrès|tendance habituelle", text, re.IGNORECASE):
        issues.append("sections interdites dans le résumé")
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]
    if len(sentences) > 3:
        issues.append(f"résumé trop long ({len(sentences)} phrases au lieu de 3 au plus)")
    return issues


def generate_summary(brief: dict, model: str = DEFAULT_MODEL, host: str | None = None,
                     max_attempts: int = MAX_ATTEMPTS, chat_fn=None) -> tuple[str | None, int, list[str]]:
    """Résumé écrit par le modèle, contrôlé, avec reprise : à chaque échec il reçoit la liste de
    ses erreurs et un autre tirage (graine différente). Renvoie (texte, tentatives, problèmes) ;
    texte vaut None si aucune tentative n'a passé le contrôle (on utilise alors le résumé du code)."""
    chat_fn = chat_fn or chat
    messages = build_summary_messages(brief)
    issues: list[str] = []
    for attempt in range(1, max_attempts + 1):
        reply = chat_fn(messages, model=model, host=host, seed=42 + attempt)
        issues = check_summary(reply, brief)
        if not issues:
            return reply, attempt, []
        messages = messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "Ton résumé contient des problèmes : " + " ; ".join(issues)
             + ". Réécris-le en corrigeant uniquement cela, sans ajouter d'information."},
        ]
    return None, max_attempts, issues


def render_markdown(match_id: str, brief: dict, text: str, issues: list[str], model: str,
                    note: str = "") -> str:
    if issues:
        check = "problèmes détectés : " + " ; ".join(issues)
    else:
        check = "aucun problème détecté (le contrôle ne vérifie pas le sens des phrases)"
    data = json.dumps(brief, ensure_ascii=False, indent=1)
    return (f"# Débrief {match_id}\n\n{text}\n\n---\n"
            f"*Modèle : {model}. Vérification automatique : {check}.{' ' + note if note else ''}*\n\n"
            f"Faits calculés par le code (en mode hybride, le modèle n'en voit qu'une partie) :"
            f"\n\n```json\n{data}\n```\n")


REVIEW_COLUMNS = ["match_id", "modele", "controle_auto", "erreurs_chiffres", "inventions",
                  "omissions", "remarques"]


def read_review(path: Path) -> list[dict]:
    """Lit la fiche de relecture, y compris enregistrée par Excel : BOM, ou « ; » et Windows-1252 en français."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    header = text.split("\n", 1)[0]
    delimiter = ";" if header.count(";") > header.count(",") else ","
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter))


def append_review_rows(path: Path, rows: list[dict]) -> None:
    """Fiche de relecture manuelle (à ouvrir dans Excel) : une ligne par débrief et par mode.

    Relancer un débrief met à jour la colonne `controle_auto` de sa ligne sans toucher à ce que
    vous avez rempli à la main ; une nouvelle partie ou un nouveau mode ajoute une ligne.
    Fermez le fichier dans Excel avant de relancer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        existing = read_review(path)
    index = {(r["match_id"], r["modele"]): r for r in existing}
    for r in rows:
        key = (r["match_id"], r["modele"])
        if key in index:
            index[key]["controle_auto"] = r.get("controle_auto", "")
        else:
            row = {c: r.get(c, "") for c in REVIEW_COLUMNS}
            existing.append(row)
            index[key] = row
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)


def recheck(folder: Path, review: Path | None = None) -> pd.DataFrame:
    """Rejoue le contrôle du résumé sur des débriefs déjà écrits, sans LLM : le fichier contient le résumé et
    les faits. Sert à mesurer un nouveau contrôle sur des résumés déjà relus à la main (colonne `erreurs_relues`).
    Les gabarits, le mode `llm` et les résumés de repli sont ignorés : ce n'est pas un résumé du modèle."""
    annotated: dict[tuple[str, str], int] = {}
    if review is not None and review.exists():
        for row in read_review(review):
            if row["erreurs_chiffres"] != "":
                annotated[row["match_id"], row["modele"]] = sum(
                    int(row[c] or 0) for c in ("erreurs_chiffres", "inventions", "omissions"))
    rows = []
    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        if path.stem.endswith(("_llm", "_gabarit")) or "Résumé du modèle refusé" in text:
            continue
        summary = text.split("**Résumé**\n", 1)[1].split("\n\n**Moments clés**", 1)[0].strip()
        brief = json.loads(text.split("```json\n", 1)[1].split("\n```", 1)[0])
        model = re.search(r"\*Modèle : (.*?)\. Vérification", text).group(1)
        rows.append({"match_id": path.stem, "modele": model, "problemes": " ; ".join(check_summary(summary, brief)),
                     "erreurs_relues": annotated.get((path.stem, model))})
    return pd.DataFrame(rows, columns=["match_id", "modele", "problemes", "erreurs_relues"])


def print_recheck(table: pd.DataFrame) -> None:
    for r in table[table["problemes"] != ""].itertuples():
        print(f"{r.match_id} ({r.modele}) : {r.problemes}")
    flagged = table["problemes"] != ""
    wrong = table["erreurs_relues"].notna() & (table["erreurs_relues"] > 0)
    right = table["erreurs_relues"] == 0
    print(f"\n{len(table)} résumés, {int(flagged.sum())} signalés par le contrôle.")
    print(f"Relus à la main avec une erreur : {int((flagged & wrong).sum())} signalés sur {int(wrong.sum())}.")
    print(f"Relus à la main sans erreur : {int((flagged & right).sum())} signalés sur {int(right.sum())}.")


# ----------------------------------------------------------------------------- CLI
def reviewed_ids(path: Path) -> set[str]:
    """Parties déjà présentes dans la fiche de relecture (quel que soit le mode)."""
    if not path.exists():
        return set()
    return {row["match_id"] for row in read_review(path)}


def _select_ids(curves_dir: Path, match_id: str | None, limit: int,
                exclude: set[str] | None = None) -> list[str]:
    if match_id:
        return [match_id]
    files = sorted(curves_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [f.stem for f in files if f.stem not in (exclude or set())][:limit]


def _written_ids(out_dir: Path, suffix: str, curves_dir: Path) -> set[str]:
    """Parties qui ont déjà un débrief à jour dans ce mode : écrit après la courbe, sinon il la contredirait."""
    others = tuple(s for s in ("_llm", "_gabarit") if s != suffix)
    written = set()
    for path in out_dir.glob(f"*{suffix}.md"):
        match_id = path.stem.removesuffix(suffix)
        curve = curves_dir / f"{match_id}.json"
        if not path.stem.endswith(others) and (not curve.exists() or path.stat().st_mtime >= curve.stat().st_mtime):
            written.add(match_id)
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--raw-dir", default="data/raw/me", help="parties brutes + puuid.txt")
    p.add_argument("--curves", default="results/curves")
    p.add_argument("--bench", default=str(DEFAULT_BENCH))
    p.add_argument("--out", default="results/debriefs")
    p.add_argument("--match-id")
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--only-unreviewed", action="store_true",
                   help="ignore les parties déjà présentes dans review.csv : sert à tester sur des "
                        "parties que vous n'avez jamais relues")
    p.add_argument("--only-missing", action="store_true",
                   help="ignore les parties qui ont déjà un débrief dans ce mode (ne traite que les nouvelles)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--attempts", type=int, default=MAX_ATTEMPTS,
                   help="mode hybride : tentatives du modèle avant le résumé de repli (défaut 3)")
    p.add_argument("--dry-run", action="store_true", help="affiche les phrases et le prompt sans appeler Ollama")
    p.add_argument("--mode", choices=["hybrid", "llm", "template"], default="hybrid",
                   help="hybrid : le modèle n'écrit que le résumé (défaut) ; llm : il écrit tout ; "
                        "template : sans modèle")
    p.add_argument("--no-llm", action="store_true", help="raccourci pour --mode template")
    p.add_argument("--recheck", metavar="DOSSIER",
                   help="rejoue le contrôle sur les débriefs déjà écrits de ce dossier, sans LLM, et s'arrête")
    p.add_argument("--review", help="avec --recheck : fiche de relecture (défaut : review.csv du dossier)")
    args = p.parse_args(argv)
    mode = "template" if args.no_llm else args.mode

    if args.recheck:
        folder = Path(args.recheck)
        print_recheck(recheck(folder, Path(args.review) if args.review else folder / "review.csv"))
        return

    raw_dir, curves_dir = Path(args.raw_dir), Path(args.curves)
    puuid_file = raw_dir / "puuid.txt"
    if not puuid_file.exists():
        raise SystemExit(f"{puuid_file} introuvable : lancez d'abord `python -m src.collect --riot-id ...`.")
    puuid = puuid_file.read_text(encoding="utf-8").strip()

    bench = load_benchmarks(args.bench)
    if bench is None:
        print(f"(Pas de {args.bench} : lancez `python -m src.benchmarks` pour comparer vos parties à des repères.)")
    profiles = {}
    if bench is not None:
        mine = build_benchmark(raw_dir)
        mine = mine[mine["puuid"] == puuid] if not mine.empty else mine
        for role, rows in mine.groupby("role") if not mine.empty else ():
            reference, label = reference_for(bench, role)
            if len(rows) >= MIN_PROFILE_GAMES and len(reference) >= MIN_REFERENCE_ROWS:
                profiles[role] = (len(rows), compare(rows, reference), label)

    out_dir = Path(args.out)
    suffix = {"hybrid": "", "llm": "_llm", "template": "_gabarit"}[mode]
    exclude = reviewed_ids(out_dir / "review.csv") if args.only_unreviewed else set()
    if args.only_missing:
        exclude |= _written_ids(out_dir, suffix, curves_dir)
    ids = _select_ids(curves_dir, args.match_id, args.limit, exclude)
    if not ids:
        if args.only_missing and any(curves_dir.glob("*.json")):
            print("Aucun nouveau débrief à écrire.")
            return
        raise SystemExit(f"Aucune analyse dans {curves_dir} : lancez `python -m src.winprob analyze`.")

    model_name = {"template": "gabarit (sans LLM)", "hybrid": f"{args.model} (hybride v5.3)",
                  "llm": f"{args.model} (complet)"}[mode]
    review_rows = []
    for match_id in ids:
        analysis_path = curves_dir / f"{match_id}.json"
        match_path = raw_dir / "matches" / f"{match_id}.json"
        if not analysis_path.exists() or not match_path.exists():
            print(f"{match_id} : analyse ou partie introuvable, ignorée.")
            continue
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        match = json.loads(match_path.read_text(encoding="utf-8"))
        tl_path = raw_dir / "timelines" / f"{match_id}.json"
        timeline = json.loads(tl_path.read_text(encoding="utf-8")) if tl_path.exists() else None

        facts = build_facts(analysis, match, timeline, puuid, bench, profiles)
        if facts is None:
            print(f"{match_id} : vous n'apparaissez pas dans cette partie, ignorée.")
            continue
        brief = build_brief(facts)

        messages = build_summary_messages(brief) if mode == "hybrid" else build_messages(brief)
        if args.dry_run:
            print(f"\n===== {match_id} =====\n{messages[1]['content']}")
            continue

        note, control = "", ""
        try:
            if mode == "template":
                text = render_template(brief)
                issues = check_text(text, brief)
            elif mode == "hybrid":
                summary, attempts, summary_issues = generate_summary(
                    brief, args.model, max_attempts=args.attempts)
                issues = []
                if summary is None:
                    text = render_template(brief)
                    note = (f"Résumé du modèle refusé après {attempts} tentatives "
                            f"({' ; '.join(summary_issues)}) : résumé de repli écrit par le code.")
                    control = "repli gabarit : " + " ; ".join(summary_issues)
                else:
                    text = render_template(brief, summary)
                    note = f"Résumé du modèle accepté à la tentative {attempts} sur {args.attempts}."
                    control = "ok" if attempts == 1 else f"ok (tentative {attempts}/{args.attempts})"
            else:
                text = chat(messages, args.model)
                issues = check_text(text, brief)
        except OllamaError as e:
            raise SystemExit(str(e)) from e
        control = control or ("; ".join(issues) or "ok")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{match_id}{suffix}.md").write_text(
            render_markdown(match_id, brief, text, issues, model_name, note), encoding="utf-8")
        review_rows.append({"match_id": match_id, "modele": model_name, "controle_auto": control})
        print(f"\n===== {match_id} : {facts['partie']['champion']}, {facts['partie']['resultat']} =====")
        print(text)
        if note:
            print(f"\n[{'!' if control.startswith('repli') else 'ok'}] {note}")
        if issues:
            print("\n[!] " + "\n[!] ".join(issues))
        elif not note:
            print("\n[ok] Aucun problème détecté par le contrôle automatique.")
    if not args.dry_run:
        if review_rows:
            append_review_rows(out_dir / "review.csv", review_rows)
        print(f"\nDébriefs écrits dans {out_dir}/ (fiche de relecture : {out_dir / 'review.csv'})")


if __name__ == "__main__":
    main()
