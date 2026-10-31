import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import pytest

from src.benchmarks import STAT_COLS
from src.debrief import (
    OllamaError,
    append_review_rows,
    build_brief,
    build_facts,
    build_messages,
    build_summary_messages,
    chat,
    check_names,
    check_numbers,
    check_summary,
    check_text,
    generate_summary,
    render_markdown,
    render_template,
)


def _player(pid, team, role, **kw):
    base = {
        "participantId": pid, "teamId": team, "teamPosition": role, "puuid": f"p{pid}",
        "championName": "Ahri", "win": team == 100, "kills": 2, "deaths": 1, "assists": 4,
        "totalMinionsKilled": 200, "neutralMinionsKilled": 20, "visionScore": 30,
        "totalDamageDealtToChampions": 10000,
    }
    base.update(kw)
    return base


def _match(mid_role="MIDDLE"):
    players = [_player(i, 100 if i <= 5 else 200, "TOP") for i in range(1, 11)]
    players[2] = _player(3, 100, mid_role, kills=6, deaths=2, assists=6)  # vous : participant 3
    players[7] = _player(8, 200, "MIDDLE")
    return {"metadata": {"matchId": "EUW1_3"},
            "info": {"gameDuration": 1800, "participants": players}}


def _timeline():
    pf = {str(i): {"totalGold": 3500, "xp": 4000, "minionsKilled": 80, "jungleMinionsKilled": 4}
          for i in range(1, 11)}
    return {"info": {"frames": [{"timestamp": m * 60_000, "participantFrames": pf}
                                for m in range(12)]}}


ANALYSIS = {
    "match_id": "EUW1_3",
    "perspective": {"team": "bleue", "champion": "Ahri", "role": "MIDDLE", "result": "victoire"},
    "curve": [{"minute": m, "win_prob": p} for m, p in [(2, 0.5), (9, 0.54), (10, 0.44), (20, 0.8)]],
    "turning_points": [{
        "minute": 9, "win_prob_before": 0.54, "win_prob_after": 0.44,
        "events": ["Les alliés prennent des Larves du Néant"] * 3
                  + ["Les ennemis prennent le Dragon"],
    }],
}


def bench_frame():
    # 100 mid de référence, de 1 à 10 sur chaque statistique : un percentile est facile à prévoir
    return pd.DataFrame({c: np.linspace(1, 10, 100) for c in STAT_COLS})


def mid_brief():
    return build_brief(build_facts(ANALYSIS, _match(), _timeline(), "p3", bench_frame()))


def crafted_brief():
    """Brief avec des percentiles choisis : CS faible, vision forte, KDA moyen."""
    facts = build_facts(ANALYSIS, _match(), _timeline(), "p3", bench_frame())
    facts["comparaison_mid_master"] = {
        "cs_par_min": {"valeur": 6.8, "mediane": 7.8, "percentile": 22},
        "vision_par_min": {"valeur": 0.9, "mediane": 0.7, "percentile": 80},
        "kda_ratio": {"valeur": 2.3, "mediane": 2.3, "percentile": 50},
    }
    return build_brief(facts)


# ----------------------------------------------------------------------------- faits
def test_facts_for_a_mid_game_include_stats_and_percentiles():
    bench = bench_frame()
    facts = build_facts(ANALYSIS, _match(), _timeline(), "p3", bench)
    assert facts["partie"] == {"champion": "Ahri", "role": "mid", "resultat": "victoire",
                               "duree_min": 30}
    assert facts["stats"]["kda"] == "6/2/6"
    assert facts["stats"]["cs_par_min"] == round(220 / 30, 1)
    expected = round(100 * (bench["cs_per_min"] < 220 / 30).mean())
    assert facts["comparaison_mid_master"]["cs_par_min"]["percentile"] == expected
    assert facts["comparaison_mid_master"]["cs_par_min"]["mediane"] == round(bench["cs_per_min"].median(), 1)
    assert facts["stats"]["cs_a_10_min"] == 84


def test_facts_collapse_repeated_events_and_round_probabilities():
    facts = build_facts(ANALYSIS, _match(), _timeline(), "p3", bench_frame())
    moment = facts["moments_cles"][0]
    assert moment["evenements"] == ["Les alliés prennent des Larves du Néant (×3)",
                                    "Les ennemis prennent le Dragon"]
    assert (moment["avant_pct"], moment["apres_pct"]) == (54, 44)
    assert facts["courbe"] == {"probabilite_depart_pct": 50, "plus_haut_pct": 80,
                               "plus_bas_pct": 44}


def test_facts_for_another_role_have_no_mid_comparison():
    facts = build_facts(ANALYSIS, _match(mid_role="TOP"), _timeline(), "p3", bench_frame())
    assert "comparaison_mid_master" not in facts
    assert facts["partie"]["role"] == "top"
    assert facts["stats"]["cs_par_min"] == round(220 / 30, 1)


def test_facts_none_when_player_absent_and_profile_included():
    assert build_facts(ANALYSIS, _match(), _timeline(), "inconnu") is None
    profile = (23, pd.DataFrame({"stat": ["cs_per_min"], "percentile": [16]}))
    facts = build_facts(ANALYSIS, _match(), _timeline(), "p3", profile=profile)
    assert facts["profil_habituel"] == {"nb_parties_mid": 23, "percentiles": {"cs_par_min": 16}}


# ----------------------------------------------------------------------------- phrases
def test_brief_moment_sentence_is_ready_to_copy():
    brief = mid_brief()
    assert brief["moments_cles"] == [
        "9e minute : la probabilité de victoire de l'équipe passe de 54 % à 44 %. "
        "Événements : Les alliés prennent des Larves du Néant (×3) ; Les ennemis prennent le Dragon."
    ]
    assert brief["partie"] == "Ahri (mid), victoire en 30 minutes, score 6/2/6 (kills/morts/assistances)."
    assert brief["courbe"].endswith("a varié entre 44 % et 80 %.")


def test_brief_judgements_come_from_thresholds_and_advice_from_rules():
    brief = crafted_brief()
    assert brief["points_forts"][0] == (
        "Vision par minute : 0,9 (médiane des mid Master : 0,7), mieux que 80 % des mid Master : fort.")
    assert brief["axes_de_progres"][0].startswith(
        "CS par minute : 6,8 (médiane des mid Master : 7,8), mieux que 22 % des mid Master : faible.")
    assert "Conseil : ne pas laisser les vagues de sbires" in brief["axes_de_progres"][0]
    # le deuxième axe est « dans la moyenne » : la phrase le dit au lieu de le présenter comme un défaut
    assert "sans être faible" in brief["axes_de_progres"][1]


def test_brief_without_comparison_has_a_remark_and_no_strengths():
    facts = build_facts(ANALYSIS, _match(mid_role="TOP"), _timeline(), "p3", bench_frame())
    brief = build_brief(facts)
    assert "remarque" in brief and "points_forts" not in brief
    assert brief["stats"][0] == "CS par minute : 7,3."


def test_brief_includes_usual_trend_when_profile_is_given():
    profile = (23, pd.DataFrame({"stat": ["cs_per_min", "vision_per_min"], "percentile": [16, 38]}))
    brief = build_brief(build_facts(ANALYSIS, _match(), _timeline(), "p3", profile=profile))
    assert brief["tendance_habituelle"][0] == (
        "Sur vos 23 parties mid, CS par minute : mieux que 16 % des mid Master (faible).")


def test_prompt_contains_rules_and_brief():
    system, user = build_messages(mid_brief())
    assert "Tu ne fais AUCUNE analyse" in system["content"]
    assert "probabilité de victoire est celle de l'équipe" in system["content"]
    assert "9e minute" in user["content"]


def test_template_debrief_has_all_sections_and_passes_its_own_check():
    brief = mid_brief()
    text = render_template(brief)
    for title in ("**Résumé**", "**Moments clés**", "**Points forts**", "**Axes de progrès**"):
        assert title in text
    assert check_text(text, brief) == []


# ----------------------------------------------------------------------------- vérification
def test_check_numbers_accepts_given_numbers_and_rounding():
    facts = build_facts(ANALYSIS, _match(), _timeline(), "p3")
    text = ("1. À la 9e minute, la probabilité passe de 54 % à 44 %.\n"
            "2. Votre KDA de 6/2/6 et 7 CS par minute.")  # 7 pour 7,3 : arrondi toléré
    assert check_numbers(text, facts) == []


def test_check_numbers_flags_invented_numbers():
    facts = build_facts(ANALYSIS, _match(), _timeline(), "p3")
    assert check_numbers("Vous êtes à 87 % de victoire et 12,5 CS par minute.", facts) == [12.5, 87.0]


def test_check_text_flags_a_real_number_attached_to_the_wrong_thing():
    brief = crafted_brief()
    faithful = ("À la 9e minute, la probabilité passe de 54 % à 44 %. "
                "Vous faites 6,8 CS par minute.")
    assert check_text(faithful, brief) == []
    # 6,8 existe dans les faits (CS par minute), mais jamais comme pourcentage
    wrong = ("À la 9e minute, la probabilité passe de 54 % à 44 %. "
             "Votre participation aux kills est de 6,8 %.")
    assert any("pourcentages absents des faits : 6.8" in i for i in check_text(wrong, brief))


def test_check_text_flags_omitted_moments_chinese_and_causal_wording():
    brief = mid_brief()
    issues = check_text("Belle partie. 你好. Victoire grâce à un bon achat.", brief)
    joined = " | ".join(issues)
    assert "moments clés omis (minute 9)" in joined
    assert "chinois" in joined
    assert "grâce à" in joined and "achat" in joined


def test_render_markdown_mentions_check_result():
    brief = mid_brief()
    ok = render_markdown("EUW1_3", brief, "Texte.", [], "qwen2.5:7b")
    assert "aucun problème détecté" in ok and "# Débrief EUW1_3" in ok
    bad = render_markdown("EUW1_3", brief, "Texte.", ["nombres absents des faits : 87"], "qwen2.5:7b")
    assert "problèmes détectés : nombres absents des faits : 87" in bad


def test_review_sheet_is_appended_without_duplicates(tmp_path):
    path = tmp_path / "review.csv"
    rows = [{"match_id": "EUW1_1", "modele": "qwen2.5:7b", "controle_auto": "ok"}]
    append_review_rows(path, rows)
    append_review_rows(path, rows + [{"match_id": "EUW1_2", "modele": "qwen2.5:7b",
                                      "controle_auto": "moments clés omis (minute 9)"}])
    with path.open(encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f))
    assert [r["match_id"] for r in data] == ["EUW1_1", "EUW1_2"]
    assert data[0]["erreurs_chiffres"] == ""  # colonnes à remplir à la main


# ----------------------------------------------------------------------------- appel Ollama
def _fake_ollama(seen):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append((self.path, json.loads(body)))
            payload = json.dumps({"message": {"content": "  Débrief de test.  "}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_chat_talks_to_ollama_api():
    seen = []
    server = _fake_ollama(seen)
    try:
        host = f"127.0.0.1:{server.server_address[1]}"  # forme sans http://, comme OLLAMA_HOST
        out = chat([{"role": "user", "content": "salut"}], model="qwen2.5:7b", host=host)
    finally:
        server.shutdown()
    assert out == "Débrief de test."
    path, body = seen[0]
    assert path == "/api/chat"
    assert body["model"] == "qwen2.5:7b" and body["stream"] is False
    assert body["options"]["temperature"] == 0.2


def test_chat_explains_when_ollama_is_down():
    with pytest.raises(OllamaError, match="ollama serve"):
        chat([{"role": "user", "content": "x"}], host="http://127.0.0.1:9")


def test_omitted_moment_check_accepts_both_forms_but_not_digits_inside_other_numbers():
    brief = mid_brief()
    assert not any("omis" in i for i in check_text("Moment clé à la 9e minute.", brief))
    assert not any("omis" in i for i in check_text("Moment clé : minute 9.", brief))
    assert any("omis" in i for i in check_text("Un moment à la 19e minute.", brief))


def test_check_text_flags_a_mixed_up_transition_and_inference_markers():
    brief = mid_brief()  # seul saut fourni : 54 % à 44 %
    assert check_text("À la 9e minute, la probabilité passe de 54 % à 44 %.", brief) == []
    # 54 et 44 existent, mais pas ce saut-là
    issues = check_text("À la 9e minute, la probabilité passe de 44 % à 54 %.", brief)
    assert any("transitions absentes des faits : 44 % à 54 %" in i for i in issues)
    issues = check_text("9e minute : 54 % à 44 %, ce qui suggère une erreur de lecture.", brief)
    assert any("ce qui suggère" in i for i in issues)


def test_usual_trend_is_only_given_for_games_played_mid():
    profile = (23, pd.DataFrame({"stat": ["cs_per_min"], "percentile": [16]}))
    jungle = build_facts(ANALYSIS, _match(mid_role="JUNGLE"), _timeline(), "p3", profile=profile)
    assert "profil_habituel" not in jungle
    assert "profil_habituel" in build_facts(ANALYSIS, _match(), _timeline(), "p3", profile=profile)


def test_check_text_flags_sections_written_without_data_and_new_causal_markers():
    brief = build_brief(build_facts(ANALYSIS, _match(mid_role="TOP"), _timeline(), "p3"))
    assert "points_forts" not in brief
    issues = check_text("9e minute : 54 % à 44 %.\nPoints forts : les alliés prennent le Dragon.\n"
                        "Axes de progrès : mieux gérer les tours.", brief)
    joined = " | ".join(issues)
    assert "« Points forts » écrite" in joined and "« Axes de progrès » écrite" in joined
    assert any("suite à" in i for i in check_text("9e minute : 54 % à 44 % suite à un kill.", brief))


def test_hybrid_mode_only_gives_the_summary_to_the_model_and_keeps_code_sections():
    brief = mid_brief()
    system, user = build_summary_messages(brief)
    assert "2 ou 3 phrases" in system["content"]
    # le modèle ne voit que la forme de la courbe : ni événements, ni sections, ni noms
    assert "variations_courbe" in user["content"]
    for hidden in ("moments_cles", "Événements", "Larves", "points_forts", "Conseil"):
        assert hidden not in user["content"]
    text = render_template(brief, "Résumé écrit par le modèle.")
    assert text.startswith("**Résumé**\nRésumé écrit par le modèle.")
    for title in ("**Moments clés**", "**Points forts**", "**Axes de progrès**"):
        assert title in text
    assert brief["moments_cles"][0] in text  # les moments viennent du code, mot pour mot


# ----------------------------------------------------------------------------- v4
def named_brief():
    return {
        "partie": "Syndra (mid), victoire en 29 minutes, score 6/7/13 (kills/morts/assistances).",
        "courbe": "La probabilité de victoire de l'équipe est partie de 42 % et a varié entre 39 % et 99 %.",
        "moments_cles": [
            "12e minute : la probabilité de victoire de l'équipe passe de 69 % à 78 %. Événements : "
            "Les alliés prennent le Dragon ; Udyr (alliés) tue Katarina (ennemis) ; "
            "Malphite (ennemis) tue Hwei (alliés) ; Sivir (alliés) tue Draven (ennemis).",
            "22e minute : la probabilité de victoire de l'équipe passe de 87 % à 96 %. Événements : "
            "Les alliés prennent Nashor ; Sivir (alliés) tue Alistar (ennemis) ; "
            "Hwei (alliés) tue Draven (ennemis).",
        ],
    }


GOOD_SUMMARY = ("Ahri remporte la partie en 30 minutes. La probabilité de victoire de l'équipe part "
                "de 50 % et varie entre 44 % et 80 %. À la 9e minute, elle passe de 54 % à 44 %.")


def test_check_names_flags_a_champion_cited_at_the_wrong_minute():
    brief = named_brief()
    wrong = "À la 12e minute, la probabilité augmente à 78 % avec la mort de Katarina, Draven et Alistar."
    assert check_names(wrong, brief) == ["champion cité hors de ses événements : Alistar (12e minute)"]
    right = "À la 22e minute, la probabilité atteint 96 % avec la mort d'Alistar et de Draven."
    assert check_names(right, brief) == []
    # une phrase sans minute peut citer n'importe quel champion des faits, sauf en mode strict
    free = "Syndra a gagné. Udyr a tué Katarina."
    assert check_names(free, brief) == []
    strict = check_names(free, brief, strict=True)
    assert strict == ["champion cité hors de ses événements : Katarina",
                      "champion cité hors de ses événements : Udyr"]


def test_check_text_flags_spelled_out_minutes_and_english_words():
    brief = mid_brief()
    text = ("Après quatre minutes, la probabilité monte. À la douzième minute encore. "
            "À la vingt-deuxne minute, plusieurs takedowns alliés.")
    joined = " | ".join(check_text(text, brief, require_moments=False))
    assert "minutes écrites en lettres" in joined
    for word in ("quatre minutes", "douzième minute", "vingt-deuxne minute"):
        assert word in joined
    assert "mot anglais : takedown" in joined
    assert not any("en lettres" in i for i in check_text("À la 12e minute, en 30 minutes.", brief,
                                                          require_moments=False))


def test_check_summary_accepts_a_faithful_summary_and_rejects_format_and_length_problems():
    brief = mid_brief()
    assert check_summary(GOOD_SUMMARY, brief) == []
    assert check_summary("   ", brief) == ["résumé vide"]
    joined = " | ".join(check_summary("- Ahri gagne.\n- Points forts : le farm.", brief))
    assert "mise en forme interdite" in joined and "sections interdites" in joined
    long = GOOD_SUMMARY + " La partie est belle. Elle est finie."
    assert any("trop long (5 phrases" in i for i in check_summary(long, brief))
    causal = "Ahri gagne en 30 minutes. À la 9e minute, la probabilité passe de 54 % à 44 % suite à un kill."
    assert any("suite à" in i for i in check_summary(causal, brief))


def scripted_chat(replies, calls):
    def chat_fn(messages, model=None, host=None, seed=None):
        calls.append((messages, seed))
        return replies[len(calls) - 1]
    return chat_fn


def test_generate_summary_returns_the_first_valid_reply():
    calls = []
    text, attempts, issues = generate_summary(mid_brief(), chat_fn=scripted_chat([GOOD_SUMMARY], calls))
    assert (text, attempts, issues) == (GOOD_SUMMARY, 1, [])
    assert len(calls) == 1


def test_generate_summary_retries_with_the_errors_and_a_different_seed():
    calls = []
    bad = "Ahri gagne en 30 minutes. À la 9e minute, la probabilité passe de 54 % à 44 % grâce à un kill."
    text, attempts, issues = generate_summary(
        mid_brief(), chat_fn=scripted_chat([bad, GOOD_SUMMARY], calls))
    assert (text, attempts, issues) == (GOOD_SUMMARY, 2, [])
    (first_messages, first_seed), (second_messages, second_seed) = calls
    assert first_seed != second_seed
    assert len(second_messages) == len(first_messages) + 2
    assert second_messages[-2] == {"role": "assistant", "content": bad}
    assert "grâce à" in second_messages[-1]["content"]


def test_generate_summary_gives_up_after_max_attempts():
    calls = []
    bad = "Ahri gagne grâce à un kill."
    text, attempts, issues = generate_summary(
        mid_brief(), max_attempts=3, chat_fn=scripted_chat([bad] * 3, calls))
    assert text is None and attempts == 3 and len(calls) == 3
    assert any("grâce à" in i for i in issues)


def test_no_strong_stat_is_not_called_a_strength():
    facts = build_facts(ANALYSIS, _match(), _timeline(), "p3", bench_frame())
    facts["comparaison_mid_master"] = {
        "cs_par_min": {"valeur": 7.3, "mediane": 7.8, "percentile": 40},
        "vision_par_min": {"valeur": 0.7, "mediane": 0.7, "percentile": 55},
        "kda_ratio": {"valeur": 2.3, "mediane": 2.3, "percentile": 30},
    }
    brief = build_brief(facts)
    assert len(brief["points_forts"]) == 1
    assert brief["points_forts"][0].startswith("Aucune statistique ne se démarque nettement")
    assert "Vision par minute" in brief["points_forts"][0]  # la plus haute, au 55e percentile


def test_axes_sentence_keeps_its_punctuation_when_stat_is_not_weak():
    brief = crafted_brief()
    assert ("dans la moyenne (parmi les plus basses de cette partie, sans être faible). Conseil :"
            in brief["axes_de_progres"][1])


def test_review_sheet_update_keeps_manual_columns(tmp_path):
    path = tmp_path / "review.csv"
    key = {"match_id": "EUW1_1", "modele": "qwen2.5:7b (hybride v4)"}
    append_review_rows(path, [{**key, "controle_auto": "résumé trop long"}])
    # relecture à la main dans Excel
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["erreurs_chiffres"] = "2"
    rows[0]["remarques"] = "OK sinon"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # nouvelle exécution : le contrôle automatique est mis à jour, la relecture est conservée
    append_review_rows(path, [{**key, "controle_auto": "ok"},
                              {"match_id": "EUW1_2", "modele": key["modele"], "controle_auto": "ok"}])
    with path.open(encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f))
    assert [r["match_id"] for r in data] == ["EUW1_1", "EUW1_2"]
    assert data[0]["controle_auto"] == "ok"
    assert data[0]["erreurs_chiffres"] == "2" and data[0]["remarques"] == "OK sinon"


def test_chat_sends_the_requested_seed_and_markdown_shows_the_summary_note():
    seen = []
    server = _fake_ollama(seen)
    try:
        chat([{"role": "user", "content": "x"}], host=f"127.0.0.1:{server.server_address[1]}", seed=7)
    finally:
        server.shutdown()
    assert seen[0][1]["options"]["seed"] == 7
    md = render_markdown("EUW1_3", mid_brief(), "Texte.", [], "qwen2.5:7b (hybride v4)",
                         note="Résumé du modèle accepté à la tentative 2 sur 3.")
    assert "accepté à la tentative 2 sur 3." in md
