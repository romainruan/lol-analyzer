import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pandas as pd
import pytest

from src.benchmarks import STAT_COLS
from src.debrief import (
    OllamaError,
    build_facts,
    build_messages,
    chat,
    check_numbers,
    render_markdown,
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
    return pd.DataFrame({c: np.linspace(1, 10, 100) for c in STAT_COLS})


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


def test_prompt_contains_data_and_rules():
    facts = build_facts(ANALYSIS, _match(), _timeline(), "p3")
    system, user = build_messages(facts)
    assert "N'invente rien" in system["content"]
    assert '"champion": "Ahri"' in user["content"]


# ----------------------------------------------------------------------------- vérification
def test_check_numbers_accepts_given_numbers_and_rounding():
    facts = build_facts(ANALYSIS, _match(), _timeline(), "p3")
    text = ("1. À la 9e minute, la probabilité passe de 54 % à 44 %.\n"
            "2. Votre KDA de 6/2/6 et 7 CS par minute.")  # 7 pour 7,3 : arrondi toléré
    assert check_numbers(text, facts) == []


def test_check_numbers_flags_invented_numbers():
    facts = build_facts(ANALYSIS, _match(), _timeline(), "p3")
    assert check_numbers("Vous êtes à 87 % de victoire et 12,5 CS par minute.", facts) == [12.5, 87.0]


def test_render_markdown_mentions_check_result():
    facts = build_facts(ANALYSIS, _match(), _timeline(), "p3")
    ok = render_markdown("EUW1_3", facts, "Texte.", [], "qwen2.5:7b")
    assert "aucun nombre inventé" in ok and "# Débrief EUW1_3" in ok
    bad = render_markdown("EUW1_3", facts, "Texte.", [87.0], "qwen2.5:7b")
    assert "à vérifier à la main : 87" in bad


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
