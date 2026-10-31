import json
import re

import numpy as np
import pandas as pd
from synthetic import make_game

from src.benchmarks import STAT_COLS
from src.report import build_report, curve_svg, debrief_html, main

ANALYSIS = {
    "match_id": "EUW1_1",
    "perspective": {"team": "bleue", "champion": "Syndra", "role": "MIDDLE", "result": "victoire"},
    "curve": [{"minute": m, "win_prob": p} for m, p in [(2, 0.42), (3, 0.39), (4, 0.62), (5, 0.56), (6, 0.63), (7, 0.99)]],
    "turning_points": [{"minute": 4, "win_prob_before": 0.39, "win_prob_after": 0.62, "ecarts": {"or": 900},
                        "events": ["Sivir (alliés) tue Draven (ennemis)", "Udyr (alliés) tue Shaco (ennemis)"]}],
}


def _write_game(raw_dir, game):
    match, timeline = game
    name = match["metadata"]["matchId"] + ".json"
    (raw_dir / "matches").mkdir(parents=True, exist_ok=True)
    (raw_dir / "timelines").mkdir(parents=True, exist_ok=True)
    (raw_dir / "matches" / name).write_text(json.dumps(match), encoding="utf-8")
    (raw_dir / "timelines" / name).write_text(json.dumps(timeline), encoding="utf-8")
    return match["metadata"]["matchId"]


def _player_dir(tmp_path, n_games=12):
    raw = tmp_path / "raw"
    (raw / "puuid.txt").parent.mkdir(parents=True)
    (raw / "puuid.txt").write_text("puuid-secret-123", encoding="utf-8")
    (raw / "riot_id.txt").write_text("Pseudo#EUW", encoding="utf-8")
    ids = [_write_game(raw, make_game(i, me="puuid-secret-123", champion="Syndra")) for i in range(n_games)]
    return raw, ids


def _bench():
    return pd.DataFrame({**{c: np.linspace(1, 500, 100) for c in STAT_COLS}, "role": "MIDDLE", "champion": "Zed"})


def _curves(tmp_path, ids):
    curves = tmp_path / "curves"
    curves.mkdir()
    for match_id in ids:
        (curves / f"{match_id}.json").write_text(json.dumps({**ANALYSIS, "match_id": match_id}), encoding="utf-8")
    return curves


# ----------------------------------------------------------------------------- courbe
def test_curve_is_drawn_from_its_points_with_every_value_available():
    svg = curve_svg(ANALYSIS)
    assert svg.startswith("<svg") and 'class="curve"' in svg
    points = json.loads(re.search(r'data-points="([^"]+)"', svg).group(1).replace("&quot;", '"'))
    assert points[0] == [2, 42] and points[-1] == [7, 99]
    assert "peu fiable" in svg  # la partie commence avant la 5e minute
    assert "99 %" in svg  # la valeur finale est écrite sur le graphique
    assert 'tabindex="0"' in svg  # accessible au clavier


def test_curve_without_early_minutes_has_no_warning():
    late = {**ANALYSIS, "curve": [{"minute": m, "win_prob": 0.5} for m in (6, 7)], "turning_points": []}
    assert "peu fiable" not in curve_svg(late)


def test_curve_with_a_single_point_does_not_crash():
    single = {**ANALYSIS, "curve": [{"minute": 8, "win_prob": 0.5}], "turning_points": []}
    assert curve_svg(single).startswith("<svg")


# ----------------------------------------------------------------------------- débrief
def test_debrief_keeps_the_text_and_leaves_out_the_facts():
    md = ("# Débrief EUW1_1\n\n**Résumé**\nUne phrase.\n\n**Moments clés**\n- 4e minute : un saut.\n- 9e minute : un autre.\n\n"
          "---\n*Modèle : m. Vérification automatique : ok.*\n\nFaits calculés par le code :\n\n```json\n{\"a\": 1}\n```\n")
    html = debrief_html(md)
    assert "<h4>Résumé</h4>" in html and "<p>Une phrase.</p>" in html
    assert "<li>4e minute : un saut.</li>" in html
    assert "Vérification automatique" not in html and '"a"' not in html and "# Débrief" not in html


def test_debrief_text_is_escaped():
    html = debrief_html("**Résumé**\n<script>alert(1)</script>\n\n**<b>Titre</b>**\n- <i>puce</i>\n")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<b>" not in html and "<i>" not in html


# ----------------------------------------------------------------------------- rapport
def test_report_shows_progress_priorities_and_games(tmp_path):
    raw, ids = _player_dir(tmp_path)
    (tmp_path / "debriefs").mkdir()
    (tmp_path / "debriefs" / f"{ids[-1]}_gabarit.md").write_text("**Résumé**\nCode.\n\n---\nFaits", encoding="utf-8")
    html = build_report(raw, _curves(tmp_path, ids), tmp_path / "debriefs", _bench(), limit=5, window=5)
    assert "Pseudo#EUW" in html and "<!doctype html>" in html.lower()
    assert "Syndra" in html
    assert "mieux que" in html and "Conseil" in html  # priorités face aux repères
    assert "pas de conclusion" in html or "en progrès" in html or "en baisse" in html  # tendances
    assert html.count('class="curve"') == 5  # `limit` parties détaillées
    assert "<p>Code.</p>" in html  # le débrief est repris
    assert "isn't endorsed by Riot Games" in html  # mention légale exigée par Riot


def test_report_starts_with_the_most_recent_game(tmp_path):
    raw, ids = _player_dir(tmp_path)
    html = build_report(raw, _curves(tmp_path, ids), tmp_path / "absent", None, limit=2, window=5)
    assert html.index(ids[-1]) < html.index(ids[-2])
    assert ids[-3] not in html


def test_report_never_contains_the_puuid_nor_external_resources(tmp_path):
    raw, ids = _player_dir(tmp_path)
    html = build_report(raw, _curves(tmp_path, ids), tmp_path / "absent", _bench(), limit=5, window=5)
    assert "puuid-secret-123" not in html
    assert not re.search(r'(?:src|href)="https?:', html)


def test_report_escapes_names_coming_from_the_data(tmp_path):
    raw, ids = _player_dir(tmp_path)
    (raw / "riot_id.txt").write_text("<img src=x onerror=alert(1)>#EUW", encoding="utf-8")
    hostile = {**ANALYSIS, "match_id": ids[0], "perspective": {**ANALYSIS["perspective"], "champion": "<script>x</script>"}}
    curves = tmp_path / "curves"
    curves.mkdir()
    (curves / f"{ids[0]}.json").write_text(json.dumps(hostile), encoding="utf-8")
    html = build_report(raw, curves, tmp_path / "absent", None)
    assert "<script>x</script>" not in html and "<img src=x" not in html
    assert "&lt;script&gt;x" in html


def test_report_without_references_or_curves_still_gives_the_trends(tmp_path):
    raw, _ = _player_dir(tmp_path)
    html = build_report(raw, tmp_path / "absent", tmp_path / "absent", None, window=5)
    assert 'class="curve"' not in html
    assert "src.run prepare" in html  # explique comment obtenir le reste
    assert "pas de conclusion" in html or "en progrès" in html or "en baisse" in html


def test_report_of_an_empty_folder_says_what_to_do(tmp_path):
    html = build_report(tmp_path, tmp_path / "absent", tmp_path / "absent", None)
    assert "src.run report" in html


def test_report_ignores_an_analysis_without_its_game(tmp_path):
    raw, ids = _player_dir(tmp_path, n_games=2)
    curves = _curves(tmp_path, ids + ["EUW1_404"])
    html = build_report(raw, curves, tmp_path / "absent", None)
    assert "EUW1_404" not in html
    assert html.count('class="curve"') == 2


def test_main_writes_a_utf8_file(tmp_path):
    raw, ids = _player_dir(tmp_path)
    out = tmp_path / "sortie" / "rapport.html"
    main(["--raw-dir", str(raw), "--curves", str(_curves(tmp_path, ids)), "--debriefs", str(tmp_path / "absent"),
          "--bench", str(tmp_path / "absent.csv"), "--out", str(out)])
    assert "Syndra" in out.read_text(encoding="utf-8")
