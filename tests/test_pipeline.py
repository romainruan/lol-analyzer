"""La chaîne complète, de la collecte au rapport, sur des parties synthétiques (ni réseau, ni clé, ni Ollama)."""
import json
from pathlib import Path

import pandas as pd
import pytest
from synthetic import FakeRiotClient

from src import run


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Un dossier de travail vide et un faux client Riot : 70 joueurs de référence, 6 parties à vous."""
    monkeypatch.chdir(tmp_path)
    client = FakeRiotClient(n_sample=70, n_mine=6)
    monkeypatch.setattr("src.collect.RiotClient", lambda **kw: client)
    return tmp_path


def prepare():
    run.main(["prepare", "--tier", "MASTER", "--max-players", "70", "--per-player", "1"])


def report(*extra):
    run.main(["report", "--riot-id", "Pseudo#EUW", "--count", "6", *extra])


def test_prepare_builds_the_dataset_the_model_and_the_references(project):
    prepare()
    assert len(list(Path("data/raw/sample/matches").glob("*.json"))) == 70
    dataset = pd.read_csv("data/dataset_all.csv")
    assert dataset["match_id"].nunique() == 70
    assert {"inhibitors_diff", "souls_diff", "gold_share_diff"} <= set(dataset.columns)
    assert Path("results/curve_model.joblib").exists()
    assert Path("results/curve_calibration.csv").exists()
    bench = pd.read_csv("data/benchmarks.csv")
    assert set(bench["role"]) == {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}
    assert set(bench["reference"]) == {"Master"}


def test_report_goes_from_the_riot_id_to_a_html_page(project, capsys):
    prepare()
    report()
    assert Path("data/raw/me/puuid.txt").read_text(encoding="utf-8") == "puuid-secret-123"
    assert len(list(Path("results/curves").glob("*.json"))) == 6
    assert len(list(Path("results/curves").glob("*.png"))) == 6
    debriefs = list(Path("results/debriefs").glob("*_gabarit.md"))
    assert len(debriefs) == 6
    for path in debriefs:  # le texte écrit par le code passe toujours son propre contrôle
        assert "aucun problème détecté" in path.read_text(encoding="utf-8"), path.read_text(encoding="utf-8")
    html = Path("results/report.html").read_text(encoding="utf-8")
    assert "Pseudo#EUW" in html and html.count('class="curve"') == 6
    assert "puuid-secret-123" not in html
    assert "report.html" in capsys.readouterr().out


def test_debriefs_written_by_the_code_always_pass_their_own_control(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client = FakeRiotClient(n_sample=70, n_mine=30)
    monkeypatch.setattr("src.collect.RiotClient", lambda **kw: client)
    prepare()
    run.main(["report", "--riot-id", "Pseudo#EUW", "--count", "30"])
    flagged = [p.name for p in Path("results/debriefs").glob("*_gabarit.md")
               if "aucun problème détecté" not in p.read_text(encoding="utf-8")]
    assert len(list(Path("results/debriefs").glob("*_gabarit.md"))) == 30
    assert flagged == []


def test_a_second_report_only_adds_what_is_new(project):
    prepare()
    report()
    first = {p: p.stat().st_mtime_ns for p in Path("results/debriefs").glob("*.md")}
    report()
    assert {p: p.stat().st_mtime_ns for p in Path("results/debriefs").glob("*.md")} == first


def test_report_without_a_model_still_gives_the_trends(project, capsys):
    report()
    html = Path("results/report.html").read_text(encoding="utf-8")
    assert 'class="curve"' not in html and "Pseudo#EUW" in html
    assert "src.run prepare" in capsys.readouterr().out


def test_report_can_reuse_games_already_downloaded(project, monkeypatch):
    prepare()
    report()
    def no_network(**kw):
        raise AssertionError("aucun appel à Riot attendu")
    monkeypatch.setattr("src.collect.RiotClient", no_network)
    report("--no-collect")
    assert Path("results/report.html").exists()


def test_llm_summary_is_used_when_ollama_answers(project, monkeypatch):
    prepare()

    def faithful_chat(messages, **kw):  # recopie la phrase de la courbe : elle passe le contrôle
        facts = json.loads(messages[1]["content"].split("```json\n")[1].split("\n```")[0])
        return facts["courbe"]

    monkeypatch.setattr("src.debrief.chat", faithful_chat)
    report("--llm")
    written = list(Path("results/debriefs").glob("EUW1_*.md"))
    assert len(written) == 6 and not any(p.stem.endswith("_gabarit") for p in written)
    assert all("hybride v5.3" in p.read_text(encoding="utf-8") for p in written)
    assert all("Résumé du modèle accepté" in p.read_text(encoding="utf-8") for p in written)


def test_llm_falls_back_on_the_code_when_ollama_is_down(project, monkeypatch, capsys):
    from src.debrief import OllamaError

    prepare()

    def down(messages, **kw):
        raise OllamaError("Ollama ne répond pas")

    monkeypatch.setattr("src.debrief.chat", down)
    report("--llm")
    assert len(list(Path("results/debriefs").glob("*_gabarit.md"))) == 6
    assert "Ollama ne répond pas" in capsys.readouterr().out


def test_prepare_can_reuse_a_sample_already_collected(project, monkeypatch):
    prepare()
    Path("results/curve_model.joblib").unlink()

    def no_network(**kw):
        raise AssertionError("aucun appel à Riot attendu")
    monkeypatch.setattr("src.collect.RiotClient", no_network)
    run.main(["prepare", "--tier", "MASTER", "--no-collect", "--kind", "monotone"])
    assert Path("results/curve_model.joblib").exists()


def test_prepare_without_data_explains_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="Moins de 50 parties|Aucune partie"):
        run.main(["prepare", "--tier", "MASTER", "--no-collect"])


def test_commands_accept_their_arguments_when_called_directly(project, monkeypatch):
    from src import benchmarks, build_dataset, collect, debrief, winprob

    collect.main(["--tier", "MASTER", "--max-players", "70", "--per-player", "1"])
    build_dataset.main(["--all-minutes", "--out", "data/d.csv"])
    winprob.main(["train", "--data", "data/d.csv", "--kind", "phases"])
    winprob.main(["compare", "--data", "data/d.csv", "--boot", "20", "--folds", "2"])
    winprob.main(["learning-curve", "--data", "data/d.csv"])
    benchmarks.main(["--label", "Master"])
    collect.main(["--riot-id", "Pseudo#EUW", "--count", "2", "--since", "2026-01-01"])
    assert Path("data/raw/me/riot_id.txt").read_text(encoding="utf-8") == "Pseudo#EUW"
    winprob.main(["analyze", "--limit", "2"])
    debrief.main(["--no-llm", "--limit", "2"])
    assert len(list(Path("results/debriefs").glob("*_gabarit.md"))) == 2
    assert Path("results/curve_comparison.csv").exists()
