"""La chaîne complète, de la collecte au rapport, sur des parties synthétiques (ni réseau, ni clé, ni Ollama)."""
import json
import os
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


def test_a_retrained_model_replaces_the_old_curves_and_debriefs(project):
    prepare()
    report()
    written = [*Path("results/curves").glob("*.json"), *Path("results/debriefs").glob("*.md")]
    old = Path("results/curve_model.joblib").stat().st_mtime - 100  # tout date d'avant le dernier entraînement
    for path in written:
        os.utime(path, (old, old))
    report("--no-collect")
    assert [p.name for p in written if p.stat().st_mtime == old] == []


def test_missing_or_expired_key_gives_a_message_instead_of_a_trace(tmp_path, monkeypatch):
    from src.riot_client import AuthError

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    monkeypatch.setattr("src.collect.load_dotenv", lambda: None)
    with pytest.raises(SystemExit, match="RIOT_API_KEY"):
        run.main(["report", "--riot-id", "Pseudo#EUW"])

    class Expired:
        def get_puuid(self, *a):
            raise AuthError("403 : clé invalide ou expirée")

    monkeypatch.setattr("src.collect.RiotClient", lambda **kw: Expired())
    with pytest.raises(SystemExit, match="expirée"):
        run.main(["report", "--riot-id", "Pseudo#EUW"])


def test_report_without_a_model_still_gives_the_trends(project, capsys):
    report()
    html = Path("results/report.html").read_text(encoding="utf-8")
    assert 'class="curve"' not in html and "Pseudo#EUW" in html
    assert "src.run prepare" in capsys.readouterr().out


def test_references_can_be_kept_in_another_file(project):
    run.main(["prepare", "--tier", "MASTER", "--max-players", "70", "--per-player", "1", "--label", "Gold",
              "--bench", "data/gold.csv"])
    assert not Path("data/benchmarks.csv").exists()
    report("--bench", "data/gold.csv")
    assert "des mid Gold" in Path("results/report.html").read_text(encoding="utf-8")


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


def test_model_evaluations_run_from_the_command_line(project, capsys):
    from src import build_dataset, collect, train, winprob

    collect.main(["--tier", "MASTER", "--max-players", "70", "--per-player", "1"])
    build_dataset.main(["--out", "data/d5.csv"])
    train.main(["--data", "data/d5.csv", "--out", "results", "--folds", "2"])
    for name in ("metrics_by_minute.csv", "auc_by_minute.png", "lane_coefficients.csv", "calibration.csv"):
        assert Path("results", name).exists(), name
    build_dataset.main(["--all-minutes", "--out", "data/d.csv"])
    winprob.main(["train", "--data", "data/d.csv", "--model", "results/m.joblib", "--time-split", "0.2",
                  "--patch-holdout", "--kind", "monotone", "--features", "etendu"])
    out = capsys.readouterr().out
    assert "Test dans le temps" in out and "auc_lo" in out
    assert "Aucun patch avec assez de parties" in out  # 70 parties : pas assez pour un test par patch
    assert "poids sont négatifs" not in out


def test_free_model_warns_that_it_cannot_explain_the_jumps(project, capsys):
    from src import build_dataset, collect, winprob

    collect.main(["--tier", "MASTER", "--max-players", "70", "--per-player", "1"])
    build_dataset.main(["--all-minutes", "--out", "data/d.csv"])
    winprob.main(["train", "--data", "data/d.csv", "--model", "results/m.joblib"])
    assert "--kind monotone" in capsys.readouterr().out


def test_progress_command_prints_trends_and_priorities(project, capsys):
    from src import progress

    with pytest.raises(SystemExit, match="collect --riot-id"):
        progress.main([])
    prepare()
    report()
    capsys.readouterr()
    progress.main(["--window", "3"])
    out = capsys.readouterr().out
    assert "parties, poste principal : mid" in out
    assert "À travailler en priorité" in out and "Conseil" in out
    assert "Ahri" in out and "Syndra" in out


def test_progress_command_without_references_says_how_to_get_them(project, capsys):
    from src import progress

    report()
    capsys.readouterr()
    progress.main([])
    assert "src.benchmarks" in capsys.readouterr().out


def test_debrief_dry_run_shows_the_prompt_without_calling_a_model(project, monkeypatch, capsys):
    from src import debrief

    def no_model(*a, **kw):
        raise AssertionError("aucun appel au modèle attendu")

    prepare()
    report()
    monkeypatch.setattr("src.debrief.chat", no_model)
    capsys.readouterr()
    debrief.main(["--limit", "1", "--dry-run"])
    assert "FAITS" in capsys.readouterr().out


def test_debrief_recheck_runs_from_the_command_line(project, capsys):
    from src import debrief

    prepare()
    report()
    capsys.readouterr()
    debrief.main(["--recheck", "results/debriefs"])
    assert "0 résumés" in capsys.readouterr().out  # des gabarits uniquement : aucun résumé de modèle à rejouer


def test_commands_explain_what_is_missing(tmp_path, monkeypatch):
    from src import build_dataset, debrief, run, winprob

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="src.collect"):
        build_dataset.main([])
    with pytest.raises(SystemExit, match="src.build_dataset"):
        winprob.main(["compare"])
    with pytest.raises(SystemExit, match="collect --riot-id"):
        debrief.main([])
    with pytest.raises(SystemExit, match="--riot-id"):
        run.main(["report"])


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
