from src.collect import resolve_puuid
from src.riot_client import NotFound


class FakeClient:
    def __init__(self, puuid=None, error=None):
        self._puuid = puuid
        self._error = error

    def get_puuid_by_summoner_id(self, summoner_id):
        if self._error:
            raise self._error
        return self._puuid


def test_uses_puuid_when_present():
    assert resolve_puuid(FakeClient(), {"puuid": "p1", "summonerId": "s1"}) == "p1"


def test_falls_back_to_summoner_id():
    assert resolve_puuid(FakeClient(puuid="p2"), {"summonerId": "s1"}) == "p2"


def test_returns_none_when_unresolvable():
    assert resolve_puuid(FakeClient(error=NotFound("x")), {"summonerId": "s1"}) is None
    assert resolve_puuid(FakeClient(), {}) is None


# ----------------------------------------------------------------------------- --start
def test_collect_for_puuid_can_skip_the_most_recent_games(tmp_path):
    from src.collect import collect_for_puuid

    calls = []

    class Client:
        def get_match_ids(self, puuid, start=0, count=20, **kw):
            calls.append((puuid, start, count))
            return []

    assert collect_for_puuid(Client(), "p1", tmp_path, count=100, start=100) == 0
    assert calls == [("p1", 100, 100)]


# ----------------------------------------------------------------------------- interruption
def test_timeline_interrupted_while_writing_is_downloaded_again(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from src.collect import collect_match

    class Client:
        def get_match(self, match_id):
            return {"metadata": {"matchId": match_id}}

        def get_timeline(self, match_id):
            return {"info": {"frames": []}}

    real_write = Path.write_text

    def interrupted_write(self, data, *args, **kwargs):  # Ctrl+C au milieu de l'écriture de la timeline
        if "timelines" in self.parts:
            real_write(self, data[: len(data) // 2], *args, **kwargs)
            raise KeyboardInterrupt
        return real_write(self, data, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(Path, "write_text", interrupted_write)
        try:
            collect_match(Client(), "EUW1_1", tmp_path)
        except KeyboardInterrupt:
            pass

    # build_dataset lit timelines/*.json : aucun fichier tronqué ne doit y apparaître
    assert list((tmp_path / "timelines").glob("*.json")) == []
    assert collect_match(Client(), "EUW1_1", tmp_path) is True
    timeline = tmp_path / "timelines" / "EUW1_1.json"
    assert json.loads(timeline.read_text(encoding="utf-8")) == {"info": {"frames": []}}
