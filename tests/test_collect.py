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


def test_expired_key_stops_the_collection_instead_of_skipping_players():
    import pytest
    from src.riot_client import AuthError

    with pytest.raises(AuthError):
        resolve_puuid(FakeClient(error=AuthError("403")), {"summonerId": "s1"})


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


class PagedClient:
    """Riot refuse plus de 100 identifiants par appel : ce faux client vérifie qu'on ne les demande jamais."""

    def __init__(self, available):
        self.available = available
        self.calls = []

    def get_match_ids(self, puuid, start=0, count=20, start_time=None):
        assert count <= 100
        self.calls.append((start, count))
        return [f"EUW1_{i}" for i in range(start, min(start + count, self.available))]

    def get_match(self, match_id):
        return {"metadata": {"matchId": match_id}}

    def get_timeline(self, match_id):
        return {"info": {"frames": []}}


def test_more_than_100_games_are_fetched_page_by_page(tmp_path):
    from src.collect import collect_for_puuid

    client = PagedClient(available=1000)
    assert collect_for_puuid(client, "p1", tmp_path, count=250, start=10) == 250
    assert client.calls == [(10, 100), (110, 100), (210, 50)]


def test_paging_stops_at_the_end_of_the_history(tmp_path):
    from src.collect import collect_for_puuid

    client = PagedClient(available=130)
    assert collect_for_puuid(client, "p1", tmp_path, count=500) == 130
    assert client.calls == [(0, 100), (100, 100)]


def test_since_is_given_to_riot_as_a_timestamp(tmp_path):
    from src.collect import collect_for_puuid, date_to_timestamp

    assert date_to_timestamp("1970-01-02") == 86_400
    calls = []

    class Client:
        def get_match_ids(self, puuid, start=0, count=20, start_time=None):
            calls.append(start_time)
            return []

    collect_for_puuid(Client(), "p1", tmp_path, count=5, start_time=86_400)
    assert calls == [86_400]


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
