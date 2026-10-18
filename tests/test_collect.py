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
