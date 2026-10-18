import pytest

from src.riot_client import NotFound, RiotAPIError, RiotClient


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append((url, headers, params))
        return self.responses.pop(0)


def make_client(responses):
    session = FakeSession(responses)
    client = RiotClient(api_key="RGAPI-test", min_interval=0, session=session)
    return client, session


def test_get_puuid_encodes_name_and_uses_region_host():
    client, session = make_client([FakeResponse(200, {"puuid": "abc"})])
    assert client.get_puuid("Mon Pseudo", "EUW") == "abc"
    url, headers, _ = session.calls[0]
    assert url.startswith("https://europe.api.riotgames.com/")
    assert "Mon%20Pseudo/EUW" in url
    assert headers["X-Riot-Token"] == "RGAPI-test"


def test_retries_after_429(monkeypatch):
    sleeps = []
    monkeypatch.setattr("src.riot_client.time.sleep", lambda s: sleeps.append(s))
    client, session = make_client(
        [FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, ["EUW1_1"])]
    )
    assert client.get_match_ids("puuid") == ["EUW1_1"]
    assert 7 in sleeps
    assert len(session.calls) == 2


def test_apex_entries_use_masterleagues_endpoint():
    client, session = make_client([FakeResponse(200, {"entries": [{"puuid": "p1"}]})])
    assert client.get_apex_entries("master") == [{"puuid": "p1"}]
    url = session.calls[0][0]
    assert url.startswith("https://euw1.api.riotgames.com/")
    assert "/lol/league/v4/masterleagues/by-queue/RANKED_SOLO_5x5" in url


def test_apex_unknown_tier_raises():
    client, _ = make_client([])
    with pytest.raises(ValueError):
        client.get_apex_entries("GOLD")


def test_404_raises_not_found():
    client, _ = make_client([FakeResponse(404)])
    with pytest.raises(NotFound):
        client.get_match("EUW1_404")


def test_403_mentions_expired_key():
    client, _ = make_client([FakeResponse(403)])
    with pytest.raises(RiotAPIError, match="expirée"):
        client.get_match("EUW1_1")


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    with pytest.raises(RiotAPIError):
        RiotClient(api_key="")
