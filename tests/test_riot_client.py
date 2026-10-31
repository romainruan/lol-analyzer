import pytest
import requests

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


class FlakySession:
    """Lève les exceptions de la liste, renvoie les réponses."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.parametrize("error", [requests.ConnectionError("coupure"), requests.Timeout("lent")])
def test_network_error_is_retried(monkeypatch, error):
    monkeypatch.setattr("src.riot_client.time.sleep", lambda s: None)
    session = FlakySession([error, FakeResponse(200, {"metadata": {}})])
    client = RiotClient(api_key="RGAPI-test", min_interval=0, session=session)
    assert client.get_match("EUW1_1") == {"metadata": {}}
    assert session.calls == 2


def test_network_error_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("src.riot_client.time.sleep", lambda s: None)
    session = FlakySession([requests.ConnectionError("coupure")] * 3)
    client = RiotClient(api_key="RGAPI-test", min_interval=0, max_retries=3, session=session)
    with pytest.raises(RiotAPIError):
        client.get_match("EUW1_1")
    assert session.calls == 3


@pytest.mark.parametrize("platform, region", [("kr", "asia"), ("na1", "americas"), ("EUN1", "europe"), ("sg2", "sea")])
def test_region_follows_the_platform(monkeypatch, platform, region):
    monkeypatch.setenv("RIOT_REGION", "europe")  # celle du .env : le choix explicite de la plateforme passe avant
    assert RiotClient(api_key="k", platform=platform).region == region


def test_region_can_still_be_forced_and_comes_from_the_environment(monkeypatch):
    assert RiotClient(api_key="k", platform="kr", region="americas").region == "americas"
    monkeypatch.setenv("RIOT_PLATFORM", "na1")
    monkeypatch.setenv("RIOT_REGION", "americas")
    client = RiotClient(api_key="k")
    assert (client.platform, client.region) == ("na1", "americas")


def test_platform_alone_in_the_environment_gives_its_region(monkeypatch):
    monkeypatch.setenv("RIOT_PLATFORM", "kr")
    monkeypatch.delenv("RIOT_REGION", raising=False)
    assert RiotClient(api_key="k").region == "asia"


def test_accounts_are_not_served_by_the_sea_region():
    session = FakeSession([FakeResponse(200, {"puuid": "abc"})])
    client = RiotClient(api_key="k", platform="sg2", min_interval=0, session=session)
    assert client.get_puuid("Pseudo", "SG2") == "abc"
    assert session.calls[0][0].startswith("https://asia.api.riotgames.com/riot/account/")


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("RIOT_API_KEY", raising=False)
    with pytest.raises(RiotAPIError):
        RiotClient(api_key="")
