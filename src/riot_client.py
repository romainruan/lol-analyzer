"""Client minimal pour l'API Riot Games, avec gestion du rate limit.

La clé de développement est limitée à ~20 requêtes/s et 100 requêtes/2 min.
On espace donc les appels (1,25 s par défaut => ~96 requêtes / 2 min) et on
respecte l'en-tête Retry-After en cas d'erreur 429.
"""
from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote

import requests


# plateforme (serveur de jeu) -> région de routage des parties
REGION_BY_PLATFORM = {
    "euw1": "europe", "eun1": "europe", "tr1": "europe", "ru": "europe", "me1": "europe",
    "na1": "americas", "br1": "americas", "la1": "americas", "la2": "americas",
    "kr": "asia", "jp1": "asia",
    "oc1": "sea", "sg2": "sea", "tw2": "sea", "vn2": "sea",
}


class RiotAPIError(Exception):
    """Erreur générique de l'API Riot."""


class NotFound(RiotAPIError):
    """Ressource introuvable (HTTP 404)."""


class AuthError(RiotAPIError):
    """Clé invalide ou expirée (HTTP 401/403) : inutile de continuer la collecte."""


class RiotClient:
    def __init__(
        self,
        api_key: str | None = None,
        platform: str | None = None,
        region: str | None = None,
        min_interval: float = 1.25,
        max_retries: int = 5,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("RIOT_API_KEY", "")
        if not self.api_key:
            raise RiotAPIError("RIOT_API_KEY manquante (voir .env.example).")
        # platform : serveur de jeu (euw1, na1...) ; region : routage (europe, americas, asia, sea).
        # Une plateforme demandée explicitement passe avant la région du .env, qui vaut pour une autre plateforme.
        self.platform = (platform or os.environ.get("RIOT_PLATFORM", "euw1")).lower()
        derived = REGION_BY_PLATFORM.get(self.platform, "europe")
        self.region = region or (derived if platform else os.environ.get("RIOT_REGION", derived))
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self._last_call = 0.0

    # ------------------------------------------------------------------ bas niveau
    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _get(self, host: str, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"https://{host}.api.riotgames.com{path}"
        headers = {"X-Riot-Token": self.api_key}
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                r = self.session.get(url, headers=headers, params=params, timeout=30)
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(2**attempt)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 10)))
                continue
            if r.status_code == 404:
                raise NotFound(url)
            if r.status_code in (401, 403):
                raise AuthError(
                    f"{r.status_code} : clé invalide ou expirée (la clé de dev dure 24 h)."
                )
            if r.status_code >= 500:
                time.sleep(2**attempt)
                continue
            raise RiotAPIError(f"{r.status_code} sur {url}")
        raise RiotAPIError(f"Trop de tentatives échouées sur {url}")

    # ------------------------------------------------------------------ endpoints
    def get_puuid(self, game_name: str, tag_line: str) -> str:
        path = (
            "/riot/account/v1/accounts/by-riot-id/"
            f"{quote(game_name, safe='')}/{quote(tag_line, safe='')}"
        )
        # les comptes ne sont pas servis par la région « sea »
        return self._get("asia" if self.region == "sea" else self.region, path)["puuid"]

    def get_match_ids(
        self,
        puuid: str,
        start: int = 0,
        count: int = 20,
        queue: int | None = 420,  # 420 = ranked solo/duo
        start_time: int | None = None,
    ) -> list[str]:
        params: dict[str, Any] = {"start": start, "count": count}
        if queue is not None:
            params["queue"] = queue
        if start_time is not None:
            params["startTime"] = start_time
        return self._get(self.region, f"/lol/match/v5/matches/by-puuid/{puuid}/ids", params)

    def get_match(self, match_id: str) -> dict:
        return self._get(self.region, f"/lol/match/v5/matches/{match_id}")

    def get_timeline(self, match_id: str) -> dict:
        return self._get(self.region, f"/lol/match/v5/matches/{match_id}/timeline")

    def get_league_entries(
        self, tier: str, division: str, page: int = 1, queue: str = "RANKED_SOLO_5x5"
    ) -> list[dict]:
        """Joueurs d'une division (IRON..DIAMOND). Les tiers apex ont un autre endpoint."""
        return self._get(
            self.platform,
            f"/lol/league/v4/entries/{queue}/{tier.upper()}/{division.upper()}",
            {"page": page},
        )

    def get_apex_entries(self, tier: str, queue: str = "RANKED_SOLO_5x5") -> list[dict]:
        """Tous les joueurs de MASTER, GRANDMASTER ou CHALLENGER (pas de divisions)."""
        endpoints = {
            "MASTER": "masterleagues",
            "GRANDMASTER": "grandmasterleagues",
            "CHALLENGER": "challengerleagues",
        }
        key = tier.upper()
        if key not in endpoints:
            raise ValueError(f"Tier apex inconnu : {tier}")
        data = self._get(self.platform, f"/lol/league/v4/{endpoints[key]}/by-queue/{queue}")
        return data["entries"]

    def get_puuid_by_summoner_id(self, summoner_id: str) -> str:
        """Repli si une entrée de ligue ne contient pas directement le puuid."""
        return self._get(self.platform, f"/lol/summoner/v4/summoners/{summoner_id}")["puuid"]
