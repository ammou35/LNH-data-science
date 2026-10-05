"""Offline tests for ift6758.data.nhl -- no real network requests are made."""
import json

import pytest
import requests

from ift6758.data import nhl


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """Serves 200 for game_ids in `existing`, 404 otherwise; counts requests."""

    def __init__(self, existing=(), status=None):
        self.existing = set(existing)
        self.status = status  # force this status for every request
        self.calls = []

    def get(self, url, timeout=None):
        game_id = url.split("/gamecenter/")[1].split("/")[0]
        self.calls.append(game_id)
        if self.status is not None:
            return FakeResponse(self.status)
        if game_id in self.existing:
            return FakeResponse(200, {"id": int(game_id), "plays": []})
        return FakeResponse(404)


def test_game_id_builders():
    assert nhl.regular_season_game_id(2016, 1) == "2016020001"
    assert nhl.regular_season_game_id(2023, 1312) == "2023021312"
    # example game from the milestone README: 2022-23, final, matchup 1, game 1
    assert nhl.playoff_game_id(2022, 4, 1, 1) == "2022030411"


def test_cache_miss_downloads_and_saves(tmp_path):
    session = FakeSession(existing=["2016020001"])
    data = nhl.get_game("2016020001", tmp_path, session, delay=0)
    assert data["id"] == 2016020001
    assert (tmp_path / "2016" / "2016020001.json").exists()
    assert session.calls == ["2016020001"]


def test_cache_hit_makes_no_request(tmp_path):
    folder = tmp_path / "2016"
    folder.mkdir()
    (folder / "2016020001.json").write_text(json.dumps({"id": 1}))
    session = FakeSession()
    assert nhl.get_game("2016020001", tmp_path, session, delay=0) == {"id": 1}
    assert session.calls == []


def test_404_returns_none_and_writes_nothing(tmp_path):
    session = FakeSession()
    assert nhl.get_game("2016029999", tmp_path, session, delay=0) is None
    assert not (tmp_path / "2016" / "2016029999.json").exists()


def test_server_error_is_retried_then_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(nhl.time, "sleep", lambda s: None)
    session = FakeSession(status=500)
    with pytest.raises(requests.HTTPError):
        nhl.get_game("2016020001", tmp_path, session, delay=0, retries=3)
    assert len(session.calls) == 3


def test_client_error_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(nhl.time, "sleep", lambda s: None)
    session = FakeSession(status=403)
    with pytest.raises(requests.HTTPError):
        nhl.get_game("2016020001", tmp_path, session, delay=0, retries=3)
    assert len(session.calls) == 1


def test_regular_season_stops_after_consecutive_misses(tmp_path):
    ids = [nhl.regular_season_game_id(2016, n) for n in range(1, 4)]
    session = FakeSession(existing=ids)
    games = nhl.download_regular_season(2016, tmp_path, session, delay=0, max_misses=2)
    assert len(games) == 3
    assert len(session.calls) == 3 + 2  # 3 hits + 2 misses, then stop


def test_playoffs_skip_to_next_series_on_first_missing_game(tmp_path):
    # round 1 matchup 1 went 4 games; everything else doesn't exist
    ids = [nhl.playoff_game_id(2016, 1, 1, g) for g in range(1, 5)]
    session = FakeSession(existing=ids)
    games = nhl.download_playoffs(2016, tmp_path, session, delay=0)
    assert len(games) == 4
    # 4 hits + 1 miss in the played series + 1 miss for each of the other 14 series
    assert len(session.calls) == 4 + 1 + 14


def test_load_season_reads_only_disk_and_filters_by_type(tmp_path):
    session = FakeSession(existing=["2016020001", "2016020002", "2016030111"])
    for gid in session.existing:
        nhl.get_game(gid, tmp_path, session, delay=0)
    n_calls = len(session.calls)
    assert len(nhl.load_season(2016, "regular", tmp_path)) == 2
    assert len(nhl.load_season(2016, "playoffs", tmp_path)) == 1
    assert len(session.calls) == n_calls
    with pytest.raises(ValueError):
        nhl.load_season(2016, "preseason", tmp_path)


def test_env_var_controls_cache_location(tmp_path, monkeypatch):
    monkeypatch.setenv("NHL_DATA_DIR", str(tmp_path / "custom"))
    session = FakeSession(existing=["2016020001"])
    nhl.get_game("2016020001", session=session, delay=0)
    assert (tmp_path / "custom" / "2016" / "2016020001.json").exists()
