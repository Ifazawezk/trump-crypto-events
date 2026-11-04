import pytest

from cryptoevents.http import USER_AGENT, BlockedError, Client


class FakeResponse:
    def __init__(self, status, content=b"", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls, self.headers = list(responses), [], {}

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        return self.responses.pop(0)


def make(tmp_path, *responses):
    session = FakeSession(responses)
    return Client(tmp_path, min_interval=0, session=session), session


def test_honest_user_agent(tmp_path):
    client, session = make(tmp_path)
    assert session.headers["User-Agent"] == USER_AGENT
    assert "Mozilla" not in USER_AGENT


def test_second_call_comes_from_cache(tmp_path):
    client, session = make(tmp_path, FakeResponse(200, b'{"a": 1}'))
    assert client.get("https://x.test/a", expect="json") == b'{"a": 1}'
    assert client.get("https://x.test/a", expect="json") == b'{"a": 1}'
    assert len(session.calls) == 1


def test_404_is_remembered(tmp_path):
    client, session = make(tmp_path, FakeResponse(404))
    assert client.get("https://x.test/missing") is None
    assert client.get("https://x.test/missing") is None
    assert len(session.calls) == 1


def test_access_page_with_status_200_is_refused_and_not_cached(tmp_path):
    page = b"<!DOCTYPE html><title>Request Access</title>"
    client, session = make(tmp_path, FakeResponse(200, page), FakeResponse(200, page))
    with pytest.raises(BlockedError):
        client.get("https://x.test/doc.txt", expect="text")
    with pytest.raises(BlockedError):
        client.get("https://x.test/doc.txt", expect="text")
    assert len(session.calls) == 2


def test_403_stops_without_retry(tmp_path):
    client, session = make(tmp_path, FakeResponse(403))
    with pytest.raises(BlockedError):
        client.get("https://x.test/a")
    assert len(session.calls) == 1


def test_zip_check(tmp_path):
    client, _ = make(tmp_path, FakeResponse(200, b"not a zip"))
    with pytest.raises(BlockedError):
        client.get("https://x.test/a.zip", expect="zip")
