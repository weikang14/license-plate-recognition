import threading
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import pytest

from tools.serve import RangeHandler


@pytest.fixture
def base(tmp_path):
    (tmp_path / "a.bin").write_bytes(bytes(range(100)))
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(tmp_path)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/a.bin"
    server.shutdown()


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": rng} if rng else {})
    with urllib.request.urlopen(req) as r:
        return r.status, r.headers.get("Content-Range"), r.read()


def test_byte_range(base):
    assert get(base, "bytes=10-19") == (206, "bytes 10-19/100", bytes(range(10, 20)))


def test_open_ended_and_suffix_ranges(base):
    assert get(base, "bytes=95-") == (206, "bytes 95-99/100", bytes(range(95, 100)))
    assert get(base, "bytes=-5") == (206, "bytes 95-99/100", bytes(range(95, 100)))


def test_no_range_sends_whole_file(base):
    assert get(base) == (200, None, bytes(range(100)))
