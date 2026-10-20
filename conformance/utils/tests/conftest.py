# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve one native publication shared by browser workers."""
import fcntl
import gzip
import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "conformance/utils/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from native_publication import publish


@pytest.fixture(scope="session")
def published_viewer(tmp_path_factory):
    configured = os.environ.get("CONFORMANCE_PUBLISHED_SITE")
    expected = json.loads((REPO / "conformance/viewer-catalog.json").read_bytes())["revision"]
    if configured:
        site = Path(configured).resolve(strict=True)
    else:
        base = tmp_path_factory.getbasetemp()
        if os.environ.get("PYTEST_XDIST_WORKER"):
            base = base.parent
        parent = base / "native-browser-publication"
        parent.mkdir(exist_ok=True)
        site = parent / "published"
        with (parent / ".native-build.lock").open("a+b") as lease:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
            catalog_path = site / "conformance/viewer-resolved-catalog.json.gz"
            existing = json.loads(gzip.decompress(catalog_path.read_bytes())) if catalog_path.is_file() else {}
            if existing.get("source_revision") != expected or existing.get("transport", {}).get("schema") != 2:
                publish(REPO, site)
    catalog = json.loads(gzip.decompress((site / "conformance/viewer-resolved-catalog.json.gz").read_bytes()))
    assert catalog.get("source_revision") == expected
    assert catalog["transport"]["schema"] == 2
    assert catalog["transport"]["raw_buckets"]
    assert not (site / "conformance/viewer-original-resources.json.gz").exists()
    return site


@pytest.fixture(scope="session")
def viewer_url(published_viewer):
    prefix = "/frontend-crates"

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(published_viewer), **kwargs)

        def translate_path(self, path):
            path = urlsplit(path).path
            if not path.startswith(prefix + "/"):
                return str(published_viewer / "__outside_prefix__")
            return super().translate_path(path[len(prefix):])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}{prefix}/index.html"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
