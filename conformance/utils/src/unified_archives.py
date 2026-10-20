# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte-preserving engine/version archives for the Unified source store."""

from __future__ import annotations

import gzip
import hashlib
import io
import re
import shutil
import tarfile
import tempfile
import zlib
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import yaml

import execution_manifests
import impls


DIRECTORY = "versions"
MAX_MEMBER = 32 * 1024 * 1024
MAX_DECODED = 512 * 1024 * 1024
MAX_COMPRESSED = 32 * 1024 * 1024


def _member(name: str) -> None:
    parts = name.split("/")
    if (
        "\\" in name or ":" in name
        or any(part in {"", ".", ".."} for part in parts)
        or not name.endswith(".yaml")
        or not (len(parts) == 3 and parts[0] == "families"
                and parts[2] not in {"inputs_and_golden.yaml", "index.yaml"}
                or len(parts) == 2 and parts[0] in {"executions", "receipts"})
    ):
        raise ValueError(f"invalid Unified archive member: {name}")


def paths(root: Path) -> list[Path]:
    directory = root / DIRECTORY
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Unified versions must be a regular directory")
    result = sorted(directory.iterdir())
    for path in result:
        if (path.is_symlink() or not path.is_file()
                or re.fullmatch(r"(?:dynamo|vllm|sglang)-[0-9][a-zA-Z0-9.-]*\.tar\.gz", path.name) is None
                or path.stat().st_size > MAX_COMPRESSED):
            raise ValueError(f"invalid Unified version archive: {path}")
    return result


def members(path: Path):
    """Read one bounded tar payload, with no links or concatenated gzip streams."""
    seen = set()
    if path.stat().st_size > MAX_COMPRESSED:
        raise ValueError(f"Unified archive exceeds compressed limit: {path}")
    decoder = zlib.decompressobj(wbits=31)
    payload = decoder.decompress(path.read_bytes(), MAX_DECODED + 1)
    if len(payload) > MAX_DECODED:
        raise ValueError(f"Unified archive exceeds decoded limit: {path}")
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError(f"truncated or trailing Unified gzip data: {path}")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive:
            _member(member.name)
            if (not member.isfile() or member.linkname or member.pax_headers
                    or member.name in seen or member.size > MAX_MEMBER):
                raise ValueError(f"invalid or duplicate Unified archive member: {member.name}")
            seen.add(member.name)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"missing Unified archive member: {member.name}")
            data = stream.read(MAX_MEMBER + 1)
            if len(data) != member.size:
                raise ValueError(f"truncated Unified archive member: {member.name}")
            yield member.name, data
        if not seen:
            raise ValueError(f"empty Unified archive: {path}")
        if len(payload) < archive.offset + 1024 or any(payload[archive.offset:]):
            raise ValueError(f"invalid Unified tar end markers: {path}")


def inventory(root: Path) -> list[dict]:
    rows = {}
    for path in sorted([*root.glob("families/*/*.yaml"),
                        *root.glob("executions/*.yaml"), *root.glob("receipts/*.yaml")]):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unified source must be a regular file: {path}")
        data = path.read_bytes()
        name = path.relative_to(root).as_posix()
        rows[name] = {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    for path in paths(root):
        data = path.read_bytes()
        descriptor = {"path": path.relative_to(root).as_posix(),
                      "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        for name, data in members(path):
            if name in rows:
                raise ValueError(f"duplicate Unified source: {name}")
            rows[name] = {"path": name, "sha256": hashlib.sha256(data).hexdigest(),
                          "size": len(data), "archive": descriptor}
    return [rows[name] for name in sorted(rows)]


def read_source(root: Path, resource: dict) -> bytes:
    if archive := resource.get("archive"):
        path = Path(archive["path"])
        data = (root / path).read_bytes()
        if len(data) != archive["size"] or hashlib.sha256(data).hexdigest() != archive["sha256"]:
            raise ValueError("source archive identity mismatch")
        member = Path(resource["path"]).relative_to(path.parent.parent).as_posix()
        found = None
        for name, data in members(root / path):
            if name == member:
                found = data
        if found is None:
            raise ValueError(f"source archive member missing: {member}")
        data = found
    else:
        data = (root / resource["path"]).read_bytes()
    if len(data) != resource["size"] or hashlib.sha256(data).hexdigest() != resource["sha256"]:
        raise ValueError("source member identity mismatch")
    return data


@contextmanager
def expanded(root: Path):
    """Keep the existing semantic reader/writer operating on ordinary paths."""
    for entry in root.rglob("*"):
        if entry.is_symlink() or not (entry.is_dir() or entry.is_file()):
            raise ValueError(f"Unified source must be regular, links forbidden: {entry}")
    archives = paths(root)
    if not archives:
        yield root
        return
    with tempfile.TemporaryDirectory(prefix="unified-versions-") as temporary:
        destination = Path(temporary) / root.name
        destination.mkdir()
        for path in root.iterdir():
            if path.name == DIRECTORY:
                continue
            if path.is_symlink():
                raise ValueError(f"Unified source link forbidden: {path}")
            if path.is_dir():
                for entry in path.rglob("*"):
                    if entry.is_symlink() or not (entry.is_dir() or entry.is_file()):
                        raise ValueError(f"Unified source must be regular: {entry}")
                shutil.copytree(path, destination / path.name)
            else:
                shutil.copyfile(path, destination / path.name)
        for archive in archives:
            for name, data in members(archive):
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    output.write(data)
        yield destination


def engine(implementation: str) -> str:
    source = impls.BATCH_SOURCE_IMPL.get(implementation, implementation)
    return next(spec.engine for spec in impls.IMPL_SPECS if spec.key == source)


def pack(root: Path, capture_pattern, version_key) -> None:
    """Pack an unpublished flat generation; the caller owns atomic installation."""
    if paths(root):
        raise ValueError("pack requires an expanded staging generation")
    histories = defaultdict(list)
    owners = defaultdict(set)
    live = set()
    manifests = {}
    # Retain metadata belonging to older, unreferenced manifests as evidence too.
    for path in sorted(root.glob("executions/*.yaml")):
        document = yaml.load(path.read_bytes(), Loader=yaml.CSafeLoader)
        name = path.relative_to(root).as_posix()
        group = f"{engine(document['implementation'])}-{document['version']}"
        dependencies = {execution_manifests.source_reference(reference, "receipts")
                        for reference in document.get("receipts", {})}
        manifests[name] = dependencies
        for dependency in dependencies | {name}:
            owners[dependency].add(group)
    for path in sorted(root.glob("families/*/*.yaml")):
        if path.name == "inputs_and_golden.yaml":
            continue
        identity = capture_pattern.fullmatch(path.stem)
        if identity is None:
            raise ValueError(f"invalid capture filename: {path}")
        implementation, version = identity["implementation"], identity["runtime_version"]
        group = f"{engine(implementation)}-{version}"
        name = path.relative_to(root).as_posix()
        document = yaml.load(path.read_bytes(), Loader=yaml.CSafeLoader)
        dependencies = {name}
        if reference := document.get("execution"):
            reference = execution_manifests.source_reference(reference, "executions")
            dependencies.add(reference)
            dependencies.update(manifests[reference])
        histories[path.parent.name, implementation].append((version, dependencies))
        for dependency in dependencies:
            owners[dependency].add(group)
    for history in histories.values():
        for _, dependencies in sorted(history, key=lambda item: version_key(item[0]))[-2:]:
            live.update(dependencies)
    groups = defaultdict(list)
    for path in sorted([*root.glob("families/*/*.yaml"),
                        *root.glob("executions/*.yaml"), *root.glob("receipts/*.yaml")]):
        name = path.relative_to(root).as_posix()
        if path.name == "inputs_and_golden.yaml" or name in live:
            continue
        groups_for_path = owners.get(name)
        if not groups_for_path:
            document = yaml.load(path.read_bytes(), Loader=yaml.CSafeLoader)
            if not document.get("implementation") or not document.get("version"):
                # An unreferenced receipt has no safe engine/version owner.
                # Keep it readable instead of dropping or relabelling evidence.
                continue
            groups_for_path = {f"{engine(document['implementation'])}-{document['version']}"}
        groups[sorted(groups_for_path)[0]].append(path)
    if not groups:
        return
    directory = root / DIRECTORY
    directory.mkdir(exist_ok=True)
    for group, sources in sorted(groups.items()):
        destination = directory / f"{group}.tar.gz"
        decoded_size = sum(512 + ((path.stat().st_size + 511) // 512) * 512 for path in sources) + 10240
        if decoded_size > MAX_DECODED:
            raise ValueError(f"version archive exceeds decoded limit: {group}")
        with destination.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=6) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                    for path in sources:
                        data = path.read_bytes()
                        if len(data) > MAX_MEMBER:
                            raise ValueError(f"Unified member exceeds limit: {path}")
                        info = tarfile.TarInfo(path.relative_to(root).as_posix())
                        info.size = len(data)
                        info.mode = 0o644
                        archive.addfile(info, io.BytesIO(data))
        if destination.stat().st_size > MAX_COMPRESSED:
            raise ValueError(f"version archive exceeds compressed limit: {group}")
        for path in sources:
            path.unlink()
