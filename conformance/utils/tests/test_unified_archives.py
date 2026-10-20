# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Version archives preserve source bytes and the existing history contracts."""

import gzip
import io
import shutil
import tarfile

import pytest

import unified_archives
import unified_history
import gen_unified_v1_cases
from viewer_catalog import _unified_records
from test_unified_history import _change, _store, _write_capture


def test_archive_keeps_latest_two_and_exact_history_bytes(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.3", {"text_only": _change("new")})
    before = unified_history.load_store(root)
    inventory = unified_history.store_inventory(root)
    original = (root / "families/gemma4/dynamo_v2-0.5.0.yaml").read_bytes()
    result = unified_history.archive_versions(root)
    assert result["archives"] == 1
    assert result["logical_documents"] == len(inventory)
    archive = root / "versions/dynamo-0.5.0.tar.gz"
    assert list(unified_archives.members(archive)) == [("families/gemma4/dynamo_v2-0.5.0.yaml", original)]
    assert not (root / "families/gemma4/dynamo_v2-0.5.0.yaml").exists()
    assert (root / "families/gemma4/dynamo_v2-0.5.2.yaml").is_file()
    assert (root / "families/gemma4/dynamo_v2-0.5.3.yaml").is_file()
    assert unified_history.store_inventory(root) == inventory
    after = unified_history.load_store(root)
    unified_history._validate_store_parity(before, after)
    assert after.root == root.resolve()
    assert after.families["gemma4"].path == root / "families/gemma4/inputs_and_golden.yaml"
    records = _unified_records(unified=root, store=after)
    archived = next(row for row in records if row.get("archive"))
    assert archived["archive"]["path"] == "conformance/fixtures-unified-v2/versions/dynamo-0.5.0.tar.gz"
    assert archived["sha256"] != archived["archive"]["sha256"]
    compressed = archive.read_bytes()
    assert unified_history.archive_versions(root) == result
    assert archive.read_bytes() == compressed
    # Existing mutations must preserve the reduced layout and rotate retention.
    unified_history.rewrite_store(root)
    unified_history._validate_store_parity(after, unified_history.load_store(root))
    assert archive.is_file()
    assert not (root / "families/gemma4/dynamo_v2-0.5.0.yaml").exists()


def test_archived_mutation_keeps_new_latest_receipt_readable(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.3", {})
    unified_history.compact_execution_store(root)
    unified_history.archive_versions(root)

    def add_receipt(store):
        capture = store.histories["gemma4", "dynamo_v2"].captures["dynamo_v2-0.5.3"]
        capture["metadata_changes"] = {"text_only": {"verified_recaptures": [{
            "provenance": {"job_sha256": "a" * 64, "adapter_sha256": "b" * 64, "source_sha256": "c" * 64, "version": "0.5.3"}, "result": {"old": None}}]}}

    unified_history._mutate_store(root, add_receipt, compact_executions=True)
    store = unified_history.load_store(root)
    manifest = store.resources[store.resources["families/gemma4/dynamo_v2-0.5.3.yaml"]["execution"]]
    assert manifest["receipts"]
    assert all((root / reference).is_file() for reference in manifest["receipts"])
    assert unified_history.compact_execution_store(root) == []


def test_engine_version_archive_groups_families(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.3", {})
    family = root / "families/gemma4"
    other = root / "families/qwen3"
    shutil.copytree(family, other)
    # Packing is byte-preserving and independent of the semantic family reader.
    unified_archives.pack(root, unified_history.CAPTURE_DIRECTORY_RE, unified_history._capture_release_sort_key)
    assert len(unified_archives.paths(root)) == 1
    assert {name for name, _ in unified_archives.members(unified_archives.paths(root)[0])} == {
        "families/gemma4/dynamo_v2-0.5.0.yaml", "families/qwen3/dynamo_v2-0.5.0.yaml"}


@pytest.mark.parametrize("name,kind", [("../escape.yaml", tarfile.REGTYPE),
                                     ("receipts/link.yaml", tarfile.SYMTYPE),
                                     ("families/gemma4/inputs_and_golden.yaml", tarfile.REGTYPE)])
def test_archives_reject_unsafe_members(tmp_path, name, kind):
    archive = tmp_path / "dynamo-0.1.0.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.size = 2
        output.addfile(member, io.BytesIO(b"{}"))
    with pytest.raises(ValueError, match="invalid"):
        list(unified_archives.members(archive))


def test_duplicate_loose_and_archived_source_is_rejected(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.3", {})
    path = root / "families/gemma4/dynamo_v2-0.5.0.yaml"
    original = path.read_bytes()
    unified_history.archive_versions(root)
    path.write_bytes(original)
    with pytest.raises(ValueError, match="duplicate"):
        unified_history.store_inventory(root)
    with pytest.raises(FileExistsError):
        unified_history.load_store(root)


def test_corrupt_archive_is_not_hidden_by_loose_files(tmp_path):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.3", {})
    unified_history.archive_versions(root)
    archive = unified_archives.paths(root)[0]
    archive.write_bytes(archive.read_bytes()[:-5])
    with pytest.raises((EOFError, gzip.BadGzipFile, ValueError)):
        unified_history.load_store(root)


@pytest.mark.parametrize("tail", [b"garbage", gzip.compress(b""), gzip.compress(b"\0" * 1024)])
def test_archives_reject_extra_gzip_members_and_bytes(tmp_path, tail):
    root = _store(tmp_path / "store")
    _write_capture(root, "0.5.3", {})
    unified_history.archive_versions(root)
    archive = unified_archives.paths(root)[0]
    archive.write_bytes(archive.read_bytes() + tail)
    with pytest.raises(ValueError, match="trailing"):
        unified_history.load_store(root)


def test_two_capture_store_remains_readable_without_empty_versions(tmp_path):
    root = _store(tmp_path / "store")
    before = unified_history.load_store(root)
    assert unified_history.archive_versions(root)["archives"] == 0
    assert not (root / "versions").exists()
    unified_history._validate_store_parity(before, unified_history.load_store(root))


def test_historical_refresh_preserves_archived_receipts_after_expanding_checkpoint(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    root = _store(repo / "conformance/fixtures-unified-v2")
    _write_capture(root, "0.5.3", {})
    path = root / "families/gemma4/dynamo_v2-0.5.0.yaml"
    capture = unified_history.load_yaml(path)
    capture["metadata_changes"] = {"text_only": {"verified_recaptures": [{"provenance": {
        "job_sha256": "a" * 64, "adapter_sha256": "b" * 64, "version": "0.5.0"}, "result": {"old": None}}]}}
    path.write_text(unified_history.dump_yaml(capture))
    unified_history.compact_execution_store(root)
    unified_history.archive_versions(root)
    before = {row["path"]: row for row in unified_history.store_inventory(root)}

    def generated(_repo, source):
        relative = path.relative_to(root)
        existing = unified_history.load_yaml(source / relative)
        family = unified_history._validate_family(source / "families/gemma4/inputs_and_golden.yaml",
                                                  unified_history.load_yaml(source / "families/gemma4/inputs_and_golden.yaml"))
        _, _, _, expanded = unified_history._validate_capture_file(source / relative, existing, {"gemma4": family})
        addition = {"changes": {}, "metadata_changes": {"text_only": {
            **expanded["metadata_changes"]["text_only"], "new": None}}, "document_overrides": {}}
        return {path: gen_unified_v1_cases._merge_generated_case(source, relative, unified_history.dump_yaml(addition))}

    monkeypatch.setattr(gen_unified_v1_cases, "_generated_documents", generated)
    assert gen_unified_v1_cases.refresh(repo)
    after = {row["path"]: row for row in unified_history.store_inventory(root)}
    assert before.keys() == after.keys()
    assert all(before[name] == row for name, row in after.items() if name != path.relative_to(root).as_posix())
    store = unified_history.load_store(root)
    record = store.histories["gemma4", "dynamo_v2"].resolve("dynamo_v2-0.5.0")["text_only"]
    assert record["document"]["record_metadata"]["new"] is None
    assert gen_unified_v1_cases.refresh(repo, check=True)
    # Catalog fallback must also locate manifests no longer referenced by a checkpoint.
    rows = _unified_records(unified=root, store=store)
    assert len(rows) == len(after)
    assert all("receipts" in row for row in rows if row["kind"] == "execution")


def test_archiving_rejects_linked_family_directory(tmp_path):
    root = _store(tmp_path / "store")
    original = root / "families/gemma4"
    outside = tmp_path / "outside"
    original.rename(outside)
    original.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="links forbidden"):
        unified_history.archive_versions(root)
    assert original.is_symlink()
    assert not (root / "versions").exists()
