from __future__ import annotations

import hashlib
import json

import pytest

from testuary.generation_evidence import construction, source_sha256
from testuary.languages import TargetIdentity
from testuary.evidence import (
    append_jsonl,
    derivation_record,
    direct_record,
    record_binds_source,
    verify_delivered_archive,
    write_json_atomic,
)


def identity() -> TargetIdentity:
    source = "def value():\n    return 1\n"
    return TargetIdentity(
        source_path="/work/pkg/target.py",
        project_root="/work",
        module_name="pkg.target",
        language="python",
        policy_version="policy-v1",
        target_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        target_source=source,
    )


@pytest.mark.parametrize("source, digest", [
    ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ("def test_a():\n    assert True\n",
     "d64dd4bae169e8af6330efb864bf5fa7b825fa80e0c63c74c8e57d9a70f4fe42"),
    ("x = 1\r\ny = 2\r\n", "5bfee2d980860c58e9fe987d5845afaef0c892b330fdffce44eaa775d78d8e95"),
    ("s = '\u00e4\u20ac'\n", "5a6d26f5c8a836365bb53e11774c11ade7b968dc4e81739cabcae45e9aba54dd"),
    ("\ufeffx = 1\n", "ac05c7c476da9f4d0b14a6d051e7b1cf9ad2eda130563dba378dda764fb558cb"),
])
def test_every_record_hashes_the_source_text_s_utf8_bytes(source, digest) -> None:
    """A record's source hash is the SHA-256 of the text's UTF-8 bytes, its
    line ends and a leading byte order mark included, and a validity record
    and a generation record of one source carry the same digest."""
    assert source_sha256(source) == digest
    record = direct_record(identity=identity(), analyzed_source=source, stage="coverage.reduced")
    assert record["analyzed_source_sha256"] == record["final_clean_source_sha256"] == digest
    assert construction(source, phase="coverage")["selected_source_sha256"] == digest


def test_direct_record_binds_only_the_clean_source(tmp_path) -> None:
    item = identity()
    record = direct_record(
        identity=item,
        analyzed_source="def test_bad():\n    pass\n",
        removed_tests=["test_bad"],
        final_source="def test_ok():\n    assert True\n",
        stage="initial.suite_001",
    )
    assert record_binds_source(
        record, "def test_ok():\n    assert True\n", item,
    )
    assert not record_binds_source(record, "def test_bad():\n    pass\n", item)
    assert "t_mono_s" in record and "clock_domain_id" in record

    path = tmp_path / "validity.json"
    write_json_atomic(path, record)
    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_derivation_and_append_only_record(tmp_path) -> None:
    item = identity()
    record = derivation_record(
        identity=item,
        parent_source="def test_one():\n    pass\n",
        child_source="",
        operation="remove_tests",
        removed_tests=["test_one"],
        stage="coverage.reduced",
    )
    assert record_binds_source(record, "", item)

    path = tmp_path / "rejections.jsonl"
    append_jsonl(path, record)
    append_jsonl(path, record)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_reduced_delivered_suite_requires_complete_derivation_chain(tmp_path) -> None:
    item = identity()
    merged_source = "def test_one():\n    pass\n"
    reduced_source = "def test_two():\n    pass\n"
    write_json_atomic(
        tmp_path / "consolidation_record.json", {"delivered_from": "reduced"},
    )
    write_json_atomic(
        tmp_path / "validity_merged.json",
        direct_record(
            identity=item,
            analyzed_source=merged_source,
            stage="phase_end.merged",
        ),
    )
    write_json_atomic(
        tmp_path / "validity_derivations.json",
        {
            "records": [
                derivation_record(
                    identity=item,
                    parent_source=merged_source,
                    child_source=reduced_source,
                    operation="set_cover_remove_restore",
                    stage="phase_end.reduced",
                )
            ]
        },
    )
    write_json_atomic(
        tmp_path / "validity_delivered.json",
        direct_record(
            identity=item,
            analyzed_source=reduced_source,
            stage="phase_end.delivered_assertion",
        ),
    )

    assert verify_delivered_archive(
        archive_dir=tmp_path,
        source=reduced_source,
        identity=item,
    ) == (True, "")

    records = json.loads(
        (tmp_path / "validity_derivations.json").read_text(encoding="utf-8")
    )
    records["records"][0]["parent_source_sha256"] = "stale"
    write_json_atomic(tmp_path / "validity_derivations.json", records)
    valid, reason = verify_delivered_archive(
        archive_dir=tmp_path,
        source=reduced_source,
        identity=item,
    )
    assert not valid
    assert "breaks the hash chain" in reason


def test_nonreduced_delivered_suite_still_requires_its_delivered_record(tmp_path) -> None:
    item = identity()
    source = "def test_one():\n    pass\n"
    write_json_atomic(
        tmp_path / "consolidation_record.json", {"delivered_from": "agent"},
    )
    valid, reason = verify_delivered_archive(
        archive_dir=tmp_path,
        source=source,
        identity=item,
    )
    assert not valid
    assert "missing or unreadable" in reason
