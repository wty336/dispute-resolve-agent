import subprocess
import sys
import uuid
import json
from pathlib import Path


ROOT = Path(__file__).parents[3]


def _run_generator(output: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "scripts/generate_data.py",
            "--seed",
            "20260817",
            "--fixture-size",
            "24",
            "--output",
            str(output),
            *extra,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_freeze_detects_a_changed_test_file():
    output = ROOT / ".tmp_tests" / f"freeze-{uuid.uuid4().hex}"
    assert _run_generator(output).returncode == 0
    public_row = json.loads((output / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    hidden_row = json.loads(
        (output / "sft_train.ground_truth.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "ground_truth" not in public_row
    assert "fact_fingerprint" not in public_row["metadata"]
    assert hidden_row["case_id"] == public_row["case_id"]
    assert "true_liability" in hidden_row["ground_truth"]
    sft_rows = [
        json.loads(line)
        for line in (output / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    logistics_row = next(
        row
        for row in sft_rows
        if row["metadata"]["sft_category"] == "multi_tool"
        and any(
            call["function"]["name"] == "check_logistics"
            for message in row["messages"]
            for call in message.get("tool_calls", [])
        )
    )
    final_decision = json.loads(logistics_row["messages"][-1]["content"])
    assert any(evidence_id.startswith("logistics:") for evidence_id in final_decision["evidence_ids"])
    quality = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert quality["duplicate_fact_fingerprints"] == 0
    assert quality["trace_validation_errors"] == 0
    assert sum(quality["sft_category_counts"].values()) == manifest["counts"]["sft_train"]

    changed_path = output / "id_test.jsonl"
    original = changed_path.read_text(encoding="utf-8")
    with changed_path.open("a", encoding="utf-8") as handle:
        handle.write('{"tampered_before_freeze": true}\n')
    first_verification = _run_generator(output, "--freeze-test")
    assert first_verification.returncode != 0
    assert "FREEZE TEST FAILED" in first_verification.stdout

    changed_path.write_text(original, encoding="utf-8")
    assert _run_generator(output, "--freeze-test").returncode == 0

    with changed_path.open("a", encoding="utf-8") as handle:
        handle.write('{"tampered": true}\n')

    verification = _run_generator(output, "--freeze-test")
    assert verification.returncode != 0
    assert "FREEZE TEST FAILED" in verification.stdout
