from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def wait_for_job(job_id: str, timeout_seconds: float = 5) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = client.get(f"/api/benchmarks/{job_id}").json()
        if result["status"] in {"complete", "error"}:
            return result
        time.sleep(0.05)
    return client.get(f"/api/benchmarks/{job_id}").json()


def test_layout_surface_17_counts() -> None:
    response = client.get("/api/layout", params={"distance": 3, "basis": "x"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["surface17"] is True
    assert payload["data_qubits"] == 9
    assert payload["measure_qubits"] == 8


def test_neural_decoders_report_checkpoint_required_or_dependency_missing() -> None:
    response = client.post(
        "/api/benchmarks",
        json={
            "distances": [3],
            "basis": "x",
            "noise": {"p": 0.001},
            "shots": 4,
            "seed": 1,
            "decoders": ["cnn", "gnn", "transformer"],
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    result = wait_for_job(job_id)
    assert result["status"] == "complete"
    statuses = {row["status"] for row in result["results"]}
    assert statuses <= {"checkpoint_required", "dependency_missing"}


def test_mwpm_zero_noise_when_quantum_stack_installed() -> None:
    pytest.importorskip("stim")
    pytest.importorskip("pymatching")

    response = client.post(
        "/api/benchmarks",
        json={
            "distances": [3],
            "basis": "x",
            "noise": {"p": 0},
            "shots": 20,
            "seed": 5,
            "decoders": ["mwpm"],
        },
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    result = wait_for_job(job_id)
    row = result["results"][0]
    assert row["status"] == "complete"
    assert row["logical_errors"] == 0
    assert row["logical_error_rate"] == 0
    assert row["runtime_ms"] is not None
    assert row["peak_memory_mb"] is not None
