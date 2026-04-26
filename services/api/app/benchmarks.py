from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .decoders import DecoderUnavailable, make_decoder
from .dependencies import missing_quantum_dependencies
from .models import (
    API_VERSION,
    BENCHMARK_SUITE_ID,
    BENCHMARK_SUITE_VERSION,
    SUBMISSION_SCHEMA_VERSION,
    BenchmarkJobResponse,
    BenchmarkRequest,
    BenchmarkResult,
    BenchmarkSubmissionBundle,
    DecoderName,
    EncodedTraceArray,
    RuntimeEnvironment,
    SubmissionValidationResponse,
    TraceArtifact,
)
from .simulator import build_surface_code_circuit, detector_sampler


CANONICAL_DISTANCES = [3, 5, 7]
REQUIRED_BASELINE = DecoderName.mwpm


def collect_runtime_environment() -> RuntimeEnvironment:
    dependencies: dict[str, str | None] = {}
    for package in ("stim", "pymatching", "numpy", "fastapi", "pydantic"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None

    return RuntimeEnvironment(
        api_version=API_VERSION,
        python=sys.version.split()[0],
        platform=platform.system(),
        platform_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count(),
        dependencies=dependencies,
    )


def suite_compliance_errors(request: BenchmarkRequest) -> list[str]:
    errors: list[str] = []
    if request.suite_id != BENCHMARK_SUITE_ID:
        errors.append(f"suite_id must be {BENCHMARK_SUITE_ID}.")
    if request.suite_version != BENCHMARK_SUITE_VERSION:
        errors.append(f"suite_version must be {BENCHMARK_SUITE_VERSION}.")
    if request.distances != CANONICAL_DISTANCES:
        errors.append("distances must be [3, 5, 7].")
    if request.rounds is not None:
        errors.append("rounds must be omitted so each case uses rounds = d.")
    if request.basis != "x":
        errors.append("basis must be memory X.")
    if request.seed != 1337:
        errors.append("seed must be 1337.")
    if REQUIRED_BASELINE not in request.decoders:
        errors.append("MWPM baseline decoder is required.")
    return errors


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def encode_array(array: np.ndarray) -> EncodedTraceArray:
    contiguous = np.ascontiguousarray(array)
    return EncodedTraceArray(
        dtype=str(contiguous.dtype),
        shape=[int(size) for size in contiguous.shape],
        data_b64=base64.b64encode(contiguous.tobytes()).decode("ascii"),
    )


def hash_encoded_array(array: EncodedTraceArray | dict[str, Any]) -> str:
    if isinstance(array, EncodedTraceArray):
        payload = array.model_dump()
    else:
        payload = {
            "dtype": array.get("dtype"),
            "shape": array.get("shape"),
            "data_b64": array.get("data_b64"),
        }
    return sha256_text(canonical_json(payload))


def noise_hash(request: BenchmarkRequest) -> str:
    payload = {"p": request.noise.p, "resolved": request.noise.resolved()}
    return sha256_text(canonical_json(payload))[:12]


def case_id_for(request: BenchmarkRequest, distance: int, rounds: int) -> str:
    return (
        f"{request.suite_id}:{request.suite_version}:"
        f"d{distance}:r{rounds}:b{request.basis}:n{noise_hash(request)}"
    )


def trace_id_for(
    *,
    case_id: str,
    sample_seed: int | None,
    circuit_sha256: str,
    detection_events_sha256: str,
    observable_flips_sha256: str,
) -> str:
    return sha256_text(
        canonical_json(
            {
                "case_id": case_id,
                "sample_seed": sample_seed,
                "circuit_sha256": circuit_sha256,
                "detection_events_sha256": detection_events_sha256,
                "observable_flips_sha256": observable_flips_sha256,
            }
        )
    )


def build_trace_artifact(
    *,
    case_id: str,
    sample_seed: int | None,
    circuit: Any,
    detection_events: np.ndarray,
    observable_flips: np.ndarray,
) -> TraceArtifact:
    circuit_text = str(circuit)
    encoded_detection_events = encode_array(detection_events)
    encoded_observable_flips = encode_array(observable_flips)
    return TraceArtifact(
        case_id=case_id,
        sample_seed=sample_seed,
        circuit_sha256=sha256_text(circuit_text),
        detection_events_sha256=hash_encoded_array(encoded_detection_events),
        observable_flips_sha256=hash_encoded_array(encoded_observable_flips),
        circuit_text=circuit_text,
        detection_events=encoded_detection_events,
        observable_flips=encoded_observable_flips,
    )


@dataclass
class BenchmarkJob:
    job_id: str
    request: BenchmarkRequest
    runtime_environment: RuntimeEnvironment = field(default_factory=collect_runtime_environment)
    suite_compliance_errors: list[str] = field(default_factory=list)
    status: str = "pending"
    progress: float = 0.0
    results: list[BenchmarkResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    traces: dict[str, TraceArtifact] = field(default_factory=dict)

    def response(self) -> BenchmarkJobResponse:
        return BenchmarkJobResponse(
            job_id=self.job_id,
            suite_id=self.request.suite_id,
            suite_version=self.request.suite_version,
            suite_compliant=not self.suite_compliance_errors,
            suite_compliance_errors=self.suite_compliance_errors,
            runtime_environment=self.runtime_environment,
            status=self.status,  # type: ignore[arg-type]
            progress=self.progress,
            results=self.results,
            errors=self.errors,
        )


jobs: dict[str, BenchmarkJob] = {}


def create_job(request: BenchmarkRequest) -> BenchmarkJob:
    job = BenchmarkJob(
        job_id=str(uuid.uuid4()),
        request=request,
        suite_compliance_errors=suite_compliance_errors(request),
    )
    jobs[job.job_id] = job
    return job


def get_job(job_id: str) -> BenchmarkJob | None:
    return jobs.get(job_id)


def unavailable_result(
    *,
    job: BenchmarkJob,
    decoder: DecoderName,
    distance: int,
    rounds: int,
    case_id: str | None,
    sample_seed: int | None,
    trace: TraceArtifact | None,
    status: str,
    error: str,
) -> BenchmarkResult:
    result = BenchmarkResult(
        decoder=decoder,
        suite_id=job.request.suite_id,
        suite_version=job.request.suite_version,
        suite_compliant=not job.suite_compliance_errors,
        distance=distance,
        rounds=rounds,
        basis=job.request.basis,
        noise_p=job.request.noise.p,
        shots=job.request.shots,
        case_id=case_id,
        sample_seed=sample_seed,
        status=status,  # type: ignore[arg-type]
        error=error,
    )
    attach_trace_metadata(result, trace)
    return result


def attach_trace_metadata(result: BenchmarkResult, trace: TraceArtifact | None) -> None:
    if trace is None:
        return
    result.case_id = trace.case_id
    result.sample_seed = trace.sample_seed
    result.circuit_sha256 = trace.circuit_sha256
    result.detection_events_sha256 = trace.detection_events_sha256
    result.observable_flips_sha256 = trace.observable_flips_sha256
    result.trace_id = trace_id_for(
        case_id=trace.case_id,
        sample_seed=trace.sample_seed,
        circuit_sha256=trace.circuit_sha256,
        detection_events_sha256=trace.detection_events_sha256,
        observable_flips_sha256=trace.observable_flips_sha256,
    )


def attach_suite_metadata(job: BenchmarkJob, result: BenchmarkResult) -> None:
    result.suite_id = job.request.suite_id
    result.suite_version = job.request.suite_version
    result.suite_compliant = not job.suite_compliance_errors


def run_job(job_id: str) -> None:
    job = jobs[job_id]
    request = job.request
    job.status = "running"
    total_work = len(request.distances) * len(request.decoders)
    completed = 0

    try:
        quantum_missing = missing_quantum_dependencies()
        for distance in request.distances:
            rounds = request.rounds or distance
            case_id = case_id_for(request, distance, rounds)
            sample_seed = None if request.seed is None else request.seed + distance * 1000
            circuit = None
            detection_events = None
            observable_flips = None
            trace: TraceArtifact | None = None
            circuit_error: str | None = None

            if quantum_missing:
                circuit_error = f"Missing quantum dependencies: {', '.join(quantum_missing)}"
            else:
                try:
                    circuit = build_surface_code_circuit(
                        distance=distance,
                        rounds=rounds,
                        basis=request.basis,
                        noise=request.noise,
                    )
                    sampler = detector_sampler(circuit, sample_seed)
                    detection_events, observable_flips = sampler.sample(
                        request.shots, separate_observables=True
                    )
                    trace = build_trace_artifact(
                        case_id=case_id,
                        sample_seed=sample_seed,
                        circuit=circuit,
                        detection_events=detection_events,
                        observable_flips=observable_flips,
                    )
                    job.traces[case_id] = trace
                except Exception as exc:
                    circuit_error = str(exc)

            for decoder_name in request.decoders:
                if circuit_error is not None:
                    job.results.append(
                        unavailable_result(
                            job=job,
                            decoder=decoder_name,
                            distance=distance,
                            rounds=rounds,
                            case_id=case_id,
                            sample_seed=sample_seed,
                            trace=trace,
                            status="dependency_missing" if quantum_missing else "error",
                            error=circuit_error,
                        )
                    )
                else:
                    decoder = make_decoder(decoder_name)
                    try:
                        result = decoder.run(
                            circuit=circuit,
                            detection_events=detection_events,
                            observable_flips=observable_flips,
                            distance=distance,
                            rounds=rounds,
                            basis=request.basis,
                            noise_p=request.noise.p,
                            shots=request.shots,
                        )
                        attach_suite_metadata(job, result)
                        attach_trace_metadata(result, trace)
                        job.results.append(result)
                    except DecoderUnavailable as exc:
                        job.results.append(
                            unavailable_result(
                                job=job,
                                decoder=decoder_name,
                                distance=distance,
                                rounds=rounds,
                                case_id=case_id,
                                sample_seed=sample_seed,
                                trace=trace,
                                status=exc.status,
                                error=str(exc),
                            )
                        )
                    except Exception as exc:
                        job.results.append(
                            unavailable_result(
                                job=job,
                                decoder=decoder_name,
                                distance=distance,
                                rounds=rounds,
                                case_id=case_id,
                                sample_seed=sample_seed,
                                trace=trace,
                                status="error",
                                error=str(exc),
                            )
                        )

                completed += 1
                job.progress = completed / total_work

        job.status = "complete"
        job.progress = 1.0
    except Exception as exc:
        job.status = "error"
        job.errors.append(str(exc))


def build_submission(job: BenchmarkJob) -> BenchmarkSubmissionBundle:
    return BenchmarkSubmissionBundle(
        suite_id=job.request.suite_id,
        suite_version=job.request.suite_version,
        suite_compliant=not job.suite_compliance_errors,
        suite_compliance_errors=job.suite_compliance_errors,
        job_id=job.job_id,
        request=job.request,
        runtime_environment=job.runtime_environment,
        results=job.results,
        traces=list(job.traces.values()),
    )


def validate_submission_payload(payload: dict[str, Any]) -> SubmissionValidationResponse:
    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("schema_version") != SUBMISSION_SCHEMA_VERSION:
        errors.append(f"schema_version must be {SUBMISSION_SCHEMA_VERSION}.")
    if payload.get("suite_id") != BENCHMARK_SUITE_ID:
        errors.append(f"suite_id must be {BENCHMARK_SUITE_ID}.")
    if payload.get("suite_version") != BENCHMARK_SUITE_VERSION:
        errors.append(f"suite_version must be {BENCHMARK_SUITE_VERSION}.")

    request = payload.get("request")
    if not isinstance(request, dict):
        errors.append("request is required.")
        request = {}
    if request.get("seed") is None:
        errors.append("request.seed is required for reproducibility.")
    if request.get("distances") != CANONICAL_DISTANCES:
        errors.append("request.distances must be [3, 5, 7].")
    if request.get("rounds") is not None:
        errors.append("request.rounds must be null for rounds = d.")
    if request.get("basis") != "x":
        errors.append("request.basis must be x.")
    if "mwpm" not in request.get("decoders", []):
        errors.append("request.decoders must include mwpm baseline.")

    runtime_environment = payload.get("runtime_environment")
    if not isinstance(runtime_environment, dict):
        errors.append("runtime_environment is required.")
        runtime_environment = {}
    for key in ("api_version", "python", "platform", "machine", "dependencies"):
        if not runtime_environment.get(key):
            errors.append(f"runtime_environment.{key} is required.")

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        errors.append("results must contain at least one benchmark row.")
        results = []

    traces = payload.get("traces")
    if not isinstance(traces, list):
        errors.append("traces must be a list.")
        traces = []
    traces_by_case = {
        trace.get("case_id"): trace for trace in traces if isinstance(trace, dict) and trace.get("case_id")
    }

    complete_mwpm = 0
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            errors.append(f"results[{index}] must be an object.")
            continue

        if result.get("status") != "complete":
            continue

        if result.get("decoder") == "mwpm":
            complete_mwpm += 1

        for key in (
            "logical_error_rate_ci_low",
            "logical_error_rate_ci_high",
            "confidence_level",
            "confidence_method",
            "case_id",
            "sample_seed",
            "trace_id",
            "circuit_sha256",
            "detection_events_sha256",
            "observable_flips_sha256",
        ):
            if result.get(key) is None:
                errors.append(f"results[{index}].{key} is required for complete rows.")

        case_id = result.get("case_id")
        trace = traces_by_case.get(case_id)
        if not isinstance(trace, dict):
            errors.append(f"results[{index}] is missing trace artifact for {case_id}.")
            continue

        validate_trace_hashes(result, trace, index, errors)

    if complete_mwpm == 0:
        errors.append("at least one complete MWPM baseline row is required.")

    if not payload.get("suite_compliant", False):
        warnings.append("submission is schema-valid but not canonical-suite compliant.")
    if payload.get("suite_compliance_errors"):
        warnings.extend(str(error) for error in payload["suite_compliance_errors"])

    valid = not errors
    leaderboard_eligible = valid and payload.get("suite_compliant", False) is True
    return SubmissionValidationResponse(
        valid=valid,
        leaderboard_eligible=leaderboard_eligible,
        warnings=warnings,
        errors=errors,
    )


def validate_trace_hashes(
    result: dict[str, Any], trace: dict[str, Any], result_index: int, errors: list[str]
) -> None:
    circuit_text = trace.get("circuit_text")
    detection_events = trace.get("detection_events")
    observable_flips = trace.get("observable_flips")

    if not isinstance(circuit_text, str):
        errors.append(f"traces[{result_index}].circuit_text is required.")
        return
    if not validate_encoded_array_payload(
        detection_events, f"traces[{result_index}].detection_events", errors
    ):
        return
    if not validate_encoded_array_payload(
        observable_flips, f"traces[{result_index}].observable_flips", errors
    ):
        return

    circuit_sha256 = sha256_text(circuit_text)
    detection_events_sha256 = hash_encoded_array(detection_events)
    observable_flips_sha256 = hash_encoded_array(observable_flips)
    trace_id = trace_id_for(
        case_id=str(trace.get("case_id")),
        sample_seed=trace.get("sample_seed"),
        circuit_sha256=circuit_sha256,
        detection_events_sha256=detection_events_sha256,
        observable_flips_sha256=observable_flips_sha256,
    )

    expected = {
        "circuit_sha256": circuit_sha256,
        "detection_events_sha256": detection_events_sha256,
        "observable_flips_sha256": observable_flips_sha256,
        "trace_id": trace_id,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            errors.append(f"results[{result_index}].{key} does not match trace artifact.")


def validate_encoded_array_payload(
    array: object, label: str, errors: list[str]
) -> bool:
    if not isinstance(array, dict):
        errors.append(f"{label} must be an object.")
        return False

    dtype = array.get("dtype")
    shape = array.get("shape")
    data_b64 = array.get("data_b64")
    if not isinstance(dtype, str) or not dtype:
        errors.append(f"{label}.dtype is required.")
        return False
    if (
        not isinstance(shape, list)
        or not shape
        or not all(isinstance(size, int) and size >= 0 for size in shape)
    ):
        errors.append(f"{label}.shape must be a list of non-negative integers.")
        return False
    if not isinstance(data_b64, str) or not data_b64:
        errors.append(f"{label}.data_b64 is required.")
        return False

    try:
        raw = base64.b64decode(data_b64, validate=True)
        expected_bytes = math.prod(shape) * np.dtype(dtype).itemsize
    except Exception:
        errors.append(f"{label} is not a valid encoded NumPy array.")
        return False

    if len(raw) != expected_bytes:
        errors.append(f"{label}.data_b64 length does not match dtype and shape.")
        return False
    return True
