from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .decoders import DecoderUnavailable, make_decoder
from .dependencies import missing_quantum_dependencies
from .models import BenchmarkJobResponse, BenchmarkRequest, BenchmarkResult, DecoderName
from .simulator import build_surface_code_circuit, detector_sampler


@dataclass
class BenchmarkJob:
    job_id: str
    request: BenchmarkRequest
    status: str = "pending"
    progress: float = 0.0
    results: list[BenchmarkResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def response(self) -> BenchmarkJobResponse:
        return BenchmarkJobResponse(
            job_id=self.job_id,
            status=self.status,  # type: ignore[arg-type]
            progress=self.progress,
            results=self.results,
            errors=self.errors,
        )


jobs: dict[str, BenchmarkJob] = {}


def create_job(request: BenchmarkRequest) -> BenchmarkJob:
    job = BenchmarkJob(job_id=str(uuid.uuid4()), request=request)
    jobs[job.job_id] = job
    return job


def get_job(job_id: str) -> BenchmarkJob | None:
    return jobs.get(job_id)


def unavailable_result(
    *,
    decoder: DecoderName,
    distance: int,
    rounds: int,
    request: BenchmarkRequest,
    status: str,
    error: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        decoder=decoder,
        distance=distance,
        rounds=rounds,
        basis=request.basis,
        noise_p=request.noise.p,
        shots=request.shots,
        status=status,  # type: ignore[arg-type]
        error=error,
    )


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
            circuit = None
            detection_events = None
            observable_flips = None
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
                    seed = None if request.seed is None else request.seed + distance * 1000
                    sampler = detector_sampler(circuit, seed)
                    detection_events, observable_flips = sampler.sample(
                        request.shots, separate_observables=True
                    )
                except Exception as exc:
                    circuit_error = str(exc)

            for decoder_name in request.decoders:
                if circuit_error is not None:
                    job.results.append(
                        unavailable_result(
                            decoder=decoder_name,
                            distance=distance,
                            rounds=rounds,
                            request=request,
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
                        job.results.append(result)
                    except DecoderUnavailable as exc:
                        job.results.append(
                            unavailable_result(
                                decoder=decoder_name,
                                distance=distance,
                                rounds=rounds,
                                request=request,
                                status=exc.status,
                                error=str(exc),
                            )
                        )
                    except Exception as exc:
                        job.results.append(
                            unavailable_result(
                                decoder=decoder_name,
                                distance=distance,
                                rounds=rounds,
                                request=request,
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
