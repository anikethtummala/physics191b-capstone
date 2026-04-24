from __future__ import annotations

from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .benchmarks import create_job, get_job, run_job
from .dependencies import missing_quantum_dependencies
from .layout import build_rotated_layout
from .models import (
    Basis,
    BenchmarkJobResponse,
    BenchmarkRequest,
    BenchmarkStartResponse,
    LayoutResponse,
    SampleRequest,
    SampleResponse,
)
from .simulator import sample_syndrome


app = FastAPI(title="Surface Code Benchmark API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, object]:
    missing = missing_quantum_dependencies()
    return {"ok": True, "quantum_stack": len(missing) == 0, "missing": missing}


@app.post("/api/benchmarks", response_model=BenchmarkStartResponse)
def start_benchmark(request: BenchmarkRequest) -> BenchmarkStartResponse:
    job = create_job(request)
    Thread(target=run_job, args=(job.job_id,), daemon=True).start()
    return BenchmarkStartResponse(job_id=job.job_id)


@app.get("/api/benchmarks/{job_id}", response_model=BenchmarkJobResponse)
def read_benchmark(job_id: str) -> BenchmarkJobResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Benchmark job not found.")
    return job.response()


@app.get("/api/layout", response_model=LayoutResponse)
def read_layout(distance: int = 3, basis: Basis = "x") -> LayoutResponse:
    try:
        return build_rotated_layout(distance, basis)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/sample", response_model=SampleResponse)
def create_sample(request: SampleRequest) -> SampleResponse:
    rounds = request.rounds or request.distance
    return sample_syndrome(
        distance=request.distance,
        rounds=rounds,
        basis=request.basis,
        noise=request.noise,
        seed=request.seed,
    )
