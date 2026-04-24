from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


Basis = Literal["x", "z"]


class DecoderName(str, Enum):
    mwpm = "mwpm"
    cnn = "cnn"
    gnn = "gnn"
    transformer = "transformer"


class NoiseSettings(BaseModel):
    p: float = Field(default=0.001, ge=0, le=0.25)
    after_clifford_depolarization: float | None = Field(default=None, ge=0, le=0.25)
    before_round_data_depolarization: float | None = Field(default=None, ge=0, le=0.25)
    before_measure_flip_probability: float | None = Field(default=None, ge=0, le=0.5)
    after_reset_flip_probability: float | None = Field(default=None, ge=0, le=0.5)

    def resolved(self) -> dict[str, float]:
        return {
            "after_clifford_depolarization": (
                self.after_clifford_depolarization
                if self.after_clifford_depolarization is not None
                else self.p
            ),
            "before_round_data_depolarization": (
                self.before_round_data_depolarization
                if self.before_round_data_depolarization is not None
                else self.p
            ),
            "before_measure_flip_probability": (
                self.before_measure_flip_probability
                if self.before_measure_flip_probability is not None
                else self.p
            ),
            "after_reset_flip_probability": (
                self.after_reset_flip_probability
                if self.after_reset_flip_probability is not None
                else self.p
            ),
        }


class BenchmarkRequest(BaseModel):
    distances: list[int] = Field(default_factory=lambda: [3, 5, 7])
    rounds: int | None = Field(default=None, ge=1, le=25)
    basis: Basis = "x"
    noise: NoiseSettings = Field(default_factory=NoiseSettings)
    shots: int = Field(default=1000, ge=1, le=50000)
    seed: int | None = Field(default=1337, ge=0)
    decoders: list[DecoderName] = Field(default_factory=lambda: [DecoderName.mwpm])

    @field_validator("distances")
    @classmethod
    def validate_distances(cls, distances: list[int]) -> list[int]:
        if not distances:
            raise ValueError("At least one distance is required.")
        unique_distances = sorted(set(distances))
        for distance in unique_distances:
            if distance < 3 or distance > 7:
                raise ValueError("Distances must be in the range 3..7.")
            if distance % 2 == 0:
                raise ValueError("Only odd distances are supported in the first version.")
        return unique_distances

    @field_validator("decoders")
    @classmethod
    def validate_decoders(cls, decoders: list[DecoderName]) -> list[DecoderName]:
        if not decoders:
            raise ValueError("At least one decoder is required.")
        return list(dict.fromkeys(decoders))


class BenchmarkStartResponse(BaseModel):
    job_id: str


class BenchmarkResult(BaseModel):
    decoder: DecoderName
    distance: int
    rounds: int
    basis: Basis
    noise_p: float
    shots: int
    status: Literal["complete", "checkpoint_required", "dependency_missing", "error"]
    logical_error_rate: float | None = None
    logical_errors: int | None = None
    runtime_ms: float | None = None
    runtime_us_per_shot: float | None = None
    peak_memory_mb: float | None = None
    model_parameters: int | None = None
    error: str | None = None


class BenchmarkJobResponse(BaseModel):
    job_id: str
    status: Literal["pending", "running", "complete", "error"]
    progress: float = Field(ge=0, le=1)
    results: list[BenchmarkResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Qubit(BaseModel):
    id: str
    index: int
    kind: Literal["data", "measure"]
    x: float
    y: float
    stabilizer_type: Literal["x", "z"] | None = None
    boundary: bool = False


class Edge(BaseModel):
    source: str
    target: str
    kind: Literal["stabilizer", "logical"]


class LayoutResponse(BaseModel):
    distance: int
    basis: Basis
    surface17: bool
    data_qubits: int
    measure_qubits: int
    bounds: dict[str, float]
    qubits: list[Qubit]
    edges: list[Edge]


class SampleRequest(BaseModel):
    distance: int = Field(default=3, ge=3, le=7)
    rounds: int | None = Field(default=None, ge=1, le=25)
    basis: Basis = "x"
    noise: NoiseSettings = Field(default_factory=NoiseSettings)
    seed: int | None = Field(default=1337, ge=0)

    @model_validator(mode="after")
    def validate_distance(self) -> "SampleRequest":
        if self.distance % 2 == 0:
            raise ValueError("Only odd distances are supported in the first version.")
        return self


class SyndromeEvent(BaseModel):
    detector: int
    x: float
    y: float
    t: float
    stabilizer_type: Literal["x", "z"] | None = None


class MatchEdge(BaseModel):
    source: int
    target: int | None = None
    boundary: bool = False


class SampleResponse(BaseModel):
    distance: int
    rounds: int
    basis: Basis
    noise_p: float
    logical_observable_flip: bool
    mwpm_prediction: bool | None = None
    using_fallback: bool = False
    events: list[SyndromeEvent]
    matches: list[MatchEdge] = Field(default_factory=list)
    error: str | None = None
