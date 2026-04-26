from __future__ import annotations

import time
import tracemalloc
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .metrics import CONFIDENCE_LEVEL, CONFIDENCE_METHOD, wilson_interval
from .models import BenchmarkResult, DecoderName


class DecoderUnavailable(Exception):
    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(message)


class Decoder(ABC):
    name: DecoderName
    model_parameters: int | None = None

    @abstractmethod
    def decode_batch(self, *, circuit: Any, detection_events: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def run(
        self,
        *,
        circuit: Any,
        detection_events: np.ndarray,
        observable_flips: np.ndarray,
        distance: int,
        rounds: int,
        basis: str,
        noise_p: float,
        shots: int,
    ) -> BenchmarkResult:
        tracemalloc.start()
        start = time.perf_counter()
        predictions = self.decode_batch(circuit=circuit, detection_events=detection_events)
        elapsed_ms = (time.perf_counter() - start) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if predictions.ndim == 1:
            predictions = predictions.reshape((-1, 1))
        logical_misses = np.any(predictions != observable_flips, axis=1)
        logical_errors = int(np.count_nonzero(logical_misses))
        ci_low, ci_high = wilson_interval(logical_errors, shots)
        return BenchmarkResult(
            decoder=self.name,
            distance=distance,
            rounds=rounds,
            basis=basis,  # type: ignore[arg-type]
            noise_p=noise_p,
            shots=shots,
            status="complete",
            logical_error_rate=logical_errors / shots,
            logical_error_rate_ci_low=ci_low,
            logical_error_rate_ci_high=ci_high,
            confidence_level=CONFIDENCE_LEVEL,
            confidence_method=CONFIDENCE_METHOD,
            logical_errors=logical_errors,
            runtime_ms=elapsed_ms,
            runtime_us_per_shot=(elapsed_ms * 1000) / shots,
            peak_memory_mb=peak / (1024 * 1024),
            model_parameters=self.model_parameters,
        )


class MWPMDecoder(Decoder):
    name = DecoderName.mwpm

    def decode_batch(self, *, circuit: Any, detection_events: np.ndarray) -> np.ndarray:
        import pymatching  # type: ignore[import-not-found]

        dem = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        return matcher.decode_batch(detection_events)


class NeuralDecoder(Decoder):
    checkpoint_path: str | None = None
    architecture: str

    def decode_batch(self, *, circuit: Any, detection_events: np.ndarray) -> np.ndarray:
        raise DecoderUnavailable(
            "checkpoint_required",
            f"{self.architecture} decoder requires a trained checkpoint before evaluation.",
        )


class CNNDecoder(NeuralDecoder):
    name = DecoderName.cnn
    architecture = "CNN"


class GNNDecoder(NeuralDecoder):
    name = DecoderName.gnn
    architecture = "GNN"


class TransformerDecoder(NeuralDecoder):
    name = DecoderName.transformer
    architecture = "Transformer"


def make_decoder(name: DecoderName) -> Decoder:
    if name == DecoderName.mwpm:
        return MWPMDecoder()
    if name == DecoderName.cnn:
        return CNNDecoder()
    if name == DecoderName.gnn:
        return GNNDecoder()
    if name == DecoderName.transformer:
        return TransformerDecoder()
    raise ValueError(f"Unknown decoder: {name}")
