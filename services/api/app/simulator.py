from __future__ import annotations

import random
import os
import tempfile
from typing import Any

from .dependencies import missing_quantum_dependencies
from .layout import build_rotated_layout
from .models import Basis, MatchEdge, NoiseSettings, SampleResponse, SyndromeEvent


def _import_quantum_stack() -> tuple[Any, Any]:
    missing = missing_quantum_dependencies()
    if missing:
        raise RuntimeError(f"Missing quantum dependencies: {', '.join(missing)}")
    os.environ.setdefault(
        "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "surface-code-benchmark-mpl")
    )
    import pymatching  # type: ignore[import-not-found]
    import stim  # type: ignore[import-not-found]

    return stim, pymatching


def build_surface_code_circuit(
    *,
    distance: int,
    rounds: int,
    basis: Basis,
    noise: NoiseSettings,
) -> Any:
    stim, _ = _import_quantum_stack()
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=rounds,
        **noise.resolved(),
    )


def detector_sampler(circuit: Any, seed: int | None = None) -> Any:
    try:
        return circuit.compile_detector_sampler(seed=seed)
    except TypeError:
        return circuit.compile_detector_sampler()


def detector_coordinates(circuit: Any) -> dict[int, list[float]]:
    try:
        raw = circuit.get_detector_coordinates()
    except AttributeError:
        return {}
    return {int(key): [float(v) for v in value] for key, value in raw.items()}


def fallback_sample(
    *, distance: int, rounds: int, basis: Basis, noise: NoiseSettings, seed: int | None, error: str
) -> SampleResponse:
    rng = random.Random(seed)
    layout = build_rotated_layout(distance, basis)
    measure_qubits = [q for q in layout.qubits if q.kind == "measure"]
    event_probability = min(max(noise.p * 12, 0.03), 0.35)
    events: list[SyndromeEvent] = []
    for detector, qubit in enumerate(measure_qubits):
        if rng.random() < event_probability:
            events.append(
                SyndromeEvent(
                    detector=detector,
                    x=qubit.x,
                    y=qubit.y,
                    t=rng.randrange(max(rounds, 1)),
                    stabilizer_type=qubit.stabilizer_type,
                )
            )

    matches: list[MatchEdge] = []
    detector_ids = [event.detector for event in events]
    for index in range(0, len(detector_ids), 2):
        if index + 1 < len(detector_ids):
            matches.append(MatchEdge(source=detector_ids[index], target=detector_ids[index + 1]))
        else:
            matches.append(MatchEdge(source=detector_ids[index], boundary=True))

    return SampleResponse(
        distance=distance,
        rounds=rounds,
        basis=basis,
        noise_p=noise.p,
        logical_observable_flip=False,
        mwpm_prediction=None,
        using_fallback=True,
        events=events,
        matches=matches,
        error=error,
    )


def sample_syndrome(
    *, distance: int, rounds: int, basis: Basis, noise: NoiseSettings, seed: int | None = None
) -> SampleResponse:
    try:
        _, pymatching = _import_quantum_stack()
        circuit = build_surface_code_circuit(
            distance=distance, rounds=rounds, basis=basis, noise=noise
        )
        sampler = detector_sampler(circuit, seed)
        detection_events, observable_flips = sampler.sample(1, separate_observables=True)
        syndrome = detection_events[0]
        coords = detector_coordinates(circuit)
        dem = circuit.detector_error_model(decompose_errors=True)
        matcher = pymatching.Matching.from_detector_error_model(dem)
        prediction = matcher.decode(syndrome)
        matched = matcher.decode_to_matched_dets_array(syndrome)

        layout = build_rotated_layout(distance, basis)
        bounds = layout.bounds
        max_x = max(bounds["max_x"], 1)
        max_y = max(bounds["max_y"], 1)
        events: list[SyndromeEvent] = []
        active_detectors = [int(i) for i, active in enumerate(syndrome) if bool(active)]
        for detector in active_detectors:
            coord = coords.get(detector, [])
            if len(coord) >= 2:
                x = float(coord[0])
                y = float(coord[1])
                t = float(coord[2]) if len(coord) >= 3 else 0.0
            else:
                x = float((detector * 1.7) % max_x)
                y = float((detector * 2.3) % max_y)
                t = float(detector % max(rounds, 1))
            events.append(SyndromeEvent(detector=detector, x=x, y=y, t=t))

        matches: list[MatchEdge] = []
        for pair in matched:
            source = int(pair[0])
            target = int(pair[1]) if len(pair) > 1 and int(pair[1]) >= 0 else None
            matches.append(MatchEdge(source=source, target=target, boundary=target is None))

        logical_flip = bool(observable_flips[0][0]) if observable_flips.shape[1] else False
        predicted_flip = bool(prediction[0]) if len(prediction) else False
        return SampleResponse(
            distance=distance,
            rounds=rounds,
            basis=basis,
            noise_p=noise.p,
            logical_observable_flip=logical_flip,
            mwpm_prediction=predicted_flip,
            events=events,
            matches=matches,
        )
    except Exception as exc:  # Fallback keeps the UI explorable without native deps.
        return fallback_sample(
            distance=distance,
            rounds=rounds,
            basis=basis,
            noise=noise,
            seed=seed,
            error=str(exc),
        )
