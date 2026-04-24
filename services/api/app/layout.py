from __future__ import annotations

from .models import Basis, Edge, LayoutResponse, Qubit


def build_rotated_layout(distance: int, basis: Basis = "x") -> LayoutResponse:
    if distance < 3 or distance > 7 or distance % 2 == 0:
        raise ValueError("Distance must be an odd integer in the range 3..7.")

    qubits: list[Qubit] = []
    edges: list[Edge] = []
    data_ids: dict[tuple[int, int], str] = {}
    index = 0

    for row in range(distance):
        for col in range(distance):
            qid = f"d{row}_{col}"
            data_ids[(row, col)] = qid
            qubits.append(
                Qubit(id=qid, index=index, kind="data", x=col * 2 + 1, y=row * 2 + 1)
            )
            index += 1

    def stabilizer_type(row: int, col: int, boundary_axis: str | None = None) -> str:
        if boundary_axis == "horizontal":
            return "x" if col % 2 == 0 else "z"
        if boundary_axis == "vertical":
            return "z" if row % 2 == 0 else "x"
        return "x" if (row + col) % 2 == 0 else "z"

    for row in range(distance - 1):
        for col in range(distance - 1):
            qid = f"s{row}_{col}"
            stype = stabilizer_type(row, col)
            qubits.append(
                Qubit(
                    id=qid,
                    index=index,
                    kind="measure",
                    stabilizer_type=stype,  # type: ignore[arg-type]
                    x=col * 2 + 2,
                    y=row * 2 + 2,
                )
            )
            index += 1
            for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
                edges.append(
                    Edge(source=qid, target=data_ids[(row + dr, col + dc)], kind="stabilizer")
                )

    for col in range(distance - 1):
        for boundary_name, row, y in (("top", 0, 0.2), ("bottom", distance - 1, distance * 2 - 0.2)):
            qid = f"{boundary_name}_{col}"
            stype = stabilizer_type(row, col, "horizontal")
            qubits.append(
                Qubit(
                    id=qid,
                    index=index,
                    kind="measure",
                    stabilizer_type=stype,  # type: ignore[arg-type]
                    x=col * 2 + 2,
                    y=y,
                    boundary=True,
                )
            )
            index += 1
            edges.append(Edge(source=qid, target=data_ids[(row, col)], kind="stabilizer"))
            edges.append(Edge(source=qid, target=data_ids[(row, col + 1)], kind="stabilizer"))

    logical_col = 0 if basis == "x" else distance - 1
    logical_rows = list(range(distance))
    for a, b in zip(logical_rows, logical_rows[1:]):
        edges.append(
            Edge(
                source=data_ids[(a, logical_col)],
                target=data_ids[(b, logical_col)],
                kind="logical",
            )
        )

    bounds = {"min_x": 0, "min_y": 0, "max_x": distance * 2, "max_y": distance * 2}
    measure_qubits = len([q for q in qubits if q.kind == "measure"])
    return LayoutResponse(
        distance=distance,
        basis=basis,
        surface17=distance == 3,
        data_qubits=distance * distance,
        measure_qubits=measure_qubits,
        bounds=bounds,
        qubits=qubits,
        edges=edges,
    )
