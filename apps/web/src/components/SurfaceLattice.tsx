import { motion } from "framer-motion";
import type { LayoutResponse, SampleResponse } from "../types/api";

interface SurfaceLatticeProps {
  layout: LayoutResponse | null;
  sample: SampleResponse | null;
}

const WIDTH = 680;
const HEIGHT = 520;
const PAD = 54;

export function SurfaceLattice({ layout, sample }: SurfaceLatticeProps) {
  if (!layout) {
    return (
      <section className="panel lattice-panel">
        <div className="empty-panel">Loading lattice</div>
      </section>
    );
  }

  const qubitById = new Map(layout.qubits.map((qubit) => [qubit.id, qubit]));
  const bounds = layout.bounds;
  const spanX = Math.max(bounds.max_x - bounds.min_x, 1);
  const spanY = Math.max(bounds.max_y - bounds.min_y, 1);
  const clamp = (value: number) => Math.min(Math.max(value, 0), 1);
  const x = (value: number) =>
    PAD + clamp((value - bounds.min_x) / spanX) * (WIDTH - PAD * 2);
  const y = (value: number) =>
    PAD + clamp((value - bounds.min_y) / spanY) * (HEIGHT - PAD * 2);
  const events = sample?.events ?? [];
  const checks = layout.qubits.filter((qubit) => qubit.kind === "measure");
  const eventAnchor = (event: (typeof events)[number]) => {
    if (checks.length === 0) {
      return { id: `event-${event.detector}`, x: event.x, y: event.y };
    }

    const byDetectorIndex = checks[event.detector % checks.length];
    const nearest = checks.reduce((best, check) => {
      const bestScore = (best.x - event.x) ** 2 + (best.y - event.y) ** 2;
      const score = (check.x - event.x) ** 2 + (check.y - event.y) ** 2;
      return score < bestScore ? check : best;
    }, byDetectorIndex);

    return { id: nearest.id, x: nearest.x, y: nearest.y };
  };
  const activeAnchors = Array.from(
    new Map(events.map((event) => {
      const anchor = eventAnchor(event);
      return [anchor.id, anchor];
    })).values()
  );

  return (
    <section className="panel lattice-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">{layout.surface17 ? "Surface-17" : `${layout.data_qubits + layout.measure_qubits} qubits`}</p>
          <h2>d={layout.distance} rotated lattice</h2>
          <p className="panel-note">Deterministic stabilizer checks; syndrome marks feed the decoder.</p>
        </div>
        <div className="metric-pair">
          <span>{layout.data_qubits} data</span>
          <span>{layout.measure_qubits} checks</span>
          <span>{events.length} detections</span>
        </div>
      </div>

      <motion.svg
        className="lattice-svg"
        role="img"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        initial={{ opacity: 0.5 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.35 }}
      >
        <g className="stabilizer-edges">
          {layout.edges.map((edge, index) => {
            const source = qubitById.get(edge.source);
            const target = qubitById.get(edge.target);
            if (!source || !target) return null;
            return (
              <line
                className={edge.kind === "logical" ? "logical-edge" : "check-edge"}
                key={`${edge.source}-${edge.target}-${index}`}
                x1={x(source.x)}
                x2={x(target.x)}
                y1={y(source.y)}
                y2={y(target.y)}
              />
            );
          })}
        </g>

        <g className="event-layer">
          {activeAnchors.map((anchor) => (
            <rect
              className="syndrome-halo"
              height="26"
              key={anchor.id}
              rx="8"
              width="26"
              x={x(anchor.x) - 13}
              y={y(anchor.y) - 13}
            />
          ))}
        </g>

        <g>
          {layout.qubits.map((qubit) => {
            const isMeasure = qubit.kind === "measure";
            const className = [
              "qubit",
              qubit.kind,
              qubit.stabilizer_type === "x" ? "x-check" : "",
              qubit.stabilizer_type === "z" ? "z-check" : "",
              qubit.boundary ? "boundary" : ""
            ]
              .filter(Boolean)
              .join(" ");
            const qx = x(qubit.x);
            const qy = y(qubit.y);
            return isMeasure ? (
              <g key={qubit.id}>
                <rect
                  className={className}
                  height="16"
                  rx="4"
                  width="16"
                  x={qx - 8}
                  y={qy - 8}
                />
                <text
                  className={`check-label ${
                    qubit.stabilizer_type === "x" ? "x-check-label" : "z-check-label"
                  }`}
                  dominantBaseline="central"
                  textAnchor="middle"
                  x={qx}
                  y={qy + 0.5}
                >
                  {qubit.stabilizer_type?.toUpperCase()}
                </text>
              </g>
            ) : (
              <circle className={className} cx={qx} cy={qy} key={qubit.id} r="6" />
            );
          })}
        </g>

      </motion.svg>

      <div className="lattice-legend" aria-label="Lattice symbol legend">
        <div className="legend-item">
          <i className="legend-data" />
          <span>
            <strong>Data qubit</strong>
            <small>Stores the logical state</small>
          </span>
        </div>
        <div className="legend-item">
          <i className="legend-check legend-x" />
          <span>
            <strong>X check</strong>
            <small>Detects phase-type changes</small>
          </span>
        </div>
        <div className="legend-item">
          <i className="legend-check legend-z" />
          <span>
            <strong>Z check</strong>
            <small>Detects bit-type changes</small>
          </span>
        </div>
        <div className="legend-item">
          <i className="legend-syndrome" />
          <span>
            <strong>Fired check</strong>
            <small>Pulse means outcome changed</small>
          </span>
        </div>
        <div className="legend-item">
          <i className="legend-coupling" />
          <span>
            <strong>Support</strong>
            <small>Qubits touched by a check</small>
          </span>
        </div>
        <div className="legend-item">
          <i className="legend-logical" />
          <span>
            <strong>Logical path</strong>
            <small>Reference boundary for failure</small>
          </span>
        </div>
      </div>

      <div className="lattice-explainer">
        <strong>Why this lattice matters</strong>
        <p>
          The grid is the surface-code patch: data qubits hold the encoded state, and
          nearby X/Z checks are measured repeatedly. Noise changes some check outcomes,
          producing syndrome events. The decoder uses that pattern to infer the most
          likely correction without directly measuring the logical state.
        </p>
      </div>
    </section>
  );
}
