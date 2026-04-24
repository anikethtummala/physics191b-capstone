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
  const eventByDetector = new Map(sample?.events.map((event) => [event.detector, event]) ?? []);
  const bounds = layout.bounds;
  const spanX = Math.max(bounds.max_x - bounds.min_x, 1);
  const spanY = Math.max(bounds.max_y - bounds.min_y, 1);
  const x = (value: number) => PAD + ((value - bounds.min_x) / spanX) * (WIDTH - PAD * 2);
  const y = (value: number) => PAD + ((value - bounds.min_y) / spanY) * (HEIGHT - PAD * 2);

  return (
    <section className="panel lattice-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">{layout.surface17 ? "Surface-17" : `${layout.data_qubits + layout.measure_qubits} qubits`}</p>
          <h2>d={layout.distance} rotated lattice</h2>
        </div>
        <div className="metric-pair">
          <span>{layout.data_qubits} data</span>
          <span>{layout.measure_qubits} checks</span>
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
              <rect
                className={className}
                height="16"
                key={qubit.id}
                rx="4"
                width="16"
                x={qx - 8}
                y={qy - 8}
              />
            ) : (
              <circle className={className} cx={qx} cy={qy} key={qubit.id} r="7.5" />
            );
          })}
        </g>

        <g className="match-layer">
          {sample?.matches.map((match, index) => {
            const source = eventByDetector.get(match.source);
            const target = match.target === null ? null : eventByDetector.get(match.target);
            if (!source) return null;
            const sourceX = x(source.x);
            const sourceY = y(source.y);
            const targetX = target ? x(target.x) : sourceX + 34;
            const targetY = target ? y(target.y) : sourceY - 34;
            return (
              <line
                className="match-edge"
                key={`${match.source}-${match.target ?? "b"}-${index}`}
                x1={sourceX}
                x2={targetX}
                y1={sourceY}
                y2={targetY}
              />
            );
          })}
        </g>

        <g className="event-layer">
          {sample?.events.map((event) => (
            <g key={event.detector}>
              <circle
                className="syndrome-ring"
                cx={x(event.x)}
                cy={y(event.y)}
                r={15 + Math.min(event.t, 5)}
              />
              <circle className="syndrome-dot" cx={x(event.x)} cy={y(event.y)} r="5.5" />
            </g>
          ))}
        </g>
      </motion.svg>
    </section>
  );
}
