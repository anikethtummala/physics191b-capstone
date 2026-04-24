import type { BenchmarkResult, DecoderName } from "../types/api";

interface BenchmarkChartProps {
  results: BenchmarkResult[];
}

const COLORS: Record<DecoderName, string> = {
  mwpm: "#1f1f1b",
  cnn: "#3f6f63",
  gnn: "#355f8a",
  transformer: "#9a4f36"
};

const WIDTH = 620;
const HEIGHT = 260;
const PAD_X = 52;
const PAD_Y = 34;

export function BenchmarkChart({ results }: BenchmarkChartProps) {
  const complete = results.filter(
    (result) => result.status === "complete" && result.logical_error_rate !== null
  );
  const distances = [...new Set(results.map((result) => result.distance))].sort((a, b) => a - b);
  const decoders = [...new Set(complete.map((result) => result.decoder))];
  const maxRate = Math.max(...complete.map((result) => result.logical_error_rate ?? 0), 0.001);
  const minDistance = distances[0] ?? 3;
  const maxDistance = distances[distances.length - 1] ?? 7;
  const spanDistance = Math.max(maxDistance - minDistance, 1);
  const px = (distance: number) =>
    PAD_X + ((distance - minDistance) / spanDistance) * (WIDTH - PAD_X * 2);
  const py = (rate: number) =>
    HEIGHT - PAD_Y - (rate / maxRate) * (HEIGHT - PAD_Y * 2);

  return (
    <section className="panel chart-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Logical error rate</p>
          <h2>Distance sweep</h2>
        </div>
      </div>
      <svg className="chart-svg" viewBox={`0 0 ${WIDTH} ${HEIGHT}`}>
        <line className="axis" x1={PAD_X} x2={WIDTH - PAD_X} y1={HEIGHT - PAD_Y} y2={HEIGHT - PAD_Y} />
        <line className="axis" x1={PAD_X} x2={PAD_X} y1={PAD_Y} y2={HEIGHT - PAD_Y} />
        {[0, 0.5, 1].map((tick) => {
          const y = HEIGHT - PAD_Y - tick * (HEIGHT - PAD_Y * 2);
          return (
            <g key={tick}>
              <line className="grid-line" x1={PAD_X} x2={WIDTH - PAD_X} y1={y} y2={y} />
              <text x={12} y={y + 4}>
                {(maxRate * tick).toPrecision(2)}
              </text>
            </g>
          );
        })}
        {distances.map((distance) => (
          <text className="x-label" key={distance} x={px(distance)} y={HEIGHT - 8}>
            d={distance}
          </text>
        ))}
        {decoders.map((decoder) => {
          const points = complete
            .filter((result) => result.decoder === decoder)
            .sort((a, b) => a.distance - b.distance);
          const path = points
            .map((point, index) => {
              const command = index === 0 ? "M" : "L";
              return `${command}${px(point.distance)},${py(point.logical_error_rate ?? 0)}`;
            })
            .join(" ");
          return (
            <g key={decoder}>
              <path className="chart-line" d={path} stroke={COLORS[decoder]} />
              {points.map((point) => (
                <circle
                  className="chart-point"
                  cx={px(point.distance)}
                  cy={py(point.logical_error_rate ?? 0)}
                  fill={COLORS[decoder]}
                  key={`${decoder}-${point.distance}`}
                  r="4"
                />
              ))}
            </g>
          );
        })}
      </svg>
      <div className="legend-row">
        {(Object.keys(COLORS) as DecoderName[]).map((decoder) => (
          <span key={decoder}>
            <i style={{ background: COLORS[decoder] }} />
            {decoder.toUpperCase()}
          </span>
        ))}
      </div>
      {complete.length === 0 && <p className="quiet-text">No complete decoder rows yet.</p>}
    </section>
  );
}
