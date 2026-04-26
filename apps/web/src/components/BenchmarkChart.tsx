import type { BenchmarkResult, DecoderName } from "../types/api";

interface BenchmarkChartProps {
  noiseP: number;
  results: BenchmarkResult[];
  shots: number;
}

const COLORS: Record<DecoderName, string> = {
  mwpm: "#1f1f1b",
  cnn: "#3f6f63",
  gnn: "#355f8a",
  transformer: "#9a4f36"
};

const WIDTH = 620;
const HEIGHT = 290;
const PAD_X = 74;
const PAD_Y = 38;

export function BenchmarkChart({ noiseP, results, shots }: BenchmarkChartProps) {
  const complete = results.filter(
    (result) => result.status === "complete" && result.logical_error_rate !== null
  );
  const distances = [...new Set(results.map((result) => result.distance))].sort((a, b) => a - b);
  const decoders = [...new Set(complete.map((result) => result.decoder))];
  const pointStats = complete.map((result) => ({
    result,
    ci: getConfidenceInterval(result)
  }));
  const maxRate = Math.max(
    ...pointStats.map((point) => point.ci.upper),
    ...complete.map((result) => result.logical_error_rate ?? 0),
    0.001
  );
  const minDistance = distances[0] ?? 3;
  const maxDistance = distances[distances.length - 1] ?? 7;
  const spanDistance = Math.max(maxDistance - minDistance, 1);
  const px = (distance: number) =>
    PAD_X + ((distance - minDistance) / spanDistance) * (WIDTH - PAD_X * 2);
  const py = (rate: number) =>
    HEIGHT - PAD_Y - (rate / maxRate) * (HEIGHT - PAD_Y * 2);
  const roundsLabel = getRoundsLabel(results);

  return (
    <section className="panel chart-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Logical error rate</p>
          <h2>Distance sweep</h2>
          <p className="panel-note">
            Lower is better. Error bars show approximate 95% binomial confidence intervals.
          </p>
        </div>
      </div>
      <div className="chart-config">
        <span>p={noiseP.toFixed(4)}</span>
        <span>{shots.toLocaleString()} requested shots</span>
        <span>{roundsLabel}</span>
      </div>
      <svg className="chart-svg" viewBox={`0 0 ${WIDTH} ${HEIGHT}`}>
        <line className="axis" x1={PAD_X} x2={WIDTH - PAD_X} y1={HEIGHT - PAD_Y} y2={HEIGHT - PAD_Y} />
        <line className="axis" x1={PAD_X} x2={PAD_X} y1={PAD_Y} y2={HEIGHT - PAD_Y} />
        <text className="axis-label y-axis-label" transform={`translate(14 ${HEIGHT / 2}) rotate(-90)`}>
          logical error rate
        </text>
        <text className="axis-label x-axis-label" x={WIDTH / 2} y={HEIGHT - 8}>
          code distance (d)
        </text>
        {[0, 0.5, 1].map((tick) => {
          const y = HEIGHT - PAD_Y - tick * (HEIGHT - PAD_Y * 2);
          return (
            <g key={tick}>
              <line className="grid-line" x1={PAD_X} x2={WIDTH - PAD_X} y1={y} y2={y} />
              <text x={30} y={y + 4}>
                {(maxRate * tick).toPrecision(2)}
              </text>
            </g>
          );
        })}
        {distances.map((distance) => (
          <text className="x-label" key={distance} x={px(distance)} y={HEIGHT - 28}>
            {formatDistanceTick(distance, results)}
          </text>
        ))}
        {decoders.map((decoder) => {
          const points = pointStats
            .filter((point) => point.result.decoder === decoder)
            .sort((a, b) => a.result.distance - b.result.distance);
          const path = points
            .map((point, index) => {
              const command = index === 0 ? "M" : "L";
              return `${command}${px(point.result.distance)},${py(point.result.logical_error_rate ?? 0)}`;
            })
            .join(" ");
          return (
            <g key={decoder}>
              <path className="chart-line" d={path} stroke={COLORS[decoder]} />
              {points.map((point) => (
                <g key={`${decoder}-${point.result.distance}`}>
                  <line
                    className="error-bar"
                    stroke={COLORS[decoder]}
                    x1={px(point.result.distance)}
                    x2={px(point.result.distance)}
                    y1={py(point.ci.lower)}
                    y2={py(point.ci.upper)}
                  />
                  <line
                    className="error-cap"
                    stroke={COLORS[decoder]}
                    x1={px(point.result.distance) - 5}
                    x2={px(point.result.distance) + 5}
                    y1={py(point.ci.lower)}
                    y2={py(point.ci.lower)}
                  />
                  <line
                    className="error-cap"
                    stroke={COLORS[decoder]}
                    x1={px(point.result.distance) - 5}
                    x2={px(point.result.distance) + 5}
                    y1={py(point.ci.upper)}
                    y2={py(point.ci.upper)}
                  />
                  <circle
                    className="chart-point"
                    cx={px(point.result.distance)}
                    cy={py(point.result.logical_error_rate ?? 0)}
                    fill={COLORS[decoder]}
                    r="4"
                  />
                </g>
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

function getConfidenceInterval(result: BenchmarkResult): { lower: number; upper: number } {
  if (
    result.logical_error_rate_ci_low !== undefined &&
    result.logical_error_rate_ci_low !== null &&
    result.logical_error_rate_ci_high !== undefined &&
    result.logical_error_rate_ci_high !== null
  ) {
    return {
      lower: result.logical_error_rate_ci_low,
      upper: result.logical_error_rate_ci_high
    };
  }

  return confidenceInterval(result.logical_errors, result.shots, result.logical_error_rate ?? 0);
}

function confidenceInterval(
  logicalErrors: number | null,
  shots: number,
  fallbackRate: number
): { lower: number; upper: number } {
  if (shots <= 0) {
    return { lower: fallbackRate, upper: fallbackRate };
  }

  const successes = logicalErrors ?? Math.round(fallbackRate * shots);
  const z = 1.96;
  const phat = successes / shots;
  const denominator = 1 + (z * z) / shots;
  const center = (phat + (z * z) / (2 * shots)) / denominator;
  const margin =
    (z * Math.sqrt((phat * (1 - phat)) / shots + (z * z) / (4 * shots * shots))) /
    denominator;
  return {
    lower: Math.max(0, center - margin),
    upper: Math.min(1, center + margin)
  };
}

function getRoundsLabel(results: BenchmarkResult[]): string {
  const complete = results.filter((result) => result.status === "complete");
  if (complete.length === 0) {
    return "rounds default to d";
  }

  const usesDistanceRounds = complete.every((result) => result.rounds === result.distance);
  if (usesDistanceRounds) {
    return "rounds = d";
  }

  const rounds = [...new Set(complete.map((result) => result.rounds))];
  return rounds.length === 1 ? `rounds = ${rounds[0]}` : "mixed rounds";
}

function formatDistanceTick(distance: number, results: BenchmarkResult[]): string {
  const row = results.find((result) => result.distance === distance);
  return row ? `d=${distance} r=${row.rounds}` : `d=${distance}`;
}
