import type { BenchmarkResult } from "../types/api";

interface ResultsTableProps {
  results: BenchmarkResult[];
}

export function ResultsTable({ results }: ResultsTableProps) {
  return (
    <section className="panel table-panel">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Benchmark rows</p>
          <h2>Measurements</h2>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Decoder</th>
              <th>d</th>
              <th>LER</th>
              <th>Runtime</th>
              <th>Memory</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {results.length === 0 ? (
              <tr>
                <td colSpan={6}>Run a benchmark to populate measurements.</td>
              </tr>
            ) : (
              results.map((result, index) => (
                <tr key={`${result.decoder}-${result.distance}-${index}`}>
                  <td>{result.decoder.toUpperCase()}</td>
                  <td>{result.distance}</td>
                  <td>{formatRate(result.logical_error_rate)}</td>
                  <td>{formatRuntime(result.runtime_us_per_shot)}</td>
                  <td>{formatMemory(result.peak_memory_mb)}</td>
                  <td>{result.status.replace("_", " ")}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function formatRate(rate: number | null): string {
  if (rate === null) return "n/a";
  if (rate === 0) return "0";
  return rate < 0.001 ? rate.toExponential(2) : rate.toFixed(4);
}

function formatRuntime(runtime: number | null): string {
  if (runtime === null) return "n/a";
  return `${runtime.toFixed(2)} us/shot`;
}

function formatMemory(memory: number | null): string {
  if (memory === null) return "n/a";
  return `${memory.toFixed(2)} MB`;
}
