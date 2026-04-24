import { BrainCircuit, GitBranch, Network, Workflow } from "lucide-react";
import type { BenchmarkResult, DecoderName } from "../types/api";

interface DecoderStripProps {
  decoders: DecoderName[];
  results: BenchmarkResult[];
}

const LABELS: Record<DecoderName, string> = {
  mwpm: "MWPM",
  cnn: "CNN",
  gnn: "GNN",
  transformer: "Transformer"
};

const ICONS: Record<DecoderName, typeof Workflow> = {
  mwpm: Workflow,
  cnn: BrainCircuit,
  gnn: GitBranch,
  transformer: Network
};

export function DecoderStrip({ decoders, results }: DecoderStripProps) {
  return (
    <section className="panel decoder-strip">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Architecture set</p>
          <h2>Decoder status</h2>
        </div>
      </div>
      <div className="decoder-cards">
        {decoders.map((decoder) => {
          const latest = [...results].reverse().find((result) => result.decoder === decoder);
          const Icon = ICONS[decoder];
          const status = latest?.status ?? "ready";
          return (
            <div className="decoder-card" key={decoder}>
              <div className="decoder-name">
                <Icon size={16} />
                <strong>{LABELS[decoder]}</strong>
              </div>
              <span className={`status-chip ${status}`}>{status.replace("_", " ")}</span>
              <p>
                {latest?.status === "complete"
                  ? `${formatRate(latest.logical_error_rate)} logical`
                  : latest?.error ?? "queued for comparison"}
              </p>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function formatRate(rate: number | null): string {
  if (rate === null) return "n/a";
  if (rate === 0) return "0";
  return rate < 0.001 ? rate.toExponential(2) : rate.toFixed(4);
}
