import {
  Cpu,
  Network,
  Play,
  RefreshCw,
  Route,
  SlidersHorizontal,
  Sparkles
} from "lucide-react";
import type { Basis, DecoderName, HealthResponse } from "../types/api";

interface ControlRailProps {
  basis: Basis;
  busy: boolean;
  decoders: DecoderName[];
  distances: number[];
  health: HealthResponse | null;
  noiseP: number;
  seed: number;
  selectedDistance: number;
  shots: number;
  onBasisChange: (basis: Basis) => void;
  onDecoderToggle: (decoder: DecoderName) => void;
  onDistanceToggle: (distance: number) => void;
  onNoiseChange: (noise: number) => void;
  onRefreshSample: () => void;
  onRunBenchmark: () => void;
  onSeedChange: (seed: number) => void;
  onSelectedDistanceChange: (distance: number) => void;
  onShotsChange: (shots: number) => void;
}

const DISTANCES = [3, 5, 7];
const DECODERS: Array<{ id: DecoderName; label: string }> = [
  { id: "mwpm", label: "MWPM" },
  { id: "cnn", label: "CNN" },
  { id: "gnn", label: "GNN" },
  { id: "transformer", label: "Transformer" }
];

export function ControlRail(props: ControlRailProps) {
  const missing = props.health?.missing.join(", ") ?? "checking";
  const canRun = props.decoders.length > 0 && props.distances.length > 0 && !props.busy;

  return (
    <aside className="control-rail">
      <div className="brand-lockup">
        <p className="eyebrow">Surface-17 to d=7</p>
        <h1>Surface code lab</h1>
      </div>

      <div className="rail-section">
        <div className="section-title">
          <Route size={15} />
          <span>Code</span>
        </div>
        <div className="segmented">
          {(["x", "z"] as Basis[]).map((basis) => (
            <button
              className={props.basis === basis ? "selected" : ""}
              key={basis}
              type="button"
              onClick={() => props.onBasisChange(basis)}
            >
              memory {basis.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="button-grid">
          {DISTANCES.map((distance) => (
            <button
              className={props.distances.includes(distance) ? "selected" : ""}
              key={distance}
              type="button"
              onClick={() => props.onDistanceToggle(distance)}
              onMouseEnter={() => props.onSelectedDistanceChange(distance)}
            >
              d={distance}
            </button>
          ))}
        </div>
        <label className="field">
          <span>Preview distance</span>
          <select
            value={props.selectedDistance}
            onChange={(event) => props.onSelectedDistanceChange(Number(event.target.value))}
          >
            {DISTANCES.map((distance) => (
              <option key={distance} value={distance}>
                d={distance}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="rail-section">
        <div className="section-title">
          <SlidersHorizontal size={15} />
          <span>Noise and shots</span>
        </div>
        <label className="field">
          <span>Depolarizing p</span>
          <input
            min="0"
            max="0.02"
            step="0.0001"
            type="range"
            value={props.noiseP}
            onChange={(event) => props.onNoiseChange(Number(event.target.value))}
          />
          <strong>{props.noiseP.toFixed(4)}</strong>
        </label>
        <label className="field">
          <span>Shots</span>
          <input
            min="1"
            max="50000"
            step="100"
            type="number"
            value={props.shots}
            onChange={(event) => props.onShotsChange(Number(event.target.value))}
          />
        </label>
        <label className="field">
          <span>Seed</span>
          <input
            min="0"
            step="1"
            type="number"
            value={props.seed}
            onChange={(event) => props.onSeedChange(Number(event.target.value))}
          />
        </label>
      </div>

      <div className="rail-section">
        <div className="section-title">
          <Cpu size={15} />
          <span>Decoders</span>
        </div>
        <div className="decoder-toggles">
          {DECODERS.map((decoder) => (
            <button
              className={props.decoders.includes(decoder.id) ? "selected" : ""}
              key={decoder.id}
              type="button"
              onClick={() => props.onDecoderToggle(decoder.id)}
            >
              {decoder.id === "transformer" ? <Sparkles size={14} /> : <Network size={14} />}
              {decoder.label}
            </button>
          ))}
        </div>
      </div>

      <div className="rail-actions">
        <button className="primary-action" disabled={!canRun} type="button" onClick={props.onRunBenchmark}>
          <Play size={16} />
          {props.busy ? "Running" : "Run benchmark"}
        </button>
        <button className="secondary-action" type="button" onClick={props.onRefreshSample}>
          <RefreshCw size={16} />
          New sample
        </button>
      </div>

      <div className="rail-footer">
        <span className={props.health?.quantum_stack ? "dot ready" : "dot"} />
        <span>{props.health?.quantum_stack ? "Stim + PyMatching available" : missing}</span>
      </div>
    </aside>
  );
}
