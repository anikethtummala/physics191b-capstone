import { useCallback, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Activity, Database, RefreshCw, Terminal } from "lucide-react";
import {
  fetchBenchmark,
  fetchHealth,
  fetchLayout,
  fetchSample,
  startBenchmark
} from "./api";
import { buildBenchmarkRequest } from "./benchmark";
import { BenchmarkChart } from "./components/BenchmarkChart";
import { ControlRail } from "./components/ControlRail";
import { DecoderStrip } from "./components/DecoderStrip";
import { ResultsTable } from "./components/ResultsTable";
import { SurfaceLattice } from "./components/SurfaceLattice";
import type {
  Basis,
  BenchmarkJob,
  DecoderName,
  HealthResponse,
  LayoutResponse,
  SampleResponse
} from "./types/api";

const DEFAULT_DECODERS: DecoderName[] = ["mwpm", "cnn", "gnn", "transformer"];

function App() {
  const [basis, setBasis] = useState<Basis>("x");
  const [selectedDistance, setSelectedDistance] = useState(3);
  const [distances, setDistances] = useState<number[]>([3, 5, 7]);
  const [noiseP, setNoiseP] = useState(0.001);
  const [shots, setShots] = useState(1000);
  const [seed, setSeed] = useState(1337);
  const [decoders, setDecoders] = useState<DecoderName[]>(DEFAULT_DECODERS);
  const [layout, setLayout] = useState<LayoutResponse | null>(null);
  const [sample, setSample] = useState<SampleResponse | null>(null);
  const [job, setJob] = useState<BenchmarkJob | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [uiError, setUiError] = useState<string | null>(null);

  const activeRequest = useMemo(
    () =>
      buildBenchmarkRequest({
        distances,
        basis,
        noiseP,
        shots,
        seed,
        decoders
      }),
    [basis, decoders, distances, noiseP, seed, shots]
  );

  const refreshSample = useCallback(async () => {
    try {
      const next = await fetchSample({
        distance: selectedDistance,
        basis,
        noise: { p: noiseP },
        seed
      });
      setSample(next);
    } catch (error) {
      setUiError(error instanceof Error ? error.message : "Sample request failed.");
    }
  }, [basis, noiseP, seed, selectedDistance]);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() =>
        setHealth({ ok: false, quantum_stack: false, missing: ["api offline"] })
      );
  }, []);

  useEffect(() => {
    fetchLayout(selectedDistance, basis)
      .then(setLayout)
      .catch((error) =>
        setUiError(error instanceof Error ? error.message : "Layout request failed.")
      );
  }, [basis, selectedDistance]);

  useEffect(() => {
    refreshSample();
  }, [refreshSample]);

  async function runBenchmark() {
    setBusy(true);
    setUiError(null);
    try {
      const jobId = await startBenchmark(activeRequest);
      setJob({
        job_id: jobId,
        status: "pending",
        progress: 0,
        results: [],
        errors: []
      });
      for (;;) {
        const next = await fetchBenchmark(jobId);
        setJob(next);
        if (next.status === "complete" || next.status === "error") {
          break;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 500));
      }
    } catch (error) {
      setUiError(error instanceof Error ? error.message : "Benchmark request failed.");
    } finally {
      setBusy(false);
      fetchHealth().then(setHealth).catch(() => undefined);
    }
  }

  function toggleDistance(distance: number) {
    setDistances((current) =>
      current.includes(distance)
        ? current.filter((value) => value !== distance)
        : [...current, distance].sort((a, b) => a - b)
    );
    setSelectedDistance(distance);
  }

  function toggleDecoder(decoder: DecoderName) {
    setDecoders((current) =>
      current.includes(decoder)
        ? current.filter((value) => value !== decoder)
        : [...current, decoder]
    );
  }

  return (
    <main className="app-shell">
      <ControlRail
        basis={basis}
        busy={busy}
        decoders={decoders}
        distances={distances}
        health={health}
        noiseP={noiseP}
        seed={seed}
        selectedDistance={selectedDistance}
        shots={shots}
        onBasisChange={setBasis}
        onDecoderToggle={toggleDecoder}
        onDistanceToggle={toggleDistance}
        onNoiseChange={setNoiseP}
        onRefreshSample={refreshSample}
        onRunBenchmark={runBenchmark}
        onSeedChange={setSeed}
        onSelectedDistanceChange={setSelectedDistance}
        onShotsChange={setShots}
      />

      <section className="workspace">
        <motion.div
          className="topline"
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
        >
          <div>
            <p className="eyebrow">Deterministic checks, probabilistic decoding</p>
            <h1>Decoder benchmark</h1>
          </div>
          <div className="status-row">
            <span className="status-pill">
              <Activity size={14} />
              {job ? `${Math.round(job.progress * 100)}%` : "idle"}
            </span>
            <span className="status-pill">
              <Database size={14} />
              {health?.quantum_stack ? "quantum stack ready" : "deps pending"}
            </span>
          </div>
        </motion.div>

        {uiError && (
          <div className="inline-error">
            <Terminal size={14} />
            <span>{uiError}</span>
          </div>
        )}

        <div className="primary-grid">
          <SurfaceLattice layout={layout} sample={sample} />
          <div className="right-stack">
            <DecoderStrip decoders={DEFAULT_DECODERS} results={job?.results ?? []} />
            <BenchmarkChart results={job?.results ?? []} />
          </div>
        </div>

        <div className="lower-grid">
          <ResultsTable results={job?.results ?? []} />
          <div className="panel compact-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Current sample</p>
                <h2>Syndrome frame</h2>
              </div>
              <button className="icon-button" type="button" onClick={refreshSample}>
                <RefreshCw size={16} />
              </button>
            </div>
            <div className="sample-stats">
              <span>{sample?.events.length ?? 0} detections</span>
              <span>{sample?.matches.length ?? 0} decoder pairs</span>
              <span>{sample?.logical_observable_flip ? "logical flip" : "logical stable"}</span>
              <span>
                {sample ? (sample.using_fallback ? "visual fallback" : "stim sample") : "loading"}
              </span>
            </div>
            <p className="quiet-text">
              Stabilizer extraction is fixed by the circuit. The decoder operates on the
              measured syndrome and predicts the logical-frame update.
            </p>
            {sample?.error && <p className="quiet-text">{sample.error}</p>}
          </div>
        </div>
      </section>
    </main>
  );
}

export default App;
